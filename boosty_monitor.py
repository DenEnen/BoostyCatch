import os
import json
import asyncio
import sys
import aiohttp
import hashlib
import re
import random
import subprocess
import requests
import time
from pathlib import Path
from typing import Optional, Dict, Tuple, List
from dotenv import load_dotenv
from twitchio.ext import commands
from playwright.async_api import async_playwright, Browser, Page
from loguru import logger
from bs4 import BeautifulSoup

# Настройка логирования
logger.remove()
logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")

# Загрузка переменных окружения
load_dotenv()

# Конфигурация
TWITCH_CHANNEL = os.getenv('TWITCH_CHANNEL')
TWITCH_TOKEN = os.getenv('TWITCH_TOKEN')
WEBSHARE_TOKEN = os.getenv('WEBSHARE_TOKEN')
COOKIES_PATH = Path('boosty_cookies.json')
SCREENSHOTS_DIR = Path('screenshots')
SCREENSHOTS_DIR.mkdir(exist_ok=True)
API_BASE_URL = "https://gachi.gay/api"
# Новые настройки для видео
ENABLE_VIDEO_RECORDING = os.getenv('ENABLE_VIDEO_RECORDING', 'true').lower() == 'true'
PAGE_TIMEOUT = int(os.getenv('PAGE_TIMEOUT', '60000'))  # Увеличенный таймаут (60 секунд)
MAX_RETRIES = int(os.getenv('MAX_RETRIES', '3'))
FRAME_RATE = int(os.getenv('FRAME_RATE', '60'))  # Кадров в секунду для записи
VIDEO_MAX_FRAMES = int(os.getenv('VIDEO_MAX_FRAMES', '3600'))  # Максимальное количество кадров

# Импорт необходимых библиотек
try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("Библиотека opencv-python не установлена. Будет использована альтернативная запись видео.")

class WebshareProxyManager:
    def __init__(self):
        self.token = WEBSHARE_TOKEN
        self.proxies = []
        self.current_proxy = None
        self.last_update = 0
        self.update_interval = 300  # 5 минут
        self.base_url = "https://proxy.webshare.io/api/v2"
        self.proxy_check_task = None
        # Авторизуем IP при инициализации
        self.authorize_ip()
        # Запускаем задачу проверки прокси
        self.start_proxy_check_task()

    def get_current_ip(self) -> Optional[str]:
        """Получение текущего IP адреса"""
        try:
            response = requests.get("https://api.ipify.org?format=json")
            if response.status_code == 200:
                return response.json()["ip"]
            else:
                logger.error(f"Ошибка при получении IP: {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"Ошибка при получении IP: {str(e)}")
            return None

    def authorize_ip(self) -> bool:
        """Авторизация текущего IP адреса"""
        try:
            current_ip = self.get_current_ip()
            if not current_ip:
                return False

            headers = {
                'Authorization': f'Token {self.token}',
                'Content-Type': 'application/json'
            }

            # Проверяем, не авторизован ли уже этот IP
            response = requests.get(
                f"{self.base_url}/proxy/ipauthorization/",
                headers=headers
            )

            if response.status_code == 200:
                authorized_ips = response.json().get("results", [])
                for ip_auth in authorized_ips:
                    if ip_auth["ip_address"] == current_ip:
                        logger.info(f"IP {current_ip} уже авторизован")
                        return True

            # Авторизуем новый IP
            response = requests.post(
                f"{self.base_url}/proxy/ipauthorization/",
                headers=headers,
                json={"ip_address": current_ip}
            )

            if response.status_code in [200, 201]:
                logger.success(f"IP {current_ip} успешно авторизован")
                return True
            else:
                logger.error(f"Ошибка при авторизации IP: {response.status_code}, {response.text}")
                return False

        except Exception as e:
            logger.error(f"Ошибка при авторизации IP: {str(e)}")
            return False

    async def check_proxy_speed(self, proxy: Dict[str, str]) -> float:
        """Проверка скорости подключения к boosty.to через прокси"""
        try:
            proxy_url = f"http://{proxy['username']}:{proxy['password']}@{proxy['proxy_address']}:{proxy['port']}"
            start_time = asyncio.get_event_loop().time()
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "https://boosty.to",
                    proxy=proxy_url,
                    timeout=10
                ) as response:
                    if response.status == 200:
                        end_time = asyncio.get_event_loop().time()
                        speed = end_time - start_time
                        logger.info(f"Скорость прокси {proxy['proxy_address']}: {speed:.2f} сек")
                        return speed
                    else:
                        logger.warning(f"Прокси {proxy['proxy_address']} вернул статус {response.status}")
                        return float('inf')
        except Exception as e:
            logger.error(f"Ошибка при проверке скорости прокси {proxy['proxy_address']}: {str(e)}")
            return float('inf')

    def get_proxies(self) -> List[Dict[str, str]]:
        """Получение списка прокси с API Webshare.io"""
        try:
            # Проверяем авторизацию IP перед получением прокси
            if not self.authorize_ip():
                logger.warning("Не удалось авторизовать IP, пробуем получить прокси без авторизации")

            headers = {
                'Authorization': f'Token {self.token}'
            }
            
            # Получаем список прокси с дополнительными параметрами
            response = requests.get(
                f"{self.base_url}/proxy/list/",
                headers=headers,
                params={
                    'mode': 'direct',
                    'valid': 'true',
                    'ordering': '-last_verification'  # Сначала самые свежие прокси
                }
            )
            
            if response.status_code == 200:
                proxy_data = response.json()
                self.proxies = []
                
                for proxy in proxy_data.get('results', []):
                    if proxy.get('valid', False):
                        self.proxies.append({
                            'proxy_address': proxy['proxy_address'],
                            'port': proxy['port'],
                            'username': proxy['username'],
                            'password': proxy['password'],
                            'country_code': proxy.get('country_code', ''),
                            'city_name': proxy.get('city_name', ''),
                            'last_verification': proxy.get('last_verification', '')
                        })
                
                self.last_update = asyncio.get_event_loop().time()
                logger.info(f"Получено {len(self.proxies)} прокси")
                return self.proxies
            else:
                logger.error(f"Ошибка при получении прокси: {response.status_code}")
                return []
        except Exception as e:
            logger.error(f"Ошибка при получении прокси: {str(e)}")
            return []

    async def get_random_proxy(self) -> Optional[Dict[str, str]]:
        """Получение случайного прокси из списка с проверкой скорости"""
        current_time = asyncio.get_event_loop().time()
        
        # Обновляем список прокси если прошло достаточно времени
        if current_time - self.last_update > self.update_interval:
            self.get_proxies()
        
        if not self.proxies:
            self.get_proxies()
        
        if not self.proxies:
            return None

        # Проверяем скорость для каждого прокси
        proxy_speeds = []
        for proxy in self.proxies:
            speed = await self.check_proxy_speed(proxy)
            proxy_speeds.append((proxy, speed))

        # Сортируем прокси по скорости и берем самый быстрый
        proxy_speeds.sort(key=lambda x: x[1])
        self.current_proxy = proxy_speeds[0][0]
        
        logger.info(f"Выбран прокси {self.current_proxy['proxy_address']} со скоростью {proxy_speeds[0][1]:.2f} сек")
        return self.current_proxy

    def start_proxy_check_task(self):
        """Запуск периодической задачи проверки прокси"""
        if not self.proxy_check_task:
            self.proxy_check_task = asyncio.create_task(self.periodic_proxy_check())
            logger.info("Запущена периодическая проверка прокси")

    async def stop_proxy_check_task(self):
        """Остановка периодической задачи проверки прокси"""
        if self.proxy_check_task:
            self.proxy_check_task.cancel()
            try:
                await self.proxy_check_task
            except asyncio.CancelledError:
                pass
            self.proxy_check_task = None
            logger.info("Остановлена периодическая проверка прокси")

    async def periodic_proxy_check(self):
        """Периодическая проверка прокси каждые 2 часа"""
        while True:
            try:
                logger.info("Начало проверки прокси...")
                # Получаем свежий список прокси
                proxies = self.get_proxies()
                
                if proxies:
                    working_proxies = []
                    for proxy in proxies:
                        speed = await self.check_proxy_speed(proxy)
                        if speed != float('inf'):
                            working_proxies.append((proxy, speed))
                            logger.info(f"Прокси {proxy['proxy_address']} работает, скорость: {speed:.2f} сек")
                        else:
                            logger.warning(f"Прокси {proxy['proxy_address']} не работает")
                    
                    # Обновляем список прокси только работающими
                    self.proxies = [proxy for proxy, _ in working_proxies]
                    logger.info(f"Проверка завершена. Работающих прокси: {len(self.proxies)}")
                else:
                    logger.error("Не удалось получить список прокси для проверки")
                
                # Ждем 2 часа перед следующей проверкой
                await asyncio.sleep(7200)  # 2 часа = 7200 секунд
                
            except asyncio.CancelledError:
                logger.info("Задача проверки прокси остановлена")
                break
            except Exception as e:
                logger.error(f"Ошибка при проверке прокси: {str(e)}")
                await asyncio.sleep(5)  # Ждем 5 секунд перед повторной попыткой при ошибке

class ScreenshotManager:
    def __init__(self):
        self.session = None
        self.uploaded_files = []
        self.videos_dir = Path('videos')
        self.videos_dir.mkdir(exist_ok=True)
        self.temp_dir = Path('temp_screenshots')
        self.temp_dir.mkdir(exist_ok=True)
        self.frames = []
        self.recording = False
        self.frame_count = 0

    async def init_session(self):
        if not self.session:
            self.session = aiohttp.ClientSession()

    async def close_session(self):
        if self.session:
            await self.session.close()
            self.session = None

    async def upload_file(self, file_path: Path, prefix: str) -> Optional[str]:
        """Загрузка файла через API"""
        if not self.session:
            await self.init_session()

        try:
            # Получаем данные файла
            with open(file_path, 'rb') as f:
                file_data = f.read()

            # Подготавливаем данные для загрузки
            filename = file_path.name
            content_type = 'video/webm' if filename.endswith('.webm') else 'image/png'
            
            data = aiohttp.FormData()
            data.add_field('file', file_data, filename=filename, content_type=content_type)

            # Отправляем файл
            async with self.session.post(f"{API_BASE_URL}/upload", data=data) as response:
                if response.status == 200:
                    result = await response.json()
                    self.uploaded_files.append(result['key'])
                    logger.info(f"Файл загружен: {result['link']}")
                    
                    # Удаляем локальный файл после успешной загрузки
                    try:
                        file_path.unlink()
                        logger.debug(f"Локальный файл удален: {file_path}")
                    except Exception as e:
                        logger.warning(f"Не удалось удалить локальный файл: {str(e)}")
                        
                    return result['link']
                else:
                    logger.error(f"Ошибка при загрузке файла: {response.status}")
                    return None

        except Exception as e:
            logger.error(f"Ошибка при загрузке файла: {str(e)}")
            return None

    async def upload_screenshot(self, screenshot_data: bytes, prefix: str) -> Optional[str]:
        """Загрузка скриншота через API"""
        if not self.session:
            await self.init_session()

        try:
            # Создаем временный файл в памяти
            filename = f"{prefix}_{hashlib.md5(screenshot_data).hexdigest()[:8]}.png"

            # Подготавливаем данные для загрузки
            data = aiohttp.FormData()
            data.add_field('file', screenshot_data, filename=filename, content_type='image/png')

            # Отправляем файл
            async with self.session.post(f"{API_BASE_URL}/upload", data=data) as response:
                if response.status == 200:
                    result = await response.json()
                    self.uploaded_files.append(result['key'])
                    logger.info(f"Скриншот загружен: {result['link']}")
                    return result['link']
                else:
                    logger.error(f"Ошибка при загрузке скриншота: {response.status}")
                    return None

        except Exception as e:
            logger.error(f"Ошибка при загрузке скриншота: {str(e)}")
            return None

    async def cleanup(self):
        """Удаление всех загруженных файлов"""
        if not self.session:
            await self.init_session()

        for key in self.uploaded_files:
            try:
                async with self.session.get(f"{API_BASE_URL}/delete?key={key}") as response:
                    if response.status == 200:
                        logger.info(f"Файл удален: {key}")
                    else:
                        logger.error(f"Ошибка при удалении файла {key}: {response.status}")
            except Exception as e:
                logger.error(f"Ошибка при удалении файла {key}: {str(e)}")

        await self.close_session()
        self.uploaded_files.clear()

        # Очистка временных видеофайлов
        for video_file in self.videos_dir.glob('*.webm'):
            try:
                video_file.unlink()
                logger.debug(f"Временный видеофайл удален: {video_file}")
            except Exception as e:
                logger.warning(f"Не удалось удалить временный видеофайл: {str(e)}")

    async def start_recording(self):
        """Начать запись видео (через серию скриншотов)"""
        self.frames = []
        self.recording = True
        self.frame_count = 0
        logger.info("Начата запись видео через скриншоты")
        
        # Очистка временной директории
        for f in self.temp_dir.glob('*.png'):
            try:
                f.unlink()
            except Exception as e:
                logger.warning(f"Не удалось удалить временный файл {f}: {str(e)}")

    async def add_frame(self, page):
        """Добавить кадр видео"""
        if not self.recording:
            return
            
        if self.frame_count >= VIDEO_MAX_FRAMES:
            logger.warning("Достигнуто максимальное количество кадров")
            return
            
        try:
            screenshot_data = await page.screenshot()
            frame_path = self.temp_dir / f"frame_{self.frame_count:04d}.png"
            with open(frame_path, "wb") as f:
                f.write(screenshot_data)
            self.frames.append(frame_path)
            self.frame_count += 1
            logger.debug(f"Добавлен кадр {self.frame_count}")
        except Exception as e:
            logger.error(f"Ошибка при добавлении кадра: {str(e)}")
            
    async def capture_frames_continuously(self, page, duration_seconds=5):
        """Непрерывно захватывать кадры с заданной частотой"""
        if not self.recording:
            return
            
        logger.info(f"Начинаем непрерывную запись кадров на {duration_seconds} секунд")
        start_time = time.time()
        frame_interval = 1.0 / FRAME_RATE  # интервал между кадрами в секундах
        
        try:
            while self.recording and (time.time() - start_time < duration_seconds):
                if self.frame_count >= VIDEO_MAX_FRAMES:
                    logger.warning("Достигнуто максимальное количество кадров")
                    break
                    
                frame_start = time.time()
                await self.add_frame(page)
                
                # Рассчитываем сколько нужно ждать до следующего кадра
                elapsed = time.time() - frame_start
                sleep_time = max(0, frame_interval - elapsed)
                
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                    
            logger.info(f"Захвачено {self.frame_count} кадров за {time.time() - start_time:.2f} секунд")
        except Exception as e:
            logger.error(f"Ошибка при непрерывной записи кадров: {str(e)}")

    async def _create_video_cv2(self, prefix: str) -> Optional[str]:
        """Создание видео с использованием OpenCV"""
        if not self.frames:
            return None
            
        video_path = self.videos_dir / f"{prefix}_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}.mp4"
        
        try:
            # Получаем размеры первого кадра
            img = cv2.imread(str(self.frames[0]))
            height, width, layers = img.shape
            
            # Создаем видеофайл с оптимизированным кодеком для высокой частоты кадров
            fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Используем H.264 кодек вместо mp4v
            video = cv2.VideoWriter(str(video_path), fourcc, FRAME_RATE, (width, height))
            
            # Добавляем все кадры
            for i, frame_path in enumerate(self.frames):
                try:
                    img = cv2.imread(str(frame_path))
                    if img is not None:
                        video.write(img)
                    else:
                        logger.warning(f"Не удалось прочитать кадр {frame_path}")
                except Exception as e:
                    logger.error(f"Ошибка при добавлении кадра {i}: {str(e)}")
                
            # Освобождаем ресурсы
            video.release()
            
            logger.info(f"Видео создано: {video_path} с {len(self.frames)} кадрами и частотой {FRAME_RATE} fps")
            
            # Загружаем видео
            return await self.upload_file(video_path, prefix)
            
        except Exception as e:
            logger.error(f"Ошибка при создании видео с OpenCV: {str(e)}")
            # Пробуем с другим кодеком если первый не сработал
            try:
                logger.info("Пробуем создать видео с другим кодеком...")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video = cv2.VideoWriter(str(video_path), fourcc, FRAME_RATE, (width, height))
                
                for frame_path in self.frames:
                    img = cv2.imread(str(frame_path))
                    if img is not None:
                        video.write(img)
                        
                video.release()
                logger.info(f"Видео создано с альтернативным кодеком: {video_path}")
                return await self.upload_file(video_path, prefix)
            except Exception as e2:
                logger.error(f"Ошибка при создании видео с альтернативным кодеком: {str(e2)}")
                return None

    async def stop_recording(self, prefix: str) -> Optional[str]:
        """Остановить запись и создать видео из кадров"""
        if not self.recording or not self.frames:
            logger.warning("Нет кадров для создания видео")
            return None
            
        self.recording = False
        logger.info(f"Остановлена запись видео, получено {len(self.frames)} кадров")
        
        # Создаем видео из скриншотов
        try:
            if CV2_AVAILABLE:
                return await self._create_video_cv2(prefix)
            else:
                return await self._create_video_alternative(prefix)
        except Exception as e:
            logger.error(f"Ошибка при создании видео: {str(e)}")
            return None
            
    async def _create_video_alternative(self, prefix: str) -> Optional[str]:
        """Альтернативное создание GIF из кадров (если OpenCV недоступен)"""
        if not self.frames:
            return None
            
        try:
            from PIL import Image
            
            # Создаем GIF из кадров
            gif_path = self.videos_dir / f"{prefix}_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}.gif"
            
            frames = []
            for frame_path in self.frames:
                try:
                    img = Image.open(frame_path)
                    frames.append(img)
                except Exception as e:
                    logger.error(f"Ошибка при добавлении кадра в GIF: {str(e)}")
            
            if frames:
                # Сохраняем как GIF
                frames[0].save(
                    gif_path,
                    save_all=True,
                    append_images=frames[1:],
                    optimize=False,
                    duration=int(1000 / FRAME_RATE),  # Миллисекунды между кадрами
                    loop=0
                )
                
                logger.info(f"GIF создана: {gif_path}")
                return await self.upload_file(gif_path, prefix)
            else:
                logger.error("Нет кадров для создания GIF")
                return None
                
        except ImportError:
            logger.error("Библиотеки Pillow и OpenCV недоступны. Невозможно создать видео.")
            # В крайнем случае загружаем последний кадр как изображение
            if self.frames:
                return await self.upload_file(self.frames[-1], f"{prefix}_last_frame")
            return None

class BoostyMonitor(commands.Bot):
    def __init__(self):
        super().__init__(
            token=TWITCH_TOKEN,
            prefix='!',
            initial_channels=[TWITCH_CHANNEL]
        )
        self.browser: Optional[Browser] = None
        self.context = None
        self.page: Optional[Page] = None
        self.playwright = None
        self._playwright_initialized = False
        self.screenshot_manager = ScreenshotManager()
        self.processed_links = set()  # Для отслеживания уже обработанных ссылок
        self.proxy_manager = WebshareProxyManager()

    async def initialize_playwright(self):
        """Инициализация Playwright"""
        if self._playwright_initialized:
            logger.info("Playwright уже инициализирован, пропускаем инициализацию")
            return

        logger.info("Начинаем инициализацию Playwright")
        
        # Закрываем предыдущие экземпляры, если они есть
        try:
            if self.context:
                logger.info("Закрываем предыдущий контекст")
                await self.context.close()
                self.context = None
            if self.browser:
                logger.info("Закрываем предыдущий браузер")
                await self.browser.close()
                self.browser = None
            if self.playwright:
                logger.info("Останавливаем предыдущий Playwright")
                await self.playwright.stop()
                self.playwright = None
        except Exception as e:
            logger.warning(f"Ошибка при закрытии предыдущих экземпляров: {str(e)}")

        try:
            logger.info("Запускаем Playwright")
            self.playwright = await async_playwright().start()
            logger.info(f"Playwright запущен: {self.playwright}")
            
            # Проверяем доступность исполняемого файла Chromium
            browser_path = None
            try:
                # Проверяем наличие установленного Chrome/Chromium
                if os.name == 'nt':  # Windows
                    possible_paths = [
                        os.path.join(os.environ.get('PROGRAMFILES', 'C:\\Program Files'), 'Google\\Chrome\\Application\\chrome.exe'),
                        os.path.join(os.environ.get('PROGRAMFILES(X86)', 'C:\\Program Files (x86)'), 'Google\\Chrome\\Application\\chrome.exe'),
                        os.path.join(os.environ.get('LOCALAPPDATA', ''), 'Google\\Chrome\\Application\\chrome.exe')
                    ]
                    for path in possible_paths:
                        if os.path.exists(path):
                            browser_path = path
                            logger.info(f"Найден Chrome: {browser_path}")
                            break
                else:  # Linux/Mac
                    # Пытаемся найти через which
                    try:
                        chrome_path = subprocess.check_output(['which', 'google-chrome'], text=True).strip()
                        if os.path.exists(chrome_path):
                            browser_path = chrome_path
                            logger.info(f"Найден Chrome: {browser_path}")
                    except:
                        pass
                    
                    # Проверяем другие возможные пути
                    if not browser_path:
                        possible_paths = [
                            '/usr/bin/google-chrome',
                            '/usr/bin/chromium-browser',
                            '/usr/bin/chromium',
                            '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
                        ]
                        for path in possible_paths:
                            if os.path.exists(path):
                                browser_path = path
                                logger.info(f"Найден Chrome/Chromium: {browser_path}")
                                break
                
                # Если не нашли, используем встроенный браузер Playwright
                if not browser_path:
                    try:
                        browser_path = self.playwright.chromium.executable_path
                        logger.info(f"Используем встроенный Chromium Playwright: {browser_path}")
                    except Exception as e:
                        logger.warning(f"Не удалось получить путь к встроенному Chromium: {str(e)}")
                        browser_path = None
            except Exception as e:
                logger.warning(f"Ошибка при поиске браузера: {str(e)}")
                browser_path = None
            
            # Проверяем найденный путь
            if browser_path and os.path.exists(browser_path):
                logger.info(f"Исполняемый файл браузера существует: {browser_path}")
            else:
                logger.warning("Не удалось найти исполняемый файл браузера, используем встроенный")
                browser_path = None

            # Настраиваем браузер для хоста
            browser_args = [
                '--no-sandbox',
                '--disable-setuid-sandbox',
                '--disable-dev-shm-usage',
                '--disable-accelerated-2d-canvas',
                '--disable-web-security',
                '--disable-features=IsolateOrigins,site-per-process',
                '--disable-site-isolation-trials',
                '--no-default-browser-check',
                '--no-first-run',
                '--allow-insecure-localhost',
                '--allow-file-access',
                '--allow-running-insecure-content',
                '--disable-blink-features=AutomationControlled',
                '--window-size=800,600',
                '--start-maximized',
                '--disable-infobars',
                '--enable-automation',
            ]

            # Получаем случайный прокси
            proxy = await self.proxy_manager.get_random_proxy()
            if proxy:
                proxy_server = {
                    "server": f"http://{proxy['proxy_address']}:{proxy['port']}",
                    "username": proxy['username'],
                    "password": proxy['password']
                }
                logger.info(f"Используется прокси: {proxy['proxy_address']}:{proxy['port']}")
            else:
                proxy_server = None
                logger.warning("Прокси не доступны, работаем без прокси")

            # Настройки видео только если включена запись
            video_options = {}
            if ENABLE_VIDEO_RECORDING:
                # Создаем временную директорию для видео
                video_dir = self.screenshot_manager.videos_dir
                video_options = {
                    "record_video_dir": str(video_dir),
                    "record_video_size": {"width": 1280, "height": 720}  # Уменьшенное разрешение
                }
                logger.info("Запись видео включена")
            else:
                logger.info("Запись видео отключена")
            
            # Запускаем браузер
            launch_options = {
                "headless": True,
                "args": browser_args,
                "chromium_sandbox": False,
                "timeout": 120000,
                "devtools": False
            }
            
            # Добавляем путь к браузеру только если он найден
            if browser_path:
                launch_options["executable_path"] = browser_path
            
            logger.info(f"Запускаем браузер с параметрами: {launch_options}")
            self.browser = await self.playwright.chromium.launch(**launch_options)
            
            # Выводим диагностическую информацию
            logger.info(f"Браузер запущен: {self.browser}")
            
            # Выводим информацию о процессе браузера
            try:
                if hasattr(self.browser, 'process') and hasattr(self.browser.process, 'pid'):
                    logger.info(f"ID процесса браузера: {self.browser.process.pid}")
                    # Проверяем процесс
                    if os.name == 'nt':  # Windows
                        subprocess.run(['tasklist', '/fi', f'pid eq {self.browser.process.pid}'], shell=True)
                    else:  # Linux/Mac
                        subprocess.run(['ps', '-p', str(self.browser.process.pid), '-f'], shell=True)
            except Exception as e:
                logger.error(f"Ошибка при получении информации о процессе: {str(e)}")

            # Создаем новый контекст и страницу с настройками прокси и записью видео
            context_options = {
                "proxy": proxy_server if proxy_server else None,  # Увеличиваем размер
                "ignore_https_errors": True,  # Игнорируем ошибки HTTPS
                "java_script_enabled": True   # Убеждаемся, что JavaScript включен
            }
            # Добавляем опции видео только если включена запись
            if ENABLE_VIDEO_RECORDING:
                context_options.update(video_options)
            
            try:
                self.context = await self.browser.new_context(**context_options)
                logger.info(f"Контекст создан: {self.context}")
            except Exception as context_error:
                logger.error(f"Ошибка при создании контекста: {str(context_error)}")
                raise
            
            # Устанавливаем таймаут для страницы
            self.context.set_default_timeout(PAGE_TIMEOUT)
            
            try:
                self.page = await self.context.new_page()
                logger.info(f"Страница создана: {self.page}")
                
                # Проверяем работу страницы через простую навигацию
                await self.page.goto("about:blank", timeout=10000)
                logger.info("Навигация до about:blank успешна")
            except Exception as page_error:
                logger.error(f"Ошибка при создании страницы: {str(page_error)}")
                raise

            # Настраиваем страницу
            logger.info("Настраиваем заголовки страницы")
            await self.page.set_extra_http_headers({
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'none',
                'Sec-Fetch-User': '?1',
                'Cache-Control': 'no-cache'
            })

            # Блокируем ненужные ресурсы
            await self.context.route("**/*", self._route_handler)

            # Загрузка сохраненных cookies если они есть
            if COOKIES_PATH.exists():
                try:
                    with open(COOKIES_PATH, 'r', encoding='utf-8') as f:
                        cookies = json.load(f)
                        # Преобразуем cookies в формат Playwright
                        playwright_cookies = []
                        for cookie in cookies:
                            # Преобразуем sameSite в формат Playwright
                            same_site = cookie.get('sameSite', 'Lax')
                            if same_site == 'no_restriction':
                                same_site = 'None'
                            elif same_site not in ['Strict', 'Lax', 'None']:
                                same_site = 'Lax'  # Значение по умолчанию

                            playwright_cookie = {
                                'name': cookie['name'],
                                'value': cookie['value'],
                                'domain': cookie['domain'],
                                'path': cookie.get('path', '/'),
                                'secure': cookie.get('secure', True),
                                'httpOnly': cookie.get('httpOnly', False),
                                'sameSite': same_site
                            }
                            # Добавляем expirationDate только если cookie не сессионный
                            if not cookie.get('session', False):
                                playwright_cookie['expires'] = int(cookie.get('expirationDate', 0))
                            playwright_cookies.append(playwright_cookie)
                        
                        await self.context.add_cookies(playwright_cookies)
                        logger.info(f"Успешно загружено {len(playwright_cookies)} cookies")
                except Exception as e:
                    logger.error(f"Ошибка при загрузке cookies: {str(e)}")

            self._playwright_initialized = True
            logger.info("Playwright успешно инициализирован")

        except Exception as e:
            logger.error(f"Ошибка при инициализации Playwright: {str(e)}")
            await self.close()
            raise

    async def _route_handler(self, route):
        """Обработчик маршрутизации для блокировки ненужных ресурсов"""
        resource_type = route.request.resource_type

        # Список разрешенных типов ресурсов
        allowed_types = {
            'document',      # HTML документы
            'script',       # JavaScript
            'xhr',         # AJAX запросы
            'fetch',       # Fetch API запросы
            'websocket',   # WebSocket соединения
            'manifest'     # Web App Manifest
        }

        # Список блокируемых расширений файлов
        blocked_extensions = {
                                           # Source maps
        }

        # Проверяем URL на наличие блокируемых расширений
        url = route.request.url.lower()

        # Блокируем все запросы к favicon.ico
        if 'favicon.ico' in url:
            await route.abort()
            return

        # Блокируем все запросы к apple-touch-icon
        if 'apple-touch-icon' in url:
            await route.abort()
            return

        # Блокируем все запросы к manifest.json
        if 'manifest.json' in url:
            await route.abort()
            return

        # Проверяем расширения файлов
        if any(url.endswith(ext) for ext in blocked_extensions):
            await route.abort()
            return

        # Проверяем тип ресурса
       

        # Продолжаем загрузку разрешенных ресурсов
        await route.continue_()

    async def close(self):
        """Закрытие браузера при завершении работы"""
        if self.context:
            await self.context.close()
        if self.playwright:
            await self.playwright.stop()
        await self.screenshot_manager.cleanup()
        # Останавливаем задачу проверки прокси
        await self.proxy_manager.stop_proxy_check_task()
        self._playwright_initialized = False
        self.context = None
        self.page = None
        self.playwright = None

    async def event_ready(self):
        """Вызывается когда бот готов к работе"""
        logger.info(f'Бот {self.nick} готов к работе!')
        await self.initialize_playwright()

    def clean_and_validate_link(self, text: str) -> Optional[str]:
        """Очистка и валидация ссылки"""
        try:
            logger.debug(f"Начало обработки текста: {text}")

            # Нормализуем текст: удаляем лишние пробелы и приводим к нижнему регистру
            normalized = ' '.join(text.lower().split())
            logger.debug(f"Текст после нормализации: {normalized}")

            # Ищем UUID - любая последовательность букв и цифр между gift/gloft и accept
            uuid_pattern = r'(?:gift|gloft)\s*/([a-z0-9-]+)\s*/(?:accept|acxept)'
            uuid_match = re.search(uuid_pattern, normalized)

            if not uuid_match:
                logger.debug("UUID не найден в тексте")
                return None

            uuid = uuid_match.group(1)
            logger.debug(f"Найден UUID: {uuid}")

            # Ищем username - это все между 'to' и 'gift' или 'gloft'
            username_pattern = r'(?:boosty\.to|boo\s*sty\.to)\s*/([a-z0-9]+)\s*/(?:gift|gloft)'
            username_match = re.search(username_pattern, normalized)

            if not username_match:
                logger.debug("Username не найден в тексте")
                return None

            username = username_match.group(1)
            logger.debug(f"Найден username: {username}")

            # Собираем валидную ссылку, так как у нас есть UUID и username
            valid_link = f"https://boosty.to/{username}/gift/{uuid}/accept"

            # Проверяем, не обрабатывали ли мы уже эту ссылку
            if valid_link in self.processed_links:
                logger.debug(f"Ссылка уже была обработана: {valid_link}")
                return None

            self.processed_links.add(valid_link)
            logger.success(f"Успешно обработана ссылка: {valid_link}")
            return valid_link

        except Exception as e:
            logger.error(f"Ошибка при очистке ссылки: {str(e)}")
            return None

    async def event_message(self, message):
        """Обработка входящих сообщений"""
        if message.echo:
            return

        # Проверяем наличие возможных ссылок в сообщении
        if any(keyword in message.content.lower() for keyword in ['boosty', 'gift', 'accept', 'gloft', 'accept']):
            # Очищаем и валидируем ссылку
            valid_link = self.clean_and_validate_link(message.content)

            if valid_link:
                try:
                    # Извлекаем username из ссылки
                    username = valid_link.split('boosty.to/')[1].split('/')[0]
                    logger.info(f'Обнаружена ссылка на подарок для пользователя: {username}')
                    await self.process_boosty_link(valid_link)
                except Exception as e:
                    logger.error(f'Ошибка при обработке ссылки: {str(e)}')

    async def process_boosty_link(self, url: str):
        """Обработка ссылки Boosty.to"""
        if not self._playwright_initialized:
            logger.error("Playwright не инициализирован")
            await self.initialize_playwright()

        max_retries = MAX_RETRIES
        for attempt in range(max_retries):
            try:
                # Создаем новую страницу для записи видео, но не закрываем контекст
                current_page = await self.context.new_page()
                
                # Устанавливаем таймаут для страницы
                current_page.set_default_timeout(PAGE_TIMEOUT)
                
                # Начинаем запись видео
                if ENABLE_VIDEO_RECORDING:
                    await self.screenshot_manager.start_recording()
                    # Добавляем первый кадр
                    await self.screenshot_manager.add_frame(current_page)
                
                # Переход по ссылке с повышенным таймаутом
                logger.info(f"Переход по ссылке {url} (попытка {attempt + 1}/{max_retries})")
                
                # Запускаем задачу непрерывной записи кадров во время загрузки страницы
                recording_task = None
                if ENABLE_VIDEO_RECORDING:
                    recording_task = asyncio.create_task(
                        self.screenshot_manager.capture_frames_continuously(current_page, 10)
                    )
                
                try:
                    await current_page.goto(url, timeout=PAGE_TIMEOUT)
                except Exception as goto_error:
                    logger.error(f"Ошибка при переходе по ссылке: {str(goto_error)}")
                    # Проверяем, загрузилась ли страница частично
                    try:
                        content = await current_page.content()
                        if "boosty" in content.lower():
                            logger.info("Страница частично загружена, продолжаем")
                        else:
                            raise Exception("Страница не загружена")
                    except:
                        raise Exception("Страница не загружена")
                
                # Если задача записи кадров была запущена, останавливаем её
                if recording_task:
                    try:
                        # Ждем завершения задачи, но не больше 1 секунды
                        await asyncio.wait_for(recording_task, timeout=1)
                    except asyncio.TimeoutError:
                        # Если задача не завершилась, отменяем её
                        recording_task.cancel()
                        try:
                            await recording_task
                        except asyncio.CancelledError:
                            pass
                    except Exception as e:
                        logger.error(f"Ошибка при остановке записи кадров: {str(e)}")
                
                # Пробуем дождаться загрузки с разными стратегиями
                try:
                    await current_page.wait_for_load_state('domcontentloaded', timeout=PAGE_TIMEOUT)
                    logger.info("DOM загружен, продолжаем")
                except Exception as e:
                    logger.warning(f"Не удалось дождаться загрузки DOM: {str(e)}")
                
                # Запускаем запись кадров во время ожидания и поиска кнопок
                if ENABLE_VIDEO_RECORDING:
                    recording_task = asyncio.create_task(
                        self.screenshot_manager.capture_frames_continuously(current_page, 5)
                    )
                else:
                    await asyncio.sleep(3)

                # Поиск и клик по кнопке ACCEPT или CLOSE (приоритет у ACCEPT)
                button_pressed = 'NONE'
                
                # Останавливаем задачу записи кадров
                if recording_task:
                    try:
                        await asyncio.wait_for(recording_task, timeout=1)
                    except asyncio.TimeoutError:
                        recording_task.cancel()
                        try:
                            await recording_task
                        except asyncio.CancelledError:
                            pass
                    except Exception as e:
                        logger.error(f"Ошибка при остановке записи кадров: {str(e)}")
                
                # Пробуем найти кнопки разными способами
                for button_selector in ['text=ACCEPT', 'button:has-text("ACCEPT")', '[data-test="accept-button"]', 
                                       'text=CLOSE', 'button:has-text("CLOSE")', '[data-test="close-button"]']:
                    try:
                        button = await current_page.query_selector(button_selector)
                        if button:
                            # Делаем кадр перед нажатием кнопки
                            if ENABLE_VIDEO_RECORDING:
                                await self.screenshot_manager.add_frame(current_page)
                                
                            await button.click()
                            
                            # Запись кадров после нажатия кнопки
                            if ENABLE_VIDEO_RECORDING:
                                recording_task = asyncio.create_task(
                                    self.screenshot_manager.capture_frames_continuously(current_page, 3)
                                )
                                
                            if 'ACCEPT' in button_selector:
                                button_pressed = 'ACCEPT'
                                logger.success(f'Успешно обработана ссылка: {url} (нажата кнопка ACCEPT)')
                            else:
                                button_pressed = 'CLOSE'
                                logger.success(f'Успешно обработана ссылка: {url} (нажата кнопка CLOSE)')
                            break
                    except Exception as button_error:
                        logger.warning(f"Ошибка при попытке нажать кнопку {button_selector}: {str(button_error)}")
                
                # Ждем завершения записи кадров после нажатия кнопки
                if recording_task:
                    try:
                        await asyncio.wait_for(recording_task, timeout=3)
                    except asyncio.TimeoutError:
                        recording_task.cancel()
                        try:
                            await recording_task
                        except asyncio.CancelledError:
                            pass
                    except Exception as e:
                        logger.error(f"Ошибка при остановке записи кадров: {str(e)}")
                
                if button_pressed == 'NONE':
                    logger.warning(f'Кнопки ACCEPT и CLOSE не найдены на странице: {url}')
                
                # Если запись видео включена
                if ENABLE_VIDEO_RECORDING:
                    # Останавливаем запись и создаем видео
                    video_link = await self.screenshot_manager.stop_recording(f'boosty_{button_pressed.lower()}')
                    if video_link:
                        logger.info(f'Видеозапись процесса загружена: {video_link}')
                    else:
                        logger.error('Не удалось создать видеозапись')
                        
                        # Делаем обычный скриншот как запасной вариант
                        try:
                            screenshot_data = await current_page.screenshot()
                            screenshot_link = await self.screenshot_manager.upload_screenshot(
                                screenshot_data, 
                                f'boosty_{button_pressed.lower()}_fallback'
                            )
                            logger.info(f'Скриншот загружен: {screenshot_link}')
                        except Exception as screenshot_error:
                            logger.error(f"Ошибка при создании скриншота: {str(screenshot_error)}")
                else:
                    # Если запись видео отключена, делаем скриншот
                    try:
                        screenshot_data = await current_page.screenshot()
                        screenshot_link = await self.screenshot_manager.upload_screenshot(
                            screenshot_data, 
                            f'boosty_{button_pressed.lower()}'
                        )
                        logger.info(f'Скриншот загружен: {screenshot_link}')
                    except Exception as screenshot_error:
                        logger.error(f"Ошибка при создании скриншота: {str(screenshot_error)}")
                
                # Закрываем страницу
                await current_page.close()
                
                # Обновляем cookies после успешной обработки
                await self.save_cookies()
                return
                
            except Exception as e:
                logger.error(f'Ошибка при обработке ссылки {url} (попытка {attempt + 1}/{max_retries}): {str(e)}')
                
                # Если запись видео включена, пытаемся сохранить последние кадры
                if ENABLE_VIDEO_RECORDING:
                    try:
                        error_video_link = await self.screenshot_manager.stop_recording('error')
                        if error_video_link:
                            logger.error(f'Видеозапись ошибки: {error_video_link}')
                    except Exception as video_error:
                        logger.error(f'Ошибка при обработке видео ошибки: {str(video_error)}')
                
                # В любом случае пытаемся сделать скриншот
                try:
                    if 'current_page' in locals():
                        screenshot_data = await current_page.screenshot()
                        error_link = await self.screenshot_manager.upload_screenshot(
                            screenshot_data, 
                            'error'
                        )
                        logger.error(f'Скриншот ошибки: {error_link}')
                except Exception as screenshot_error:
                    logger.error(f'Ошибка при создании скриншота: {str(screenshot_error)}')
                
                # Закрываем только проблемную страницу, если она существует
                try:
                    if 'current_page' in locals():
                        await current_page.close()
                except Exception as close_error:
                    logger.error(f'Ошибка при закрытии страницы: {str(close_error)}')

                # При ошибке пытаемся переинициализировать только страницу, но не весь браузер
                if attempt < max_retries - 1:
                    # Проверяем, жив ли контекст, если нет - инициализируем заново
                    try:
                        # Простая проверка - пытаемся получить cookies
                        logger.info("Проверяем работоспособность контекста...")
                        await self.context.cookies()
                        logger.info("Контекст работоспособен")
                    except Exception as context_error:
                        logger.error(f"Контекст не работает: {str(context_error)}")
                        # Если контекст не отвечает, инициализируем Playwright заново,
                        # но не останавливаем сервисы проверки прокси
                        logger.info("Начинаем переинициализацию Playwright")
                        
                        try:
                            if self.context:
                                logger.info("Закрываем контекст")
                                await self.context.close()
                            if self.browser:
                                logger.info("Закрываем браузер")
                                await self.browser.close()
                            if self.playwright:
                                logger.info("Останавливаем Playwright")
                                await self.playwright.stop()
                        except Exception as close_error:
                            logger.warning(f"Ошибка при закрытии: {str(close_error)}")
                        
                        # Сбрасываем флаг инициализации, чтобы вызвать initialize_playwright
                        self._playwright_initialized = False
                        
                        # Ждем немного перед повторной инициализацией
                        logger.info("Ждем 5 секунд перед повторной инициализацией")
                        await asyncio.sleep(5)
                        
                        # Повторная инициализация
                        logger.info("Вызываем повторную инициализацию")
                        try:
                            await self.initialize_playwright()
                            logger.info("Повторная инициализация завершена успешно")
                        except Exception as init_error:
                            logger.error(f"Ошибка при повторной инициализации: {str(init_error)}")
                            # Если не удалось инициализировать, делаем еще одну попытку после задержки
                            await asyncio.sleep(10)
                            logger.info("Делаем последнюю попытку инициализации")
                            await self.initialize_playwright()
                    
                    logger.info(f"Ждем {2 * (attempt + 1)} секунд перед следующей попыткой")
                    await asyncio.sleep(2 * (attempt + 1))  # Увеличиваем время ожидания с каждой попыткой
                else:
                    logger.error(f'Не удалось обработать ссылку после {max_retries} попыток')

    async def save_cookies(self):
        """Сохранение cookies после успешного входа"""
        try:
            cookies = await self.context.cookies()
            # Преобразуем cookies в формат для сохранения
            save_cookies = []
            for cookie in cookies:
                # Преобразуем sameSite обратно в формат файла
                same_site = cookie.get('sameSite', 'Lax')
                if same_site == 'None':
                    same_site = 'no_restriction'

                save_cookie = {
                    'name': cookie['name'],
                    'value': cookie['value'],
                    'domain': cookie['domain'],
                    'path': cookie.get('path', '/'),
                    'secure': cookie.get('secure', True),
                    'httpOnly': cookie.get('httpOnly', False),
                    'sameSite': same_site,
                    'session': 'expires' not in cookie,
                    'storeId': None
                }
                if 'expires' in cookie:
                    save_cookie['expirationDate'] = cookie['expires']
                save_cookies.append(save_cookie)

            with open(COOKIES_PATH, 'w', encoding='utf-8') as f:
                json.dump(save_cookies, f, indent=4)
            logger.info(f'Успешно сохранено {len(save_cookies)} cookies')
        except Exception as e:
            logger.error(f"Ошибка при сохранении cookies: {str(e)}")

async def main():
    bot = BoostyMonitor()
    try:
        await bot.start()
    except KeyboardInterrupt:
        logger.info('Завершение работы бота...')
    finally:
        await bot.close()

if __name__ == '__main__':
    asyncio.run(main())
