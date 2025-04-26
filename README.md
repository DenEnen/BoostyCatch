# Boosty.to Gift Monitor

Скрипт для мониторинга Twitch чата и автоматической обработки ссылок на подарки Boosty.to.

## Требования

- Python 3.8+
- Playwright
- Twitch OAuth токен

## Установка

1. Клонируйте репозиторий:
```bash
git clone <repository-url>
cd boosty-monitor
```

2. Установите зависимости:
```bash
pip install -r requirements.txt
```

3. Установите браузеры для Playwright:
```bash
playwright install
```

4. Создайте файл `.env` на основе `.env.example`:
```bash
cp .env.example .env
```

5. Отредактируйте `.env` файл, добавив ваши данные:
- `TWITCH_CHANNEL`: имя канала Twitch для мониторинга
- `TWITCH_TOKEN`: OAuth токен Twitch (можно получить на https://twitchapps.com/tmi/)

## Первый запуск и настройка

1. Запустите скрипт в режиме с GUI для первоначального входа:
```bash
python boosty_monitor.py
```

2. После успешного входа в Boosty.to, cookies будут автоматически сохранены.

## Запуск на сервере

1. Скопируйте файл `cookies.json` на сервер
2. Запустите скрипт:
```bash
python boosty_monitor.py
```

## Структура проекта

- `boosty_monitor.py` - основной скрипт
- `requirements.txt` - зависимости проекта
- `.env` - конфигурация (не включена в репозиторий)
- `cookies.json` - сохраненные cookies (не включены в репозиторий)
- `screenshots/` - директория для скриншотов

## Логирование

Скрипт использует loguru для логирования. Все действия и ошибки записываются в консоль.

## Безопасность

- Храните `.env` файл и `cookies.json` в безопасном месте
- Не публикуйте эти файлы в репозиторий
- Регулярно обновляйте OAuth токен Twitch 