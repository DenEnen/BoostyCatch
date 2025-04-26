# playwright_installer.py
import subprocess

def install_playwright():
    """
    Executes the command 'python -m playwright install' to install Playwright browsers.
    
    Raises:
        subprocess.CalledProcessError: If the command fails.
    """
    try:
        result = subprocess.run(
            ["python", "-m", "playwright", "install"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True  # This ensures that stdout and stderr are returned as strings.
        )
        print("Playwright installation output:")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("An error occurred during Playwright installation:")
        print(e.stderr)
        raise

if __name__ == "__main__":
    install_playwright()
