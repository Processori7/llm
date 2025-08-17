import subprocess
import sys

# Получаем список установленных пакетов
def get_installed_packages():
    from importlib.metadata import distributions
    return [dist.metadata['Name'] for dist in distributions()]

# Обновляем pip
subprocess.call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])

# Обновляем зависимости из requirements.txt
subprocess.call([sys.executable, "-m", "pip", "install", "--upgrade", "-r", "requirements.txt"])

# Обновляем все установленные пакеты
installed_packages = get_installed_packages()
subprocess.call([sys.executable, "-m", "pip", "install", "--upgrade"] + installed_packages)
