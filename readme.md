[![security: bandit](https://img.shields.io/badge/security-bandit-yellow.svg)](https://github.com/PyCQA/bandit)
# Описание
Это приложение, которое позволяет использовать бесплатные модели ИИ из пакета Webscout.  
# Возможности  
- Общение с различными моделями LLM
- Генерация картинок по запросу и сохранение их в папку img  
- Смена темы на светлую  
- Смена языка на английский и обратно на русский  
- Поиск LLM  
- Распознавание текста с фото на русском и английском языке - внимание, необходимо скачать tesseract.exe, он находится в релизах, вместе с исполняемым файлом llm.exe для старых версий. Начиная с версии 1.43 это не требуется.  
- Распознавание текста с QR кода
- Голосовой ввод
- Очистка чата
- Возможность использовать режим API - при активации открывается страница, ссылку на неё можно отправить другим пользовать и использовать чат в локальной сети, также доступна страница со Swagger  
- Читать текст в слух  
- Читать текст из файлов .txt, .docx, .xlsx, .xls, .csv, .odt, .doc и вставлять его в поле ввода  
- Чтение параметров из файла .env (пример ниже)  
- Запуск Telegram бота со схожим функционалом программы (файл llm.exe является сервером)

# Использование
1. Клонировать репозиторий:  
```git clone https://github.com/Processori7/llm.git```
2. Перейти в папку (на Windows):  
```cd /d llm```  
Unix:  
```cd llm```  
3. Создать виртуальное окружение:  
```python -m venv venv```  
На Unix: ```python3 -m venv venv```  
4. Активировать виртуальное окружение:  
. На Windows:  
```venv\Scripts\activate```  
. Unix:  
```source venv/bin/activate```
5. Установить зависимости:
```pip install -r requirements.txt```
6. Запустить файл:
```python llm.py```  

## Известные проблемы:  
Так как пока не принят пулл реквест, то при запуске кода может возникнуть ошибка:  
from webscout import KOBOLDAI, BLACKBOXAI, YouChat, Felo, PhindSearch, DARKAI, VLM, TurboSeek, Netwrck, Qwenlm, Marcus, WEBS as w  
ImportError: cannot import name 'Qwenlm' from 'webscout' (E:\Users\User\Desktop\LLM\.venv\Lib\site-packages\webscout\__init__.py)   
Решение:  
Скачать файлы init.py и Qwenlm.py [отсюда](https://github.com/Processori7/Webscout/tree/qwenlm/webscout/Provider) и копировать с заменой эти файлы в папку venv\Lib\site-packages\webscout\Provider   

## Известные проблемы при установке на Unix:
1. Не удаётся установить PyAudio.  
   Решение:  
```sudo apt-get install libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0```  
    ```sudo apt-get install ffmpeg libav-tools```  
    ```pip install PyAudio```  
И заново выполнить:  
```pip install -r requirements.txt```  
2. Модуль Tkinter не найден.  
Решение:
```sudo apt-get install python3-tk```  
3. Модуль aiohttp не найден.  
   Решение:
   ```sudo apt install python3-aiohttp```  
4. Установить дополнительные модули:
```pip install colorama python-xlib aiohttp```  
5. Установка Tesseract OCR - для распознавания текста на картинке:  
```sudo apt update && sudo apt upgrade```  
```sudo apt install tesseract-ocr```  
```sudo apt install libtesseract-dev```  
Для поддержки русского языка:
```sudo apt install tesseract-ocr-rus```  
6. Ошибка при запуске:  
"PyGetWindow currently does not support this platform."  
"If you have knowledge of the platform's windowing system, please contribute! "  
"https://github.com/asweigart/pygetwindow"  
   Решение:  
Пока PyGetWindow официально не поддерживает Linux, но можно попробовать исправить эту проблему. Выполните эти действия:  
Скачайте файлы [отсюда](https://github.com/Processori7/PyGetWindow/tree/experimental_Linux_support/src/pygetwindow)   
В вашем виртуальном окружении найдите папку PyGetWindow и копируйте новые файлы туда с заменой.  
После этого заново попробуйте запустить Ваше приложение.  
Код тестировался на последней версии Lubuntu с последними версиями пакетов Tkinter и CustomTkinter, в теории код может быть запущен и на других OC с поддержкой X-lib.  


# Description
This is an application that allows you to use free services from the Webscout package.  
# Features  
- Communication with different LLM models
- Generate images on request and save them to the img folder  
- Changing the theme to a light one  
- Change the language to English and back to Russian  
- LLM Search  
- Text recognition from photos in Russian and English - attention, you need to download tesseract.exe it is in the releases, along with the executable file. llm.exe for older versions. Starting from version 1.43, this is not required.  
- QR Code text recognition  
- Voice input
- Clearing the chat
- The ability to use API mode - upon activation, a page opens, a link to it can be sent to others to use and use chat on the local network, a page with Swagger is also available  
- Read text aloud  
- Read text from files .txt, .docx, .xlsx, .xls, .csv, .odt, .doc and insert in input field  
- Reading parameters from an.env file (example below)  
- Launching a Telegram bot with similar program functionality (file llm.exe is a server)
# Usage
1. Clone the repository:  
`git clone https://github.com/Processori7/llm.git `
2. Navigate to the folder (on Windows):  
```cd /d llm```  
Unix:  
```cd llm```  
3. Create a virtual environment:  
```python -m venv venv```  
On Unix: ```python3 -m venv venv```
4. Activate the virtual environment:  
. On Windows:  
```venv\Scripts\activate```  
. Unix:  
```source venv/bin/activate```  
5. Install dependencies:
```pip install -r requirements.txt ```
6. Run the file:
```python llm.py```  

## Known issues:  
Since the pool request has not been accepted yet, an error may occur when running the code.:  
from webscout import KOBOLDAI, BLACKBOXAI, YouChat, File, Find Search, DARKAI, VLM, Turbo Seek, Netwrck, Qwenlm, Marcus, WEB asp  
ImportError: cannot import name 'Qwenlm' from 'webscout' (E:\Users\User\Desktop\LLM\.venv\Lib\site-packages\webscout\__init__.py )  
Solution:  
Download files init.py and Qwenlm.py [from here](https://github.com/Processori7/Webscout/tree/qwenlm/webscout/Provider) and copy and replace these files to the venv\Lib\site-packages\webscout\Provider folder  

## Known problems when installing on Unix:
1. PyAudio cannot be installed.  
   Decision:  
```sudo apt-get install libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0```  
    ```sudo apt-get install ffmpeg libav-tools```  
    ```pip install PyAudio```  
And re-execute:  
```pip install -r requirements.txt```   
2. The Tkinter module was not found.  
Solution:
```sudo apt-get install python3-tk```  
4. The aiohttp module was not found.  
   Solution:
```sudo apt install python3-aiohttp```  
5. Install additional modules:
```pip install colorama python-xlib aiohttp```  
6. Install Tesseract OCR:  
```sudo apt update && sudo apt upgrade```  
```sudo apt install tesseract-ocr```  
```sudo apt install libtesseract-dev```  
7. Startup error:  
"PyGetWindow currently does not support this platform."  
"If you have knowledge of the platform's windowing system, please contribute! "  
"https://github.com/asweigart/pygetwindow"  
   Decision:
PyGetWindow does not officially support Linux yet, but you can try to fix this problem. Follow these steps:  
Download the files [from here](https://github.com/Processori7/PyGetWindow/tree/experimental_Linux_support/src/pygetwindow)  
In your virtual environment, find the PyGetWindow folder and copy the new files there with the replacement.  
After that, try launching your application again.  
The code was tested on the latest version of Lubuntu with the latest versions of the Tkinter and CustomTkinter packages. In theory, the code can be run on other OCS with X-lib support.  
## Example .env file/ Пример файла .env  
```env
FONT_SIZE=15  
HOST=AUTO  
PORT=8000  
MODEL=0  
IS_TRANSLATE=False  
IMG_FOLDER=img  
WRITE_HISTORY=False  
MODE=dark  
DEFAULT_COLOR_THEM=green  
SYSTEM_PROMPT="Your custom system prompt here"  
FREEAICHAT_API_KEY=get_your_key_here:https://freeaichatplayground.com/profile  
TELEGRAM_BOT_TOKEN=get_your_token_here:https://t.me/BotFather  
```

## Описание настроек .env файла / .env file settings description

**FONT_SIZE** - размер шрифта интерфейса / interface font size (по умолчанию/default: 15)

**HOST** - IP-адрес для API сервера / IP address for API server:  
- `AUTO` - автоматическое определение локального IP (рекомендуется) / automatic local IP detection (recommended)  
- `127.0.0.1` - только локальный доступ / localhost only  
- Любой другой IP-адрес / any other IP address  

**PORT** - порт для API сервера / port for API server (по умолчанию/default: 8000)

**MODEL** - индекс модели по умолчанию / default model index (0 = первая модель/first model)

**IS_TRANSLATE** - язык интерфейса / interface language:  
- `False` - русский / Russian  
- `True` - английский / English  

**IMG_FOLDER** - папка для сохранения изображений / folder for saving images (по умолчанию/default: img)

**WRITE_HISTORY** - сохранение истории чата / save chat history:  
- `False` - не сохранять / don't save  
- `True` - сохранять в файл llm_history.txt / save to llm_history.txt  

**MODE** - тема оформления / theme:  
- `dark` - тёмная тема / dark theme  
- `light` - светлая тема / light theme  

**DEFAULT_COLOR_THEM** - цветовая схема / color scheme:  
- `green`, `blue`, `dark-blue` и другие / and others

**SYSTEM_PROMPT** - системный промпт для LLM моделей / system prompt for LLM models

**FREEAICHAT_API_KEY** - API ключ для FreeAIChat моделей / API key for FreeAIChat models  
Получить можно здесь / Get it here: https://freeaichatplayground.com/profile

**TELEGRAM_BOT_TOKEN** - токен Telegram бота / Telegram bot token  
Получить можно здесь / Get it here: https://t.me/BotFather
