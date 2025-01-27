# Описание
Это приложение, которое позволяет использовать бесплатные модели ИИ из пакета Webscout.  
# Возможности  
- Общение с различными моделями LLM
- Генерация картинок по запросу и сохранение их в папку img  
- Смена темы на светлую  
- Смена языка на английский и обратно на русский  
- Поиск LLM  
- Распознавание текста с фото на русском и английском языке - внимание, необходимо скачать tesseract.exe, он находится в релизах, вместе с исполняемым файлом llm.exe для старых версий. Начиная с версии 1.43 это не требуется.
- Голосовой ввод
- Очистка чата
- Возможность использовать режим API - при активации открывается страница, ссылку на неё можно отправить другим пользовать и использовать чат в локальной сети, также доступна страница со Swagger

# Использование
1. Клонировать репозиторий:  
```git clone https://github.com/Processori7/llm.git```
2. Перейти в папку (на Windows):  
```cd /d llm```  
Unix:  
```cd llm```
3. Создать виртуальное окружение:  
```python -m venv venv```
4. Активировать виртуальное окружение:  
```venv\Scripts\activate```
5. Установить зависимости:  
```pip install -r requirements.txt```
6. Запустить файл:  
```python llm.py```  

# Description
This is an application that allows you to use free services from the Webscout package.  
# Features  
- Communication with different LLM models
- Generate images on request and save them to the img folder  
- Changing the theme to a light one  
- Change the language to English and back to Russian  
- LLM Search  
- Text recognition from photos in Russian and English - attention, you need to download tesseract.exe it is in the releases, along with the executable file. llm.exe for older versions. Starting from version 1.43, this is not required.
- Voice input
- Clearing the chat
- The ability to use API mode - upon activation, a page opens, a link to it can be sent to others to use and use chat on the local network, a page with Swagger is also available
# Usage
1. Clone the repository:  
`the bastard clone https://github.com/Processori7/llm.git `
2. Navigate to the folder (on Windows):  
```cd /d llm```  
Unix:  
```cd llm```
3. Create a virtual environment:  
```python -m venv venv```
4. Activate the virtual environment:  
`venv\Scripts\activate`
5. Install dependencies:  
`pip install -r requirements.txt `
6. Run the file:  
```python llm.py```