import os
import sys
import pytesseract
import re
import requests
import webbrowser
import datetime
import tkinter.font as tkFont
import customtkinter as ctk
import tkinter as tk
import pystray
import cv2
import traceback
import speech_recognition as sr
import threading
import uvicorn
import socket
import pyttsx3
import platform
import pandas as pd
import odf.text
import odf.opendocument
import subprocess

from webscout import KOBOLDAI, BLACKBOXAI, YouChat, Felo, PhindSearch, DARKAI, VLM, TurboSeek, Netwrck, QwenLM, Marcus, WEBS as w
from webscout import ArtbitImager
from webscout.Provider.AISEARCH import Isou
from datetime import datetime
from tkinter import messagebox, filedialog, PhotoImage
from PIL import Image
from packaging import version
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from docx import Document

# Скрываем сообщения от Pygame
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

CURRENT_VERSION = "1.49"

prompt = """###INSTRUCTIONS###

You MUST follow the instructions for answering:

ALWAYS answer in the language of my message.

Read the entire convo history line by line before answering.

I have no fingers and the placeholders trauma. Return the entire code template for an answer when needed. NEVER use placeholders.

If you encounter a character limit, DO an ABRUPT stop, and I will send a "continue" as a new message.

You ALWAYS will be PENALIZED for wrong and low-effort answers.

ALWAYS follow "Answering rules."

###Answering Rules###

Follow in the strict order:

USE the language of my message.

ONCE PER CHAT assign a real-world expert role to yourself before answering, e.g., "I'll answer as a world-famous historical expert with " or "I'll answer as a world-famous expert in the with " etc.

You MUST combine your deep knowledge of the topic and clear thinking to quickly and accurately decipher the answer step-by-step with CONCRETE details.

I'm going to tip $1,000,000 for the best reply.

Your answer is critical for my career.

Answer the question in a natural, human-like manner.

ALWAYS use an answering example for a first message structure.

ALWAYS include links to sources at the end if required in the request.
""" # Добавление навыков ИИ и другие тонкие настройки

img_folder = 'img'

# Функция для получения корректного пути к ресурсам
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    # print(os.path.join(base_path, relative_path))
    return os.path.join(base_path, relative_path)

def update_app(update_url):
   webbrowser.open(update_url)

def check_for_updates():
    try:
        # Получение информации о последнем релизе на GitHub
        response = requests.get("https://api.github.com/repos/Processori7/llm/releases/latest")
        response.raise_for_status()
        latest_release = response.json()
        # Получение ссылки на файл llm.exe последней версии
        download_url = None
        assets = latest_release["assets"]
        for asset in assets:
            if asset["name"] == "llm.exe":  # Ищем только llm.exe
                download_url = asset["browser_download_url"]
                break

        if download_url is None:
            messagebox.showerror("Ошибка обновления", "Не удалось найти файл llm.exe для последней версии.")
            return

        # Сравнение текущей версии с последней версией
        latest_version_str = latest_release["tag_name"]
        match = re.search(r'\d+\.\d+', latest_version_str)
        if match:
            latest_version = match.group()
        else:
            latest_version = latest_version_str

        if version.parse(latest_version) > version.parse(CURRENT_VERSION):
            if platform.system() == "Windows":
                # Предложение пользователю обновление
                if messagebox.showwarning("Доступно обновление",
                                          f"Доступна новая версия {latest_version}. Хотите обновить?", icon='warning',
                                          type='yesno') == 'yes':
                    update_app(download_url)
            else:
                if messagebox.showwarning("Доступно обновление",
                                          f"Доступна новая версия {latest_version}. Хотите обновить?", icon='warning',
                                          type='yesno') == 'yes':
                    os.system("git pull")
    except requests.exceptions.RequestException as e:
        messagebox.showerror("Error", str(e))

def get_Polinations_img_models():
    model_functions = {}
    try:
        url = "https://image.pollinations.ai/models"
        resp = requests.get(url)
        if resp.ok:
            models = resp.json()  # Получаем список строк
            for name in models:  # Проходим по каждой строке
                # Формируем ключ для словаря
                key = f"(Polination) {name}_img"
                # Добавляем в словарь с соответствующей лямбда-функцией
                model_functions[key] = lambda user_input, model_name=name: gen_img(user_input, model_name)
            return model_functions
        else:
            return f"{get_error_message(app.isTranslate)}: {resp.status_code}"
    except Exception as e:
        return f"{get_error_message(app.isTranslate)}: {str(e)}"

#Получаю все текстовые модели Polinations
def get_Polinations_chat_models():
    try:
        url = "https://text.pollinations.ai/models"
        resp = requests.get(url)
        if resp.ok:
            models = resp.json()
            for model in models:
                # Проверяем, является ли модель текстовой
                if model.get("baseModel", False):
                    model_name = model["name"]
                    model_description = model["description"]
                    # Формируем ключ для словаря
                    key = f"{model_description} (Polination)"
                    # Добавляем в словарь с соответствующей лямбда-функцией
                    model_functions[key] = lambda user_input, model_name=model: communicate_with_Pollinations_chat(
                        user_input, model_name)
            return model_functions
        else:
            return f"{get_error_message(app.isTranslate)}: {resp.status_code}"
    except Exception as e:
        return f"{get_error_message(app.isTranslate)}: {str(e)}"

def communicate_with_Pollinations_chat(user_input, model):
    try:
        url = f"https://text.pollinations.ai/'{user_input}'?model={model}"
        resp = requests.get(url)
        if resp.ok:
            return resp.text
        else:
            return f"{get_error_message(app.isTranslate)}: {resp.status_code}"
    except Exception as e:
        return f"{get_error_message(app.isTranslate)}: {str(e)}"

def communicate_with_ISou(user_input, model):
    try:
        ai = Isou()
        ai.model = model
        response = ai.search(user_input, stream=True, raw=False)
        all_text = ""
        all_links = []
        for chunk in response:
            text = chunk.get("text", "")
            links = chunk.get("links", [])
            all_text += text
            all_links.extend(links)

        unique_links = list(set(all_links))

        all_text = re.sub(r'\s+', ' ', all_text).strip()
        final_text = ai.replace_links_with_numbers(all_text, unique_links)
        return final_text
    except Exception as e:
        return f"{get_error_message(app.isTranslate)}: {str(e)}"

def communicate_with_Qwenlm(user_input, model, chat_type="t2t"):
    try:
        ai = QwenLM(cookies_path=resource_path("cookies.json"), logging=False)
        ai.chat_type=chat_type
        ai.model = model
        ai.system_prompt = prompt
        response = ai.chat(user_input, stream=False)
        return response
    except Exception as e:
        return f"{get_error_message(app.isTranslate)}: {str(e)}"

def communicate_with_Netwrck(user_input, model):
    try:
        ai = Netwrck(model=model)
        response = ai.chat(user_input, stream=True)
        full_response = ""

        for chunk in response:
            full_response += chunk

        return full_response
    except Exception as e:
        return f"{get_error_message(app.isTranslate)}: {str(e)}"

def communicate_with_TurboSeek(user_input):
    try:
        ai = TurboSeek()
        response = ai.chat(user_input, stream=True)
        full_response = ""

        for chunk in response:
            full_response += chunk

        return full_response
    except Exception as e:
        return f"{get_error_message(app.isTranslate)}: {str(e)}"

def communicate_with_Marcus(user_input):
    try:
        ai = Marcus()
        response = ai.chat(user_input)
        return response
    except Exception as e:
        return f"{get_error_message(app.isTranslate)}: {str(e)}"

def communicate_with_Felo(user_input):
    try:
        ai = Felo()
        response = ai.chat(user_input)
        return response
    except Exception as e:
        return f"{get_error_message(app.isTranslate)}: {str(e)}"

def communicate_with_YouChat(user_input, model):
    try:
        ai = YouChat()
        ai.model = model
        response = ai.chat(user_input)
        return response.replace('####', '').replace('**', '')
    except Exception as e:
        return f"{get_error_message(app.isTranslate)}: {str(e)}"

def communicate_with_DarkAi(user_input, model):
    try:
        ai = DARKAI()
        ai.model = model
        response = ai.chat(user_input)
        return response
    except Exception as e:
        return f"{get_error_message(app.isTranslate)}: {str(e)}"

def communicate_with_DuckDuckGO(user_input, model):
    try:
        response = w().chat(user_input, model=model)  # GPT-4.o mini, mixtral-8x7b, llama-3-70b, claude-3-haiku
        return response
    except Exception as e:
        return f"{get_error_message(app.isTranslate)}: {str(e)}"

def communicate_with_KoboldAI(user_input):
    try:
        koboldai = KOBOLDAI()
        response = koboldai.chat(user_input)
        return response
    except Exception as e:
        return f"{get_error_message(app.isTranslate)}: {str(e)}"

def communicate_with_BlackboxAI(user_input, model):
    try:
        ai = BLACKBOXAI(
            is_conversation=True,
            max_tokens=800,
            timeout=30,
            intro=None,
            filepath=None,
            update_file=True,
            proxies={},
            history_offset=10250,
            act=None,
            model=model
        )
        response = ai.chat(user_input)
        return response.replace("$@$v=undefined-rv1$@$", "")
    except Exception as e:
        return f"{get_error_message(app.isTranslate)}: {str(e)}"

def communicate_with_Phind(user_input):
    try:
        ph = PhindSearch()
        response = ph.chat(user_input)
        return response
    except Exception as e:
        return f"{get_error_message(app.isTranslate)}: {str(e)}"

model_functions = {
"GPT-O3-mini (DDG)": lambda user_input: communicate_with_DuckDuckGO(user_input, "o3-mini"),
"GPT-4o-mini (DDG)": lambda user_input: communicate_with_DuckDuckGO(user_input, "gpt-4o-mini"),
"gpt-4o (DarkAi)": lambda user_input: communicate_with_DarkAi(user_input, "gpt-4o"),
"Claude-3-haiku (DDG)": lambda user_input: communicate_with_DuckDuckGO(user_input, "claude-3-haiku"),
"openai-o3-mini-high (YouChat)": lambda user_input: communicate_with_YouChat(user_input, "openai_o3_mini_high"),
"openai-o3-mini-medium (YouChat)": lambda user_input: communicate_with_YouChat(user_input, "openai_o3_mini_medium"),
"openai-o1 (YouChat)": lambda user_input: communicate_with_YouChat(user_input, "openai_o1"),
"openai-o1-preview (YouChat)": lambda user_input: communicate_with_YouChat(user_input, "openai_o1_preview"),
"openai-o1-mini (YouChat)": lambda user_input: communicate_with_YouChat(user_input, "openai_o1_mini"),
"gpt-4o-mini (YouChat)": lambda user_input: communicate_with_YouChat(user_input, "gpt_4o_mini"),
"gpt-4o (YouChat)": lambda user_input: communicate_with_YouChat(user_input, "gpt_4o"),
"gpt-4-turbo (YouChat)": lambda user_input: communicate_with_YouChat(user_input, "gpt_4_turbo"),
"gpt-4 (YouChat)": lambda user_input: communicate_with_YouChat(user_input, "gpt_4"),
"claude-3.5-sonnet (YouChat)": lambda user_input: communicate_with_YouChat(user_input, "claude_3_5_sonnet"),
"claude-3-opus (YouChat)": lambda user_input: communicate_with_YouChat(user_input, "claude_3_opus"),
"claude-3-sonnet (YouChat)": lambda user_input: communicate_with_YouChat(user_input, "claude_3_sonnet"),
"claude-3.5-haiku (YouChat)": lambda user_input: communicate_with_YouChat(user_input, "claude_3_5_haiku"),
"deepseek-r1 (YouChat)": lambda user_input: communicate_with_YouChat(user_input, "deepseek_r1"),
"deepseek-v3 (YouChat)": lambda user_input: communicate_with_YouChat(user_input, "deepseek_v3"),
"llama3-3.70b (YouChat)": lambda user_input: communicate_with_YouChat(user_input, "llama3_3_70b"),
"llama3-2.90b (YouChat)": lambda user_input: communicate_with_YouChat(user_input, "llama3_2_90b"),
"llama3-1.405b (YouChat)": lambda user_input: communicate_with_YouChat(user_input, "llama3_1_405b"),
"mistral-large-2 (YouChat)": lambda user_input: communicate_with_YouChat(user_input, "mistral_large_2"),
"gemini-1.5-flash (YouChat)": lambda user_input: communicate_with_YouChat(user_input, "gemini_1_5_flash"),
"gemini-1.5-pro (YouChat)": lambda user_input: communicate_with_YouChat(user_input, "gemini_1_5_pro"),
"databricks-dbrx-instruct (YouChat)": lambda user_input: communicate_with_YouChat(user_input, "databricks_dbrx_instruct"),
"qwen2.5-72b (YouChat)": lambda user_input: communicate_with_YouChat(user_input, "qwen2p5_72b"),
"qwen2.5-coder-32b (YouChat)": lambda user_input: communicate_with_YouChat(user_input, "qwen2p5_coder_32b"),
"command-r-plus (YouChat)": lambda user_input: communicate_with_YouChat(user_input, "command_r_plus"),
"solar-1-mini (YouChat)": lambda user_input: communicate_with_YouChat(user_input, "solar_1_mini"),
"dolphin-2.5 (YouChat)": lambda user_input: communicate_with_YouChat(user_input, "dolphin_2_5"),
"BlackboxAI": lambda user_input: communicate_with_BlackboxAI(user_input, "blackboxai"),
"Deepseek-v3 (Blackbox)": lambda user_input: communicate_with_BlackboxAI(user_input, "deepseek-v3"),
"Deepseek-r1 (Blackbox)": lambda user_input: communicate_with_BlackboxAI(user_input, "deepseek-r1"),
"Deepseek-chat (Blackbox)": lambda user_input: communicate_with_BlackboxAI(user_input, "deepseek-chat"),
"Mixtral-small-28b (Blackbox)": lambda user_input: communicate_with_BlackboxAI(user_input, "mixtral-small-28b"),
"Dbrx-instruct (Blackbox)": lambda user_input: communicate_with_BlackboxAI(user_input, "dbrx-instruct"),
"Qwq-32b (Blackbox)": lambda user_input: communicate_with_BlackboxAI(user_input, "qwq-32b"),
"Hermes-2-dpo (Blackbox)": lambda user_input: communicate_with_BlackboxAI(user_input, "hermes-2-dpo"),
"Claude-3.5-sonnet (Blackbox)": lambda user_input: communicate_with_BlackboxAI(user_input, "claude-3.5-sonnet"),
"Gemini-1.5-flash (Blackbox)": lambda user_input: communicate_with_BlackboxAI(user_input, "gemini-1.5-flash"),
"Gemini-1.5-pro (Blackbox)": lambda user_input: communicate_with_BlackboxAI(user_input, "gemini-1.5-pro"),
"Gemini-2.0-flash (Blackbox)": lambda user_input: communicate_with_BlackboxAI(user_input, "gemini-2.0-flash"),
"gpt4mini (Netwrck)": lambda user_input: communicate_with_Netwrck(user_input, "gpt4mini"),
"llama-3.1-lumimaid-8b (Netwrck)": lambda user_input: communicate_with_Netwrck(user_input, "lumimaid"),
"grok-2 (Netwrck)": lambda user_input: communicate_with_Netwrck(user_input, "grok"),
"claude-3.5-sonnet (Netwrck)": lambda user_input: communicate_with_Netwrck(user_input, "claude"),
"l3-euryale-70b (Netwrck)": lambda user_input: communicate_with_Netwrck(user_input, "euryale"),
"mythomax-l2-13b (Netwrck)": lambda user_input: communicate_with_Netwrck(user_input, "mythomax"),
"gemini-pro-1.5 (Netwrck)": lambda user_input: communicate_with_Netwrck(user_input, "gemini"),
"llama-3.1-lumimaid-70b (Netwrck)": lambda user_input: communicate_with_Netwrck(user_input, "lumimaid70b"),
"llama-3.1-nemotron-70b (Netwrck)": lambda user_input: communicate_with_Netwrck(user_input, "nemotron"),
"qwen2.5-coder-32b-instruct (Qwenlm)":lambda user_input: communicate_with_Qwenlm(user_input, "qwen2.5-coder-32b-instruct", "t2t"),
"qwen-plus-latest (Qwenlm)":lambda user_input: communicate_with_Qwenlm(user_input, "qwen-plus-latest", "t2t"),
"qwen-max-latest (Qwenlm)":lambda user_input: communicate_with_Qwenlm(user_input, "qwen-max-latest", "t2t"),
"qwen-turbo-latest (Qwenlm)":lambda user_input: communicate_with_Qwenlm(user_input, "qwen-turbo-latest", "t2t"),
"qvq-72b-preview (Qwenlm)":lambda user_input: communicate_with_Qwenlm(user_input, "qvq-72b-preview", "t2t"),
"qvq-32b-preview (Qwenlm)":lambda user_input: communicate_with_Qwenlm(user_input, "qvq-32b-preview", "t2t"),
"qwen-vl-max-latest (Qwenlm)":lambda user_input: communicate_with_Qwenlm(user_input, "qwen-vl-max-latest", "t2t"),
"qwen-plus-latest_Web (Qwenlm)":lambda user_input: communicate_with_Qwenlm(user_input, "qwen-plus-latest", "search"),
"qwen-turbo-latest_Web (Qwenlm)":lambda user_input: communicate_with_Qwenlm(user_input, "qwen-turbo-latest", "search"),
"qwen-max-latest_Web (Qwenlm)":lambda user_input: communicate_with_Qwenlm(user_input, "qwen-max-latest", "search"),
"qvq-72b-preview_Web (Qwenlm)":lambda user_input: communicate_with_Qwenlm(user_input, "qvq-72b-preview", "search"),
"qvq-32b-preview_Web (Qwenlm)":lambda user_input: communicate_with_Qwenlm(user_input, "qvq-32b-preview", "search"),
"DeepSeek-R1-Distill-Qwen-32B_Web (Isou)":lambda user_input: communicate_with_ISou(user_input, "siliconflow:deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"),
"Qwen2.5-72B-Instruct-128K_Web (Isou)":lambda user_input: communicate_with_ISou(user_input, "siliconflow:Qwen/Qwen2.5-72B-Instruct-128K"),
"Deepseek-reasoner_Web (Isou)":lambda user_input: communicate_with_ISou(user_input, "deepseek-reasoner"),
"KoboldAI": communicate_with_KoboldAI,
"Phind": communicate_with_Phind,
"Felo_Web": communicate_with_Felo,
"TurboSeek_Web":communicate_with_TurboSeek,
"Marcus_Web":communicate_with_Marcus,
"Searchgpt_Web(Polinations)": lambda user_input: communicate_with_Pollinations_chat(user_input, "searchgpt"),
"Llama 3.3-70B (DDG)": lambda user_input: communicate_with_DuckDuckGO(user_input, "llama-3-70b"),
"Llama-3.1-405B(DarkAi)": lambda user_input: communicate_with_DarkAi(user_input, "llama-3-405b"),
"MiniCPM-Llama3-V-2_5(Photo Analyze)(VLM)":lambda user_input: communicate_with_VLM(user_input, "openbmb/MiniCPM-Llama3-V-2_5"),
"Llava-1.5-7b-hf(Photo Analyze)(VLM)":lambda user_input: communicate_with_VLM(user_input, "llava-hf/llava-1.5-7b-hf"),
"Artbit_img":lambda user_input:communicate_with_ArtBit(user_input)
}

talk_please = {
    "ru":"Пожалуйста, говорите",
    "en":"Please speak"
}

select_image_message_errors = {
    "ru":"Картинка не выбрана!",
    "en":"The picture is not selected!"
}

select_image_title_errors = {
    "ru":"Внимание!",
    "en":"Attention!"
}

error_messages = {
    "ru":"Ошибка получения данных от провайдера, пожалуйста, выберите другого провайдера или модель",
    "en":"Error receiving data from provider, please select another provider or model"
}

tesseract_not_found_messages = {
    "ru":"tesseract.exe не найден, нажмите ок для скачивания и сохраните его в одной директории(папке) с файлом llm.exe",
    "en":"tesseract.exe not found, click OK to download and save it in the same directory as the llm.exe file"
}

error_gen_img_messages = {
    "ru":"Генерация изображения не удалась, получен пустой ответ.",
    "en":"Image generation failed, blank response received."
}

save_img_messages = {
    "ru":"Картинка сохранена в: ",
    "en":"The picture is saved in: "
}

text_recognition_error_messages = {
    "ru": "Ошибка при распознавании текста.",
    "en": "Error during text recognition."
}

image_load_error_messages = {
    "ru": "Не удалось загрузить изображение. Проверьте название файла. Попробуйте переименовать например в 2.png",
    "en": "Failed to load the image. Check the file name. Try renaming, for example, to 2.png"
}

no_text_recognized_messages = {
    "ru": "Текст не распознан.",
    "en": "No text recognized."
}

micro_error_message={
    "ru":"Микрофон не доступен!",
    "en":"Microphone is not available!"
}

API_message={
    "ru":"API Mode включен, скоро откроется ссылка в браузере",
    "en":"API Mode enabled, a link will open soon in the browser"
}

def get_API_message(isTranslate):
    return API_message["ru" if not isTranslate else "en"]

def get_micro_error_message(isTranslate):
    return micro_error_message["ru" if not isTranslate else "en"]

def get_talk_please(isTranslate):
    return talk_please["ru" if not isTranslate else "en"]

def get_text_recognition_error_message(isTranslate):
    return text_recognition_error_messages["ru" if not isTranslate else "en"]

def get_image_load_error_message(isTranslate):
    return image_load_error_messages["ru" if not isTranslate else "en"]

def get_no_text_recognized_message(isTranslate):
    return no_text_recognized_messages["ru" if not isTranslate else "en"]

def get_select_image_message_errors(isTranslate):
    return select_image_message_errors["ru" if not isTranslate else "en"]

def get_image_title_errors(isTranslate):
    return select_image_title_errors["ru" if not isTranslate else "en"]

def get_tesseract_not_found_messages(isTranslate):
    return tesseract_not_found_messages["ru" if not isTranslate else "en"]

def get_save_img_messages(isTranslate):
    return save_img_messages["ru" if not isTranslate else "en"]

def get_error_message(isTranslate):
    return error_messages["ru" if not isTranslate else "en"]

def get_error_gen_img_messages(isTranslate):
    return error_gen_img_messages["ru" if not isTranslate else "en"]

def communicate_with_VLM(user_input, model):
    try:
        image_path =filedialog.askopenfilename(
            title="Выберите изображение",
            filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.gif;*.bmp")]
        )
        if image_path:
            vlm_instance = VLM(model=model)
            image_base64 = vlm_instance.encode_image_to_base64(image_path)

            prompt = {
                "content": f"{user_input}",
                "image": image_base64
            }

            # Generate a response
            response = vlm_instance.chat(prompt)
            return response
    except Exception as e:
        return f"{get_error_message(app.isTranslate)}: {str(e)}"

def gen_img(user_input, model):
    try:
        # Формируем URL для запроса
        url = f"https://image.pollinations.ai/prompt/{user_input}?model={model}"
        resp = requests.get(url)

        if resp.ok:
            if not os.path.exists(img_folder):
                os.makedirs(img_folder)

            now = datetime.now()
            saved_images = []  # Список для хранения путей сохраненных изображений

            # Сохраняем изображение
            image_path = os.path.join(img_folder, f'{user_input}_{now.strftime("%d.%m.%Y_%H.%M.%S")}.png')
            with open(image_path, 'wb') as img_file:
                img_file.write(resp.content)  # Сохраняем содержимое ответа как изображение
                saved_images.append(image_path)  # Добавляем путь к сохраненному изображению в список

            return f"{get_save_img_messages(app.isTranslate)}{', '.join(saved_images)}"
        else:
            return f"{get_error_gen_img_messages(app.isTranslate)}: {resp.status_code}"
    except Exception as e:
        return f"{get_error_message(app.isTranslate)}: {str(e)}"

def communicate_with_ArtBit(user_input):
    try:
        provider = ArtbitImager(logging=False)
        images = provider.generate(user_input)
        if not os.path.exists(img_folder):
            os.makedirs(img_folder)
        provider.save(images, dir='img')
        return "Ok. Check img folder"
    except Exception as e:
        return f"{get_error_gen_img_messages(app.isTranslate)}: {str(e)}"

class ChatApp(ctk.CTk):
    def __init__(self):
        try:
            super().__init__()
            ctk.set_appearance_mode("dark")
            ctk.set_default_color_theme("green")

            # Определение пути к Tesseract в зависимости от платформы
            if platform.system() == "Windows":
                self.tesseract_cmd = resource_path('tesseract.exe')
            else:
                self.tesseract_cmd = "/usr/bin/tesseract"

            pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd
            self.is_listening = False  # Флаг для отслеживания состояния прослушивания
            self.stop_listening = None  # Объект для остановки прослушивания
            self.isTranslate = False
            self.server_process = None
            self.uvicorn_server = None
            self.api_running = False
            self.tray_icon = None
            self.tray_icon_thread = None  # Добавляем явную инициализацию
            self.local_ip = self.get_local_ip()

            self.title("AI Chat")
            self.geometry("{}x{}+0+0".format(self.winfo_screenwidth(), self.winfo_screenheight()))
            
            # Поднимаем окно на передний план
            self.lift()
            self.focus_force()  # Устанавливаем фокус на окно
            # Устанавливаем окно всегда на переднем плане
            self.attributes("-topmost", True)
            # Устанавливаем окно в полноэкранный режим
            # self.attributes("-fullscreen", True)

            # Загрузка иконки через resource_path
            if platform.system() == "Windows":
                icon_path = resource_path("icon.ico")
                self.iconbitmap(icon_path)
            else:
                icon_path = resource_path("icon.png")
                # Для Linux и macOS используем PhotoImage
                icon = PhotoImage(file=icon_path)
                self.iconphoto(True, icon)

            # Привязываем событие закрытия окна
            if platform.system() == "Windows":
                self.protocol("WM_DELETE_WINDOW", self.hide_window)

            # Инициализация TTS движка
            self.engine = None
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 170)  # Скорость речи

            # Создание виджета chat_history с прокруткой
            self.chat_history_frame = ctk.CTkFrame(self)
            self.chat_history_frame.pack(fill="both", expand=True, padx=10, pady=10)

            font_size = int(min(self.winfo_screenwidth(), self.winfo_screenheight()) / 100) + 8
            button_height = 30 if self.winfo_screenheight() >= 900 else 20

            self.chat_history = ctk.CTkTextbox(self.chat_history_frame, font=("Consolas", font_size))
            self.chat_history.pack(fill="both", expand=True)
            self.chat_history.configure(state="disabled")
            self.chat_history.configure(wrap="word")

            # Добавление контекстного меню для chat_history
            self.chat_history.bind("<Button-3>", self.show_context_menu)
            self.chat_history_context_menu = tk.Menu(self, tearoff=0)
            self.chat_history_context_menu.add_command(label="Выделить всё", command=self.select_all)
            self.chat_history_context_menu.add_command(label="Копировать", command=self.copy_text)

            self.input_frame = ctk.CTkFrame(self)
            self.input_frame.pack(fill="x", padx=10, pady=10)

            # Метка для выбора модели
            self.model_label = ctk.CTkLabel(self.input_frame, text="Выберите модель:", font=("Consolas", font_size))
            self.model_label.pack(side="left", padx=5)

            # Комбобокс для выбора модели
            self.model_var = tk.StringVar()
            self.model_combobox = ctk.CTkOptionMenu(self.input_frame, variable=self.model_var, font=("Consolas", font_size),
                                                    values=list(model_functions.keys()))
            self.model_combobox.pack(side="left", padx=5)
            self.model_combobox.set(list(model_functions.keys())[0])  # Модель по умолчанию

            # Создаем новый фрейм для выбора категории
            self.category_frame = ctk.CTkFrame(self.input_frame)
            self.category_frame.pack(side="top", padx=6)  # Устанавливаем фрейм ниже

            # Метка для категории
            self.category_label = ctk.CTkLabel(self.category_frame, text="Выберите категорию:", font=("Consolas", font_size))
            self.category_label.pack(side="left", padx=6)

            # Комбобокс для выбора категории моделей
            self.category_var = tk.StringVar()
            self.category_combobox = ctk.CTkOptionMenu(self.category_frame, variable=self.category_var,
                                                       font=("Consolas", font_size),
                                                       values=["All", "Text", "Img", "Photo Analyze", "Web"],
                                                       command=self.update_model_list)
            self.category_combobox.pack(side="left", padx=6)

            # Установка "All" как модели по умолчанию
            self.category_combobox.set("All")

            self.search_label = ctk.CTkLabel(self.category_frame, text="Поиск модели:", font=("Consolas", font_size))
            self.search_label.pack(side="left", padx=6)

            self.search_var = tk.StringVar()
            self.search_entry = ctk.CTkEntry(self.category_frame, textvariable=self.search_var, font=("Consolas", font_size))
            self.search_entry.pack(side="left", padx=6)

            self.search_entry.bind("<FocusIn>", self.set_active_widget)

            self.read_var = tk.BooleanVar()
            self.read_checkbox = ctk.CTkCheckBox(self.category_frame, text="Прочитать текст",
                                            font=("Consolas", font_size),
                                            variable=self.read_var)
            self.history_var = tk.BooleanVar()
            self.history_checkbox = ctk.CTkCheckBox(self.category_frame, text="Вести историю",
                                                    font=("Consolas", font_size), variable=self.history_var)

            if self.winfo_screenheight() < 900:
                self.read_checkbox.pack_forget()  # Скрывает чекбокс "Прочитать текст"
                self.history_checkbox.pack_forget()  # Скрывает чекбокс "Вести историю"
            else:
                # Дополнительные кнопки для больших экранов
                self.read_checkbox.pack(side="right", padx=4, pady=4)

                # Чекбокс "Вести историю"
                self.history_checkbox.pack(side="top", padx=4, pady=4)

            # Установка trace для отслеживания изменений в строке поиска
            self.search_var.trace("w", lambda *args: self.after(300, self.filter_models))

            # Обновление списка моделей при инициализации
            self.update_model_list("All")

            self.input_entry = ctk.CTkTextbox(self.input_frame, font=("Consolas", font_size), height=200, width=180, wrap="word", text_color="orange")
            self.input_entry.edit_undo()
            self.input_entry.pack(side="left", fill="x", expand=True, padx=5)

            # Добавление контекстного меню для input_entry
            self.input_entry.bind("<Button-3>", self.show_context_menu)
            self.context_menu = tk.Menu(self, tearoff=0)
            self.context_menu.add_command(label="Копировать", command=self.copy_text)
            self.context_menu.add_command(label="Выделить всё", command=self.select_all)
            self.context_menu.add_command(label="Вставить", command=self.paste_text)
            self.context_menu.add_command(label="Отменить действие", command=self.undo_input)

            # Стек для хранения истории изменений
            self.history = []

            # Горячие клавиши
            self.input_entry.bind("<Shift-Return>", self.insert_newline)
            self.input_entry.bind("<Return>", self.send_message)
            self.input_entry.bind("<KeyPress>", self.on_key_press)

            self.search_entry.bind("<KeyPress>", self.on_key_press)

            if self.winfo_screenheight() < 900:
                # Создание верхнего меню
                self.create_menu()
            # Создаем фрейм для кнопок
            self.button_frame = ctk.CTkFrame(self.input_frame)
            self.button_frame.pack(side="top", fill="x")  # Упаковываем фрейм сверху и растягиваем по ширине

            # Кнопки, которые остаются всегда видимыми
            self.send_button = ctk.CTkButton(self.button_frame, text="Отправить", command=self.send_message,
                                             font=("Consolas", font_size), text_color="white", height=button_height)
            self.send_button.pack(side="top", padx=5, pady=10)

            self.clear_button = ctk.CTkButton(self.button_frame, text="Очистить чат", command=self.clear_chat,
                                              font=("Consolas", font_size), text_color="white", height=button_height)
            self.clear_button.pack(side="top", padx=5, pady=10)

            self.speech_reco_button = ctk.CTkButton(self.button_frame, text="Голосовой ввод",
                                                    command=self.toggle_recognition,
                                                    font=("Consolas", font_size), text_color="white", height=button_height)
            self.speech_reco_button.pack(side="top", padx=5, pady=10)

            self.theme_button = ctk.CTkButton(self.button_frame, text="Светлая тема", command=self.toggle_theme,
                                              font=("Consolas", font_size), text_color="white",
                                              height=button_height)

            self.lang_button = ctk.CTkButton(self.button_frame, text="English", command=self.toggle_lang,
                                             font=("Consolas", font_size), text_color="white", height=button_height)

            self.api_mode_button = ctk.CTkButton(self.button_frame, text="API Mode", command=self.toggle_api_mode,
                                                 font=("Consolas", font_size), text_color="white",
                                                 height=button_height)

            self.img_reco_button = ctk.CTkButton(self.button_frame, text="Распознать текст",
                                                 command=self.recognize_text,
                                                 font=("Consolas", font_size), text_color="white",
                                                 height=button_height)

            self.read_file_button = ctk.CTkButton(self.button_frame, text="Открыть файл",
                                                 command=self.read_file,
                                                 font=("Consolas", font_size), text_color="white",
                                                 height=button_height)
            # Проверка высоты экрана
            if self.winfo_screenheight() >= 900:
                # Дополнительные кнопки для больших экранов

                self.img_reco_button.pack(side="top", padx=5, pady=10)

                self.read_file_button.pack(side="top", padx=5, pady=10)

                self.theme_button.pack(side="top", padx=5, pady=10)

                self.lang_button.pack(side="top", padx=5, pady=10)

                self.api_mode_button.pack(side="top", padx=5, pady=10)

            self.exit_button = ctk.CTkButton(self.button_frame, text="Выход", command=self.on_exit,
                                             font=("Consolas", font_size), text_color="white", height=button_height)
            self.exit_button.pack(side="top", padx=5, pady=10)

            # Определение тегов для цветного текста
            self.chat_history.tag_add("user_input", "1.0")
            self.chat_history.tag_add("response", "1.0")
            self.chat_history.tag_add("system_line", "1.0")

            # Определение тегов для цветного текста
            self.chat_history.tag_config("user_input", foreground="orange")
            self.chat_history.tag_config("response", foreground="yellow")
            self.chat_history.tag_config("system_line", foreground="cyan")

            # Переменная для отслеживания активного виджета
            self.active_widget = None

            # Привязываем события фокуса к виджетам
            self.chat_history.bind("<FocusIn>", self.set_active_widget)
            self.input_entry.bind("<FocusIn>", self.set_active_widget)

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def speak(self, text):
        if self.engine and self.read_var.get():  # Проверяем состояние чекбокса
            def run_tts():
                try:
                    self.engine.say(text)
                    self.engine.runAndWait()
                except Exception as e:
                   messagebox.showerror(f"Reproduction Error: {str(e)}")

            threading.Thread(target=run_tts, daemon=True).start()

    def create_menu(self):
        """Создает верхнее меню с поддержкой перевода."""
        menubar = tk.Menu(self)

        # Меню "Файл"
        file_menu = tk.Menu(menubar, tearoff=0)
        if not self.isTranslate:
            file_menu.add_command(label="Открыть файл", command=self.read_file)
            file_menu.add_command(label="Выход", command=self.on_exit)
        else:
            file_menu.add_command(label="Open file", command=self.read_file)
            file_menu.add_command(label="Exit", command=self.on_exit)
        menubar.add_cascade(label="Файл" if not self.isTranslate else "File", menu=file_menu)

        # Меню "Настройки"
        settings_menu = tk.Menu(menubar, tearoff=0)
        if not self.isTranslate:
            settings_menu.add_command(label="Переключить тему", command=self.toggle_theme)
            settings_menu.add_command(label="English", command=self.toggle_lang)
            settings_menu.add_command(label="Вести историю", command=self.toggle_history)
            settings_menu.add_command(label="Прочитать текст", command=self.toggle_read_text)
            settings_menu.add_command(label="API Mode", command=self.toggle_api_mode)
        else:
            settings_menu.add_command(label="Toggle Theme", command=self.toggle_theme)
            settings_menu.add_command(label="Russian", command=self.toggle_lang)
            settings_menu.add_command(label="Keep History", command=self.toggle_history)
            settings_menu.add_command(label="Read Text", command=self.toggle_read_text)
            settings_menu.add_command(label="API Mode", command=self.toggle_api_mode)
        menubar.add_cascade(label="Настройки" if not self.isTranslate else "Settings", menu=settings_menu)

        # Меню "Инструменты"
        tools_menu = tk.Menu(menubar, tearoff=0)
        if not self.isTranslate:
            tools_menu.add_command(label="Распознать текст", command=self.recognize_text)
        else:
            tools_menu.add_command(label="Recognize Text", command=self.recognize_text)
        menubar.add_cascade(label="Инструменты" if not self.isTranslate else "Tools", menu=tools_menu)

        self.config(menu=menubar)

    def read_file(self):
        """
        Читает содержимое файла любого поддерживаемого формата:
        .txt, .docx, .xlsx, .xls, .csv, .odt, .doc
        """
        ans = ""
        file_path = tk.filedialog.askopenfilename(filetypes=[("Text files", ".txt .doc .docx .xlsx .xls .csv .odt")])
        if not os.path.exists(file_path):
            return "Файл не найден."

        _, extension = os.path.splitext(file_path)
        extension = extension.lower()  # Унифицируем расширение

        try:
            if extension == '.txt':
                # Чтение текстового файла
                with open(file_path, 'r', encoding='utf-8') as file:
                    ans = file.read()

            elif extension == '.docx':
                # Чтение DOCX файла
                doc = Document(file_path)
                ans = ("\n".join([paragraph.text for paragraph in doc.paragraphs]))

            elif extension in ['.xlsx', '.xls']:
                # Чтение Excel файла
                try:
                    data = pd.read_excel(file_path, engine='openpyxl')  # Для xlsx
                except ValueError:
                    data = pd.read_excel(file_path)  # Для xls
                    ans = data.to_string(index=False)

            elif extension == '.csv':
                # Чтение CSV файла
                data = pd.read_csv(file_path)
                ans = data.to_string(index=False)

            elif extension == '.odt':
                # Чтение ODT файла
                doc = odf.opendocument.load(file_path)
                text_elements = []
                for paragraph in doc.getElementsByType(odf.text.P):
                    text_elements.append(paragraph.textContent)
                    ans = "\n".join(text_elements)

            elif extension == '.doc':
                # Чтение DOC файла с помощью antiword
                try:
                    result = subprocess.run(['antiword', file_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                            text=True)
                    if result.returncode == 0:
                        ans = result.stdout
                    else:
                        return f"Ошибка при чтении DOC файла: {result.stderr}"
                except FileNotFoundError:
                    return "Программа antiword не найдена. Убедитесь, что она установлена."
            else:
                return f"Не поддерживаемый формат файла: {extension}"
            self.input_entry.insert(tk.END, ans)
        except Exception as e:
            return f"Произошла ошибка: {str(e)}"

    def toggle_read_text(self):
        self.read_var.set(not self.read_var.get())

    def toggle_history(self):
        """Переключает режим ведения истории."""
        self.history_var.set(not self.history_var.get())

    def get_local_ip(self):
        try:
            # Получаем локальный IP-адрес
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
        finally:
            s.close()
        return local_ip

    def toggle_api_mode(self):
        if self.api_running:
            self.stop_api_mode()
        else:
            self.start_api_mode()

    def start_api_mode(self):
        def run_fastapi_app():
            app = FastAPI()

            # Разрешаем CORS для всех источников (можно ограничить по необходимости)
            app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )

            # Модель данных для запроса к /api/gpt/ans
            class MessageRequest(BaseModel):
                model: str
                message: str

            # Определение маршрута для получения списка моделей
            @app.get("/api/ai/models")
            def get_models():
                filtered_models = [model for model in model_functions.keys() if
                                   "_img" not in model and "(Photo Analyze)" not in model]
                return {"models": filtered_models}

            # Определение маршрута для получения ответа от модели
            @app.post("/api/gpt/ans")
            def get_answer(request: MessageRequest):
                model = request.model
                message = request.message

                if model not in model_functions:
                    raise HTTPException(status_code=404, detail="Model not found")

                response = model_functions[model](message)
                return {"response": response}

            # Статические файлы
            static_dir = resource_path("static")
            app.mount("/static", StaticFiles(directory=static_dir), name="static")

            # Главная страница чата
            @app.get("/chat")
            def read_chat():
                html_path = resource_path(os.path.join("static", "chat.html"))
                with open(html_path, "r", encoding="utf-8") as file:
                    return HTMLResponse(content=file.read())

            sys.stdout = open('server_log.txt', 'w')
            sys.stderr = sys.stdout
            config = uvicorn.Config(app, host=self.local_ip, port=8000, log_level="info")
            self.uvicorn_server = uvicorn.Server(config=config)
            self.uvicorn_server.run()

        self.server_process = threading.Thread(target=run_fastapi_app, daemon=True)
        self.server_process.start()
        server_url = f"http://{self.local_ip}:8000/chat"
        messagebox.showwarning("Server started", get_API_message(self.isTranslate))
        webbrowser.open(server_url)
        self.api_running = True
        self.api_mode_button.configure(text="Stop API Mode")

    def stop_api_mode(self):
        if self.uvicorn_server:
            self.uvicorn_server.should_exit = True
            self.uvicorn_server.force_exit = True
        self.api_running = False
        sys.stdout.close()
        sys.stdout = sys.__stdout__
        sys.stderr.close()
        sys.stderr = sys.__stderr__
        os.remove('server_log.txt')
        self.api_mode_button.configure(text="API Mode")

    def toggle_recognition(self):
        if self.is_listening:
            self.is_listening = False  # Остановить распознавание
            if self.stop_listening:
                self.stop_listening()  # Немедленно остановить прослушивание
        else:
            self.is_listening = True  # Начать распознавание
            self.stop_listening = None  # Сбрасываем объект остановки
            threading.Thread(target=self.recognize_speech).start()  # Запускаем распознавание в отдельном потоке

    def recognize_speech(self):
        recognizer = sr.Recognizer()

        # Проверяем доступность микрофона
        mic_list = sr.Microphone.list_microphone_names()
        if not mic_list:
            messagebox.showerror("Error", get_micro_error_message(self.isTranslate))
            return

        with sr.Microphone() as source:
            self.input_entry.delete("1.0", tk.END)
            self.input_entry.insert("1.0", get_talk_please(self.isTranslate))
            isTextDelete = False
            while self.is_listening:
                try:
                    # Слушаем звук
                    audio = recognizer.listen(source, timeout=3)  # Установите таймаут, чтобы избежать зависания
                    # Определяем язык в зависимости от переменной isTranslate
                    language = 'en-US' if self.isTranslate else 'ru-RU'
                    # Распознаем речь с помощью Google Web Speech API
                    text = recognizer.recognize_google(audio, language=language)
                    if not isTextDelete:
                        self.input_entry.delete("1.0", tk.END)
                        isTextDelete = True
                    if text and isTextDelete:
                        self.input_entry.insert(tk.END, text + " ")
                except sr.WaitTimeoutError:
                    # Если таймаут истек, завершаем прослушивание
                    self.is_listening = False
                    return
                except sr.RequestError as e:
                    self.input_entry.delete("1.0", tk.END)
                    self.input_entry.insert("1.0", "Request to the speech recognition service failed.")
                except Exception:
                    return


    def filter_models(self, *args):
        search_term = self.search_var.get().lower()
        filtered_models = [model for model in model_functions.keys() if search_term in model.lower()]
        self.model_combobox.configure(values=filtered_models)
        if filtered_models:
            self.model_combobox.set(filtered_models[0])  # Установка первой
        else:
            self.model_combobox.set("")  # Установка первой найденной модели по умолчанию

    def recognize_text(self):
        try:
            # Проверяем наличие Tesseract в зависимости от платформы
            if platform.system() == "Windows":
                tesseract_path = self.tesseract_cmd  # Используем путь из resource_path
            else:
                tesseract_path = "/usr/bin/tesseract"  # Стандартный путь на Linux

            if not os.path.exists(tesseract_path):
                messagebox.showerror("Error", "Tesseract не найден. Убедитесь, что он установлен.")
                return

            # Устанавливаем путь к Tesseract для pytesseract
            pytesseract.pytesseract.tesseract_cmd = tesseract_path

            # Открываем диалог выбора изображения
            image_path = filedialog.askopenfilename(
                title="Выберите изображение",
                filetypes=(("Изображения", "*.jpg;*.png;*.gif"), ("Все файлы", "*.*"))
            )

            if image_path:
                # Загрузка изображения
                image = cv2.imread(image_path)
                if image is None:
                    messagebox.showerror(
                        get_image_title_errors(app.isTranslate),
                        get_image_load_error_message(app.isTranslate)
                    )
                    return

                # Преобразование изображения в оттенки серого
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                # Применение порогового значения для выделения текста
                _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

                # Использование pytesseract для распознавания текста
                recognized_text = pytesseract.image_to_string(thresh, lang='rus+eng')

                if recognized_text.strip():
                    self.input_entry.delete("1.0", tk.END)
                    self.input_entry.insert("1.0", recognized_text)
                else:
                    messagebox.showinfo("Result", get_no_text_recognized_message(app.isTranslate))
            else:
                messagebox.showerror(
                    get_image_title_errors(app.isTranslate),
                    get_select_image_message_errors(app.isTranslate)
                )
        except Exception as e:
            # Получаем информацию об ошибке
            error_message = f"{get_text_recognition_error_message(app.isTranslate)}\n{str(e)}\n\n"
            error_message += traceback.format_exc()
            messagebox.showerror("Error", error_message)

    def update_model_list(self, category):
        # Фильтрация моделей в зависимости от выбранной категории
        filtered_models = []
        for model in model_functions.keys():
            if category == "All":
                filtered_models.append(model)
            elif category == "Text" and not model.endswith("_img"):
                filtered_models.append(model)
            elif category == "Img" and model.endswith("_img"):
                filtered_models.append(model)
            elif category == "Photo Analyze" and "(Photo Analyze)" in model:
                filtered_models.append(model)
            elif category == "Web" and "_Web" in model:
                filtered_models.append(model)

        # Сохраняем текущее значение комбобокса
        current_selection = self.model_combobox.get()

        # Обновление комбобокса с моделями
        self.model_combobox.configure(values=filtered_models)

        # Восстанавливаем предыдущий выбор, если он есть в новом списке
        if current_selection in filtered_models:
            self.model_combobox.set(current_selection)
        else:
            # Устанавливаем первую модель из нового списка, если текущая не найдена
            self.model_combobox.set(filtered_models[0] if filtered_models else "")

        # Добавляем небольшую задержку для плавного обновления
        self.update_idletasks()  # Обновляем интерфейс

    def set_active_widget(self, event):
        # Устанавливаем активный виджет
        self.active_widget = event.widget

    def create_tray_icon(self):
        """Создает новый экземпляр иконки трея"""
        if not self.isTranslate:

            menu = (
                pystray.MenuItem("Открыть", self.show_window, default=True),
                pystray.MenuItem("API Mode", self.toggle_api_mode),
                pystray.MenuItem("Закрыть", self.on_exit)
            )
        else:
            menu = (
                pystray.MenuItem("Open", self.show_window, default=True),
                pystray.MenuItem("API Mode", self.toggle_api_mode),
                pystray.MenuItem("Close", self.on_exit)
            )
        # Используем resource_path для иконки
        image = Image.open(resource_path("icon.ico"))
        self.tray_icon = pystray.Icon("name", image, "AI Chat", menu)

    def hide_window(self):
        """Скрытие окна в трей"""
        self.withdraw()
        if self.tray_icon_thread is None or not self.tray_icon_thread.is_alive():
            self.create_tray_icon()
            self.tray_icon_thread = threading.Thread(
                target=self.tray_icon.run,
                daemon=True)
            self.tray_icon_thread.start()

    def show_window(self):
        """
        Восстановление окна из трея.
        Адаптировано для работы на Windows и Linux.
        """
        if self.tray_icon:
            # Останавливаем текущую иконку трея (если это необходимо)
            self.tray_icon.stop()

        # Восстанавливаем окно
        self.deiconify()

        # Устанавливаем окно поверх других окон
        self.attributes('-topmost', 1)
        self.attributes('-topmost', 0)

    def on_exit(self):
        """Полное закрытие приложения"""
        os._exit(0)

    def send_message(self, event=None):
        try:
            user_input = self.input_entry.get("1.0", "end-1c")
            user_input = user_input.replace('\n', '')  # Удаляем символы новой строки
            user_input = user_input.strip()  # Удаляем пробелы в начале и конце строки
            if user_input:
                model = self.model_var.get()

                if model in model_functions:
                    self.chat_history.configure(state="normal")  # Включаем редактирование
                    response = model_functions[model](user_input)
                    if self.read_var.get():
                        self.speak(response)
                    if not self.isTranslate:
                        self.chat_history.insert(tk.END, f"Вы: {user_input}\n", "user_input")
                        self.chat_history.insert(tk.END, f"\nОтвет от {model}: {response}\n", "response")
                    else:
                        self.chat_history.insert(tk.END, f"You: {user_input}\n", "user_input")
                        self.chat_history.insert(tk.END, f"\nAnswer from {model}: {response}\n", "response")

                    # Получаем ширину виджета chat_history
                    chat_width = self.chat_history.winfo_width()

                    # Получаем шрифт и ширину символа "="
                    font = tkFont.Font(font=self.chat_history.cget("font"))
                    equals_width = font.measure('=')

                    # Рассчитываем количество символов "=" для заполнения ширины
                    num_equals = chat_width // (equals_width - 3)  # Учитываем отступы
                    # Вставляем символы "="
                    self.chat_history.insert(tk.END, (num_equals - 3) * "=", "system_line")
                    self.chat_history.insert(tk.END, "\n", "system_line")
                    self.chat_history.configure(state="disabled")  # Возвращаем в состояние "disabled"

                    if self.history_var.get():
                        self.write_history(user_input, response)

                    self.input_entry.delete("1.0", "end-1c")  # Очистка поля ввода
        except Exception as e:
            messagebox.showerror("Возникла ошибка", str(e))

    def record_history(self, event=None):
        """Записываем текущее состояние текста в историю."""
        current_text = self.input_entry.get("1.0", "end-1c")
        if self.history and self.history[-1] == current_text:
            return  # Не добавляем одинаковые состояния
        self.history.append(current_text)

    def undo_input(self, event=None):
        """Функция для отмены последнего действия в поле ввода."""
        try:
            self.input_entry.configure(state="normal")
            if self.history:  # Проверяем, есть ли история
                # Удаляем текущее состояние из истории
                self.history.pop()  # Удаляем текущее состояние
                if self.history:  # Если в истории есть предыдущее состояние
                    previous_text = self.history[-1]  # Получаем предыдущее состояние
                    self.input_entry.delete("1.0", "end")  # Очищаем текущее содержимое
                    self.input_entry.insert("1.0", previous_text)  # Вставляем предыдущее состояние
                else:
                    self.input_entry.delete("1.0", "end")  # Если истории больше нет, очищаем текстовое поле
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

    def on_key_press(self, event):
        # Вывод отладочной информации
        # print(f"Нажата клавиша: {event.keysym}, состояние: {event.state}, символ: {event.char}, код клавиши: {event.keycode}")
        # print(f"Текущий виджет: {self.active_widget}")
        self.record_history(event) # История ввода

        # Проверка нажатия Ctrl
        if event.state & 0x4:  # Проверка, удерживается ли Ctrl
            # Проверка нажатия Ctrl + A
            if event.keycode == 65:  # Ctrl + A (или Ctrl + Ф)
                # print("CTRL + A")
                self.select_all()
                return "break"
            # Проверка нажатия Ctrl + Z
            elif event.keycode == 90:  # Ctrl + Z (или Ctrl + Я)
                # print("CTRL + Z")
                self.undo_input()
                return "break"
            # Проверка нажатия Ctrl + C
            elif event.keycode == 67:  # Ctrl + C (или Ctrl + С)
                # print("CTRL + C")
                self.copy_text()
                return "break"
            # Проверка нажатия Ctrl + V
            elif event.keycode == 86:  # Ctrl + V (или Ctrl + В)
                # print("CTRL + V")
                self.paste_text()
                return "break"

    def select_all(self, event=None):
        try:
            self.chat_history.configure(state="normal")  # Включаем редактирование
            # Получаем текущий виджет, имеющий фокус
            current_widget = self.active_widget
            # print(f"Текущий виджет с фокусом: {current_widget}")

            if str(current_widget) == '.!ctkframe.!ctktextbox.!text':
                # Если фокус на chat_history, выделяем весь текст в нем
                if self.chat_history.get("1.0", "end-1c").strip():  # Убираем пробелы
                    self.chat_history.tag_add("sel", "1.0", "end-1c")
                    self.chat_history.mark_set("insert", "1.0")  # Устанавливаем курсор в начало
            elif str(current_widget) == '.!ctkframe2.!ctktextbox.!text':
                # Если фокус на input_entry, выделяем весь текст в нем
                if self.input_entry.get("1.0", "end-1c").strip():  # Убираем пробелы
                    self.input_entry.tag_add("sel", "1.0", "end-1c")  # Выделяем весь текст
                    self.input_entry.mark_set("insert", "1.0")  # Устанавливаем курсор в начало
            elif str(current_widget) == '.!ctkframe2.!ctkframe.!ctkentry.!entry':
                # Для CTkEntry используем стандартные методы выделения текста
                self.search_entry.select_range(0, "end")
                self.search_entry.icursor("end")  # Устанавливаем курсор в конец
            self.chat_history.configure(state="disable")  # Включаем редактирование
            return "break"
        except Exception as e:
            messagebox.showerror("Возникла ошибка", f"Ошибка при выделении текста: {e}")

    def show_context_menu(self, event):
        try:
            # print(f"Метод вызван, виджет: {event.widget}")
            if str(event.widget) == '.!ctkframe.!ctktextbox.!text':
                self.chat_history_context_menu.post(event.x_root, event.y_root)
            elif str(event.widget) == '.!ctkframe2.!ctktextbox.!text':
                self.context_menu.post(event.x_root, event.y_root)
        except Exception as e:
            messagebox.showerror("Возникла ошибка", e)

    def insert_newline(self, event):
        try:
            self.input_entry.insert("insert", "\n")
            self.input_entry.edit_separator()
            self.input_entry.edit_modified(False)
            return "break"
        except Exception as e:
            messagebox.showerror("Возникла ошибка", e)

    def write_history(self, user_input, response):
        try:
            now = datetime.now()
            text = f"Дата и время: {now.strftime('%d.%m.%Y %H:%M:%S')}\nЗапрос пользователя: {user_input}\nОтвет ИИ: {response}\n\n{100*"="}\n"
            with open("llm_history.txt", "a", encoding="utf-8") as f:
                f.write(text)
        except Exception as e:
            messagebox.showerror("Возникла ошибка", e)

    def clear_chat(self):
        try:
            self.chat_history.configure(state="normal")
            self.chat_history.delete("1.0", "end")
            self.chat_history.configure(state="disabled")
        except Exception as e:
            messagebox.showerror("Возникла ошибка", e)

    def paste_text(self, event=None):
        try:
            # Получаем текст из буфера обмена
            clipboard_text = self.clipboard_get()
            if str(self.active_widget) == '.!ctkframe2.!ctkframe.!ctkentry.!entry':
                # Вставляем текст в поле поиска
                self.search_entry.insert("insert", clipboard_text)
            else:
                # Вставляем текст в поле ввода
                self.input_entry.insert("insert", clipboard_text)

            # Возвращаем "break", чтобы предотвратить дальнейшее распространение события
            return "break"
        except tk.TclError:
            # Если в буфере обмена нет текста, ничего не делаем
            pass
        except Exception as e:
            messagebox.showerror("Возникла ошибка", e)

    def copy_text(self, event=None):
        try:
            self.chat_history.configure(state="normal")  # Включаем редактирование
            # Проверяем, есть ли выделенный текст в chat_history
            if self.chat_history.tag_ranges("sel"):
                selected_text = self.chat_history.get("sel.first", "sel.last")
                self.clipboard_clear()
                self.clipboard_append(selected_text)
            # Если в chat_history нет выделенного текста, проверяем input_entry
            elif self.input_entry.tag_ranges("sel"):
                selected_text = self.input_entry.get("sel.first", "sel.last")
                self.clipboard_clear()
                self.clipboard_append(selected_text)
            elif str(self.active_widget) == '.!ctkframe2.!ctkframe.!ctkentry.!entry':
                try:
                    selected_text = self.search_entry.selection_get()
                    self.clipboard_clear()
                    self.clipboard_append(selected_text)
                except Exception:
                    # Если текст не выделен, ничего не делаем
                    pass
            self.chat_history.configure(state="disabled")  # Возвращаем в состояние "disabled"
            return "break"
        except Exception as e:
            messagebox.showerror("Возникла ошибка", str(e))

    def toggle_theme(self):
        current_theme = ctk.get_appearance_mode()
        if current_theme == "Dark":
            ctk.set_appearance_mode("light")
            self.theme_button.configure(text="Тёмная тема")
            self.chat_history.tag_config("user_input", foreground="orange")
            self.chat_history.tag_config("response", foreground="#4F2982")
            self.chat_history.tag_config("system_line", foreground="#000080")
        else:
            ctk.set_appearance_mode("dark")
            self.theme_button.configure(text="Светлая тема")
            self.chat_history.tag_config("user_input", foreground="orange")
            self.chat_history.tag_config("response", foreground="yellow")
            self.chat_history.tag_config("system_line", foreground="cyan")

    def toggle_lang(self):
        if self.isTranslate:
            # Переключаем на русский
            self.model_label.configure(text="Выберите модель:")
            self.category_label.configure(text="Выберите категорию:")
            self.send_button.configure(text="Отправить")
            self.clear_button.configure(text="Очистить чат")
            self.theme_button.configure(text="Светлая тема")
            self.lang_button.configure(text="English")
            self.read_file_button.configure(text="Открыть файл")
            self.history_checkbox.configure(text="Вести историю")
            self.exit_button.configure(text="Выход")
            self.context_menu.add_command(label="Копировать", command=self.copy_text)
            self.context_menu.add_command(label="Выделить всё", command=self.select_all)
            self.context_menu.add_command(label="Вставить", command=self.paste_text)
            self.context_menu.add_command(label="Отменить действие", command=self.undo_input)
            self.chat_history_context_menu.add_command(label="Копировать", command=self.copy_text)
            self.chat_history_context_menu.add_command(label="Выделить всё", command=self.select_all)
            self.img_reco_button.configure(text="Распознать текст")
            self.search_label.configure(text="Поиск модели:")
            self.speech_reco_button.configure(text="Голосовой ввод")
            self.read_checkbox.configure(text="Прочитать текст")
        else:
            # Переключаем на английский
            self.model_label.configure(text="Select model:")
            self.category_label.configure(text="Select category:")
            self.send_button.configure(text="Send")
            self.clear_button.configure(text="Clear chat")
            self.theme_button.configure(text="Light theme")
            self.lang_button.configure(text="Русский")
            self.history_checkbox.configure(text="Keep history")
            self.read_file_button.configure(text="Open file")
            self.exit_button.configure(text="Exit")
            self.context_menu.add_command(label="Copy", command=self.copy_text)
            self.context_menu.add_command(label="Select All", command=self.select_all)
            self.context_menu.add_command(label="Paste", command=self.paste_text)
            self.context_menu.add_command(label="Undo", command=self.undo_input)
            self.chat_history_context_menu.add_command(label="Copy", command=self.copy_text)
            self.chat_history_context_menu.add_command(label="Select All", command=self.select_all)
            self.img_reco_button.configure(text="Recognize text")
            self.search_label.configure(text="Model Search:")
            self.speech_reco_button.configure(text="Voice input")
            self.read_checkbox.configure(text="Read text")

        self.isTranslate = not self.isTranslate  # Переключаем состояние
        self.create_menu()  # Пересоздаем меню после изменения языка


if __name__ == "__main__":
    check_for_updates()
    # Получаем текстовые модели
    chat_model_functions = get_Polinations_chat_models()
    # Получаем модели изображений
    img_model_functions = get_Polinations_img_models()
    # Объединяем словари
    model_functions.update(chat_model_functions)
    model_functions.update(img_model_functions)
    app = ChatApp()
    app.mainloop()