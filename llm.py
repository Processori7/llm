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

from webscout import KOBOLDAI, DeepSeek, BLACKBOXAI, HuggingFaceChat, YouChat, FreeAIChat, Venice, HeckAI, AllenAI, WiseCat, JadveOpenAI, PerplexityLabs, ElectronHub, Felo, PhindSearch, VLM, TurboSeek, Netwrck, QwenLM, Marcus, WEBS as w
from webscout.Provider.AISEARCH import Isou
from webscout.Provider.TTI.aiarta import AIArtaImager
from webscout import FastFluxImager
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
from dotenv import load_dotenv
from pyzbar.pyzbar import decode

# Загружаем переменные из .env файла в окружение
load_dotenv()

FONT_SIZE_KEY = "FONT_SIZE"
HOST_KEY = "HOST"
PORT_KEY = "PORT"
MODEL_KEY = "MODEL"
IS_TRANSLATE_KEY = "IS_TRANSLATE"
IMG_FOLDER_KEY = "IMG_FOLDER"
MODE_KEY = "MODE"
DEFAULT_COLOR_THEM_KEY = "DEFAULT_COLOR_THEM"
WRITE_HISTORY_KEY = "WRITE_HISTORY"
ELECTRON_API_KEY = "ELECTRON_API_KEY"

# Инициализируем переменные значениями по умолчанию (_val добавлено к имени переменной)
font_size_val  = None
host_val  = None
port_val = None
model_val  = None
isTranslate_val  = None
img_folder_val  = None
mode_val  = None
def_color_them_val  = None
write_history_val = None
electron_api_key_val = None

# Проверяем, существует ли файл .env и считываем значения переменных
if os.path.exists(".env"):
    font_size_val = os.getenv(FONT_SIZE_KEY)
    host_val = os.getenv(HOST_KEY)
    port_val = os.getenv(PORT_KEY)
    model_val = os.getenv(MODEL_KEY)
    isTranslate_val = os.getenv(IS_TRANSLATE_KEY)
    img_folder_val = os.getenv(IMG_FOLDER_KEY)
    mode_val = os.getenv(MODE_KEY)
    def_color_them_val = os.getenv(DEFAULT_COLOR_THEM_KEY)
    write_history_val = os.getenv(WRITE_HISTORY_KEY)
    electron_api_key_val = os.getenv(ELECTRON_API_KEY)

# Скрываем сообщения от Pygame
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

CURRENT_VERSION = "1.54"

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

if img_folder_val is not None:
    img_folder = img_folder_val
else:
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

def communicate_with_FastFluxImager(user_input, model):
    provider = FastFluxImager()
    try:
        images = provider.generate(user_input, model=model, amount=1, size="1_1")
        paths = provider.save(images, dir=img_folder)
        return paths
    except Exception as e:
        return f"{get_error_message(app.isTranslate)}: {str(e)}"

def communicate_with_AiArta(user_input, model):
    provider = AIArtaImager()
    try:
        images = provider.generate(user_input, model=model, amount=1)
        paths = provider.save(images, dir=img_folder)
        return paths
    except Exception as e:
        return f"{get_error_message(app.isTranslate)}: {str(e)}"

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

def get_ElectronHub_credits():
    url = "https://api.electronhub.top/user/me"

    headers = {
        "accept": "*/*",
        "accept-encoding": "gzip, deflate, br, zstd",
        "accept-language": "ru-RU,ru;q=0.8",
        "authorization": "Bearer ek-wzXf1DtIewOj0VnwFxYkIuz1gvvcxeSjosfJoiomIjtJyI1Qc2",
        "cache-control": "no-cache",
        "dnt": "1",
        "origin": "https://playground.electronhub.top",
        "pragma": "no-cache",
        "referer": "https://playground.electronhub.top/",
        "sec-ch-ua": '"Chromium";v="134", "Not:A-Brand";v="24", "Brave";v="134"',
        "sec-ch-ua-mobile": "?1",
        "sec-ch-ua-platform": '"Android"',
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-site",
        "sec-gpc": "1",
        "user-agent": "Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Mobile Safari/537.36"
    }

    response = requests.get(url, headers=headers)

    if response.ok:
        try:
            data = response.json()
            credits = data.get("credits")
            return int(credits)
        except Exception as e:
            return f"{get_error_message(app.isTranslate)}: {str(e)}"

def communicate_with_ElectronHub(user_input, model):
    try:
        ai = ElectronHub(api_key=electron_api_key_val)
        ai.model = model
        ai.system_prompt = prompt
        response = ai.chat(user_input, stream=True)
        full_response = ""
        for chunk in response:
            full_response += chunk
        credits = get_ElectronHub_credits()
        full_response = full_response + f"\nBalance: {credits}"
        return full_response
    except Exception as e:
        return f"{get_error_message(app.isTranslate)}: {str(e)}"

def communicate_with_PerplexityLabs(user_input, model):
    try:
        ai = PerplexityLabs()
        ai.model = model
        ai.system_prompt = prompt
        response = ai.chat(user_input, stream=False)
        return response
    except Exception as e:
        return f"{get_error_message(app.isTranslate)}: {str(e)}"

def communicate_with_JadveOpenAI(user_input, model):
    try:
        ai = JadveOpenAI()
        ai.model = model
        ai.system_prompt = prompt
        response = ai.chat(user_input, stream=False)
        return response
    except Exception as e:
        return f"{get_error_message(app.isTranslate)}: {str(e)}"

def communicate_with_WiseCat(user_input, model):
    try:
        ai = WiseCat()
        ai.model = model
        ai.system_prompt = prompt
        response = ai.chat(user_input, stream=False)
        return response
    except Exception as e:
        return f"{get_error_message(app.isTranslate)}: {str(e)}"

def communicate_with_AllenAI(user_input, model):
    try:
        ai = AllenAI()
        ai.model = model
        ai.system_prompt = prompt
        response = ai.chat(user_input, stream=False)
        return response
    except Exception as e:
        return f"{get_error_message(app.isTranslate)}: {str(e)}"

def communicate_with_HeckAI(user_input, model):
    try:
        ai = HeckAI()
        ai.model = model
        ai.system_prompt = prompt
        response = ai.chat(user_input, stream=False)
        response = ai.fix_encoding(response)
        return response
    except Exception as e:
        return f"{get_error_message(app.isTranslate)}: {str(e)}"

def communicate_with_Venice(user_input, model):
    try:
        ai = Venice()
        ai.model = model
        ai.system_prompt = prompt
        response = ai.chat(user_input, stream=False)
        return response
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

def communicate_with_FreeAIChat(user_input, model):
    try:
        ai = FreeAIChat()
        ai.model = model
        ai.system_prompt = prompt
        response = ai.chat(user_input, stream=False)
        response = ai.fix_encoding(response)
        return response
    except Exception as e:
        return f"{get_error_message(app.isTranslate)}: {str(e)}"

def communicate_with_ISou(user_input, model):
    try:
        ai = Isou()
        ai.model = model
        response = ai.search(user_input, stream=True, raw=True)
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

def communicate_with_DeepSeek(user_input, model):
    try:
        ai = DeepSeek()
        ai.model = model
        ai.system_prompt = prompt
        response = ai.chat(user_input, stream=False)
        return response
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
"Claude-3-haiku (DDG)": lambda user_input: communicate_with_DuckDuckGO(user_input, "claude-3-haiku"),
"gpt_4_5 (YouChat)": lambda user_input: communicate_with_YouChat(user_input, "gpt_4_5"),
"claude_3_7_sonnet (YouChat)": lambda user_input: communicate_with_YouChat(user_input, "claude_3_7_sonnet"),
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
"qvq-32b (Qwenlm)":lambda user_input: communicate_with_Qwenlm(user_input, "qvq-32b", "t2t"),
"qwen-vl-max-latest (Qwenlm)":lambda user_input: communicate_with_Qwenlm(user_input, "qwen-vl-max-latest", "t2t"),
"(Qwenlm) qwen-plus-latest_Web":lambda user_input: communicate_with_Qwenlm(user_input, "qwen-plus-latest", "search"),
"(Qwenlm) qwen-turbo-latest_Web":lambda user_input: communicate_with_Qwenlm(user_input, "qwen-turbo-latest", "search"),
"(Qwenlm) qwen-max-latest_Web":lambda user_input: communicate_with_Qwenlm(user_input, "qwen-max-latest", "search"),
"(Qwenlm) qvq-72b-preview_Web":lambda user_input: communicate_with_Qwenlm(user_input, "qvq-72b-preview", "search"),
"(Qwenlm) qvq-32b-preview_Web":lambda user_input: communicate_with_Qwenlm(user_input, "qvq-32b-preview", "search"),
"deepseek-v3 (Deepseek)":lambda user_input: communicate_with_DeepSeek(user_input, "deepseek-v3"),
"deepseek-r1 (Deepseek)":lambda user_input: communicate_with_DeepSeek(user_input, "deepseek-r1"),
"deepseek-llm-67b-chat (Deepseek)":lambda user_input: communicate_with_DeepSeek(user_input, "deepseek-llm-67b-chat"),
"gpt-3.5-turbo (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "gpt-3.5-turbo"),
"gpt-3.5-turbo-16k (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "gpt-3.5-turbo-16k"),
"gpt-3.5-turbo-1106 (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "gpt-3.5-turbo-1106"),
"gpt-3.5-turbo-0125 (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "gpt-3.5-turbo-0125"),
"gpt-4 (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "gpt-4"),
"gpt-4-turbo (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "gpt-4-turbo"),
"gpt-4-turbo-preview (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "gpt-4-turbo-preview"),
"gpt-4-0125-preview (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "gpt-4-0125-preview"),
"gpt-4-1106-preview (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "gpt-4-1106-preview"),
"gpt-4o (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "gpt-4o"),
"gpt-4o-2024-05-13 (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "gpt-4o-2024-05-13"),
"gpt-4o-2024-08-06 (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "gpt-4o-2024-08-06"),
"gpt-4o-2024-11-20 (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "gpt-4o-2024-11-20"),
"gpt-4o-mini (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "gpt-4o-mini"),
"gpt-4o-mini-2024-07-18 (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "gpt-4o-mini-2024-07-18"),
"chatgpt-4o-latest (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "chatgpt-4o-latest"),
"gpt-4.5-preview (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "gpt-4.5-preview"),
"gpt-4.5-preview-2025-02-27 (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "gpt-4.5-preview-2025-02-27"),
"o1-mini (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "o1-mini"),
"o1-preview (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "o1-preview"),
"o1 (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "o1"),
"o1-low (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "o1-low"),
"o1-high (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "o1-high"),
"o3-mini (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "o3-mini"),
"o3-mini-low (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "o3-mini-low"),
"o3-mini-high (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "o3-mini-high"),
"o3-mini-online (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "o3-mini-online"),
"claude-2 (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "claude-2"),
"claude-2.1 (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "claude-2.1"),
"claude-3-opus-20240229 (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "claude-3-opus-20240229"),
"claude-3-sonnet-20240229 (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "claude-3-sonnet-20240229"),
"claude-3-haiku-20240307 (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "claude-3-haiku-20240307"),
"claude-3-5-sonnet-20240620 (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "claude-3-5-sonnet-20240620"),
"claude-3-5-sonnet-20241022 (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "claude-3-5-sonnet-20241022"),
"claude-3-5-haiku-20241022 (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "claude-3-5-haiku-20241022"),
"claude-3-7-sonnet-20250219 (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "claude-3-7-sonnet-20250219"),
"claude-3-7-sonnet-20250219-thinking (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "claude-3-7-sonnet-20250219-thinking"),
"gemini-1.0-pro (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "gemini-1.0-pro"),
"gemini-1.5-pro (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "gemini-1.5-pro"),
"gemini-1.5-pro-latest (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "gemini-1.5-pro-latest"),
"gemini-1.5-flash-8b (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "gemini-1.5-flash-8b"),
"gemini-1.5-flash (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "gemini-1.5-flash"),
"gemini-1.5-flash-latest (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "gemini-1.5-flash-latest"),
"gemini-1.5-flash-exp (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "gemini-1.5-flash-exp"),
"gemini-1.5-flash-online (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "gemini-1.5-flash-online"),
"gemini-exp-1206 (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "gemini-exp-1206"),
"learnlm-1.5-pro-experimental (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "learnlm-1.5-pro-experimental"),
"gemini-2.0-flash-001 (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "gemini-2.0-flash-001"),
"gemini-2.0-flash-exp (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "gemini-2.0-flash-exp"),
"gemini-2.0-flash-thinking-exp (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "gemini-2.0-flash-thinking-exp"),
"gemini-2.0-flash-thinking-exp-1219 (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "gemini-2.0-flash-thinking-exp-1219"),
"gemini-2.0-flash-thinking-exp-01-21 (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "gemini-2.0-flash-thinking-exp-01-21"),
"gemini-2.0-flash-lite-preview-02-05 (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "gemini-2.0-flash-lite-preview-02-05"),
"gemini-2.0-flash-lite-001 (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "gemini-2.0-flash-lite-001"),
"gemini-2.0-pro-exp-02-05 (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "gemini-2.0-pro-exp-02-05"),
"palm-2-chat-bison (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "palm-2-chat-bison"),
"palm-2-codechat-bison (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "palm-2-codechat-bison"),
"palm-2-chat-bison-32k (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "palm-2-chat-bison-32k"),
"palm-2-codechat-bison-32k (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "palm-2-codechat-bison-32k"),
"llama-2-13b-chat (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "llama-2-13b-chat"),
"llama-2-70b-chat (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "llama-2-70b-chat"),
"llama-guard-3-8b (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "llama-guard-3-8b"),
"code-llama-34b-instruct (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "code-llama-34b-instruct"),
"llama-3-8b (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "llama-3-8b"),
"llama-3-70b (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "llama-3-70b"),
"llama-3.1-8b (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "llama-3.1-8b"),
"llama-3.1-70b (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "llama-3.1-70b"),
"llama-3.1-405b (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "llama-3.1-405b"),
"llama-3.2-1b (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "llama-3.2-1b"),
"llama-3.2-3b (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "llama-3.2-3b"),
"llama-3.2-11b (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "llama-3.2-11b"),
"llama-3.2-90b (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "llama-3.2-90b"),
"llama-3.3-70b-instruct (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "llama-3.3-70b-instruct"),
"llama-3.1-nemotron-70b-instruct (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "llama-3.1-nemotron-70b-instruct"),
"llama-3.1-tulu-3-8b (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "llama-3.1-tulu-3-8b"),
"llama-3.1-tulu-3-70b (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "llama-3.1-tulu-3-70b"),
"llama-3.1-tulu-3-405b (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "llama-3.1-tulu-3-405b"),
"mistral-7b-instruct (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "mistral-7b-instruct"),
"mistral-tiny-latest (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "mistral-tiny-latest"),
"mistral-tiny (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "mistral-tiny"),
"mistral-tiny-2312 (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "mistral-tiny-2312"),
"mistral-tiny-2407 (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "mistral-tiny-2407"),
"mistral-small-24b-instruct-2501 (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "mistral-small-24b-instruct-2501"),
"mistral-small-latest (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "mistral-small-latest"),
"mistral-small (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "mistral-small"),
"mistral-small-2312 (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "mistral-small-2312"),
"mistral-small-2402 (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "mistral-small-2402"),
"mistral-small-2409 (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "mistral-small-2409"),
"mistral-medium-latest (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "mistral-medium-latest"),
"mistral-medium (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "mistral-medium"),
"mistral-medium-2312 (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "mistral-medium-2312"),
"mistral-large-latest (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "mistral-large-latest"),
"mistral-large-2411 (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "mistral-large-2411"),
"mistral-large-2407 (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "mistral-large-2407"),
"mistral-large-2402 (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "mistral-large-2402"),
"mixtral-8x7b (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "mixtral-8x7b"),
"mixtral-8x22b (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "mixtral-8x22b"),
"deepseek-r1 (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "deepseek-r1"),
"deepseek-r1-nitro (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "deepseek-r1-nitro"),
"deepseek-r1-distill-llama-8b (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "deepseek-r1-distill-llama-8b"),
"deepseek-r1-distill-llama-70b (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "deepseek-r1-distill-llama-70b"),
"deepseek-r1-distill-qwen-1.5b (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "deepseek-r1-distill-qwen-1.5b"),
"deepseek-r1-distill-qwen-7b (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "deepseek-r1-distill-qwen-7b"),
"deepseek-r1-distill-qwen-14b (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "deepseek-r1-distill-qwen-14b"),
"deepseek-r1-distill-qwen-32b (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "deepseek-r1-distill-qwen-32b"),
"deepseek-v3 (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "deepseek-v3"),
"deepseek-coder (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "deepseek-coder"),
"deepseek-v2.5 (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "deepseek-v2.5"),
"deepseek-vl2 (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "deepseek-vl2"),
"deepseek-llm-67b-chat (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "deepseek-llm-67b-chat"),
"deepseek-math-7b-instruct (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "deepseek-math-7b-instruct"),
"deepseek-coder-6.7b-base-awq (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "deepseek-coder-6.7b-base-awq"),
"deepseek-coder-6.7b-instruct-awq (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "deepseek-coder-6.7b-instruct-awq"),
"qwen-1.5-0.5b-chat (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "qwen-1.5-0.5b-chat"),
"qwen-1.5-1.8b-chat (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "qwen-1.5-1.8b-chat"),
"qwen-1.5-14b-chat-awq (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "qwen-1.5-14b-chat-awq"),
"qwen-1.5-7b-chat-awq (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "qwen-1.5-7b-chat-awq"),
"qwen-2-7b-instruct (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "qwen-2-7b-instruct"),
"qwen-2-72b-instruct (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "qwen-2-72b-instruct"),
"qwen-2-vl-7b-instruct (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "qwen-2-vl-7b-instruct"),
"qwen-2-vl-72b-instruct (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "qwen-2-vl-72b-instruct"),
"qwen-2.5-7b-instruct (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "qwen-2.5-7b-instruct"),
"qwen-2.5-32b-instruct (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "qwen-2.5-32b-instruct"),
"qwen-2.5-72b-instruct (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "qwen-2.5-72b-instruct"),
"qwen-2.5-coder-32b-instruct (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "qwen-2.5-coder-32b-instruct"),
"qwq-32b-preview (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "qwq-32b-preview"),
"qvq-72b-preview (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "qvq-72b-preview"),
"qwen-vl-plus (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "qwen-vl-plus"),
"qwen2.5-vl-72b-instruct (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "qwen2.5-vl-72b-instruct"),
"qwen-turbo (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "qwen-turbo"),
"qwen-plus (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "qwen-plus"),
"qwen-max (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "qwen-max"),
"phi-4 (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "phi-4"),
"phi-3.5-mini-128k-instruct (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "phi-3.5-mini-128k-instruct"),
"phi-3-medium-128k-instruct (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "phi-3-medium-128k-instruct"),
"phi-3-mini-128k-instruct (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "phi-3-mini-128k-instruct"),
"phi-2 (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "phi-2"),
"gemma-7b-it (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "gemma-7b-it"),
"gemma-2-9b-it (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "gemma-2-9b-it"),
"gemma-2-27b-it (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "gemma-2-27b-it"),
"nemotron-4-340b (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "nemotron-4-340b"),
"pixtral-large-2411 (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "pixtral-large-2411"),
"pixtral-12b (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "pixtral-12b"),
"open-mistral-nemo (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "open-mistral-nemo"),
"open-mistral-nemo-2407 (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "open-mistral-nemo-2407"),
"open-mixtral-8x22b-2404 (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "open-mixtral-8x22b-2404"),
"open-mixtral-8x7b (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "open-mixtral-8x7b"),
"codestral-mamba (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "codestral-mamba"),
"codestral-latest (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "codestral-latest"),
"codestral-2405 (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "codestral-2405"),
"codestral-2412 (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "codestral-2412"),
"codestral-2501 (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "codestral-2501"),
"codestral-2411-rc5 (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "codestral-2411-rc5"),
"ministral-3b (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "ministral-3b"),
"ministral-3b-2410 (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "ministral-3b-2410"),
"ministral-8b (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "ministral-8b"),
"ministral-8b-2410 (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "ministral-8b-2410"),
"mistral-saba-latest (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "mistral-saba-latest"),
"mistral-saba-2502 (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "mistral-saba-2502"),
"f1-mini-preview (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "f1-mini-preview"),
"f1-preview (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "f1-preview"),
"dolphin-mixtral-8x7b (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "dolphin-mixtral-8x7b"),
"dolphin-mixtral-8x22b (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "dolphin-mixtral-8x22b"),
"dolphin3.0-mistral-24b (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "dolphin3.0-mistral-24b"),
"dolphin3.0-r1-mistral-24b (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "dolphin3.0-r1-mistral-24b"),
"dbrx-instruct (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "dbrx-instruct"),
"command (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "command"),
"command-light (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "command-light"),
"command-nightly (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "command-nightly"),
"command-light-nightly (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "command-light-nightly"),
"command-r (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "command-r"),
"command-r-03-2024 (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "command-r-03-2024"),
"command-r-08-2024 (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "command-r-08-2024"),
"command-r-plus (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "command-r-plus"),
"command-r-plus-04-2024 (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "command-r-plus-04-2024"),
"command-r-plus-08-2024 (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "command-r-plus-08-2024"),
"command-r7b-12-2024 (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "command-r7b-12-2024"),
"c4ai-aya-expanse-8b (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "c4ai-aya-expanse-8b"),
"c4ai-aya-expanse-32b (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "c4ai-aya-expanse-32b"),
"reka-flash (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "reka-flash"),
"reka-core (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "reka-core"),
"grok-2 (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "grok-2"),
"grok-2-mini (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "grok-2-mini"),
"grok-beta (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "grok-beta"),
"grok-vision-beta (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "grok-vision-beta"),
"grok-2-1212 (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "grok-2-1212"),
"grok-2-vision-1212 (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "grok-2-vision-1212"),
"grok-3-early (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "grok-3-early"),
"grok-3-preview-02-24 (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "grok-3-preview-02-24"),
"sonar-deep-research (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "sonar-deep-research"),
"sonar-reasoning-pro (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "sonar-reasoning-pro"),
"sonar-reasoning (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "sonar-reasoning"),
"sonar-pro (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "sonar-pro"),
"sonar (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "sonar"),
"llama-3.1-sonar-small-128k-online (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "llama-3.1-sonar-small-128k-online"),
"llama-3.1-sonar-large-128k-online (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "llama-3.1-sonar-large-128k-online"),
"llama-3.1-sonar-huge-128k-online (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "llama-3.1-sonar-huge-128k-online"),
"llama-3.1-sonar-small-128k-chat (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "llama-3.1-sonar-small-128k-chat"),
"llama-3.1-sonar-large-128k-chat (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "llama-3.1-sonar-large-128k-chat"),
"wizardlm-2-7b (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "wizardlm-2-7b"),
"wizardlm-2-8x22b (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "wizardlm-2-8x22b"),
"minimax-01 (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "minimax-01"),
"jamba-1.5-large (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "jamba-1.5-large"),
"jamba-1.5-mini (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "jamba-1.5-mini"),
"jamba-instruct (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "jamba-instruct"),
"openchat-3.5-7b (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "openchat-3.5-7b"),
"openchat-3.6-8b (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "openchat-3.6-8b"),
"aion-1.0 (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "aion-1.0"),
"aion-1.0-mini (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "aion-1.0-mini"),
"aion-rp-llama-3.1-8b (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "aion-rp-llama-3.1-8b"),
"nova-lite-v1 (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "nova-lite-v1"),
"nova-micro-v1 (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "nova-micro-v1"),
"nova-pro-v1 (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "nova-pro-v1"),
"inflection-3-pi (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "inflection-3-pi"),
"inflection-3-productivity (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "inflection-3-productivity"),
"mytho-max-l2-13b (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "mytho-max-l2-13b"),
"deephermes-3-llama-3-8b-preview (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "deephermes-3-llama-3-8b-preview"),
"hermes-3-llama-3.1-8b (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "hermes-3-llama-3.1-8b"),
"hermes-3-llama-3.1-405b (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "hermes-3-llama-3.1-405b"),
"hermes-2-pro-llama-3-8b (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "hermes-2-pro-llama-3-8b"),
"nous-hermes-2-mixtral-8x7b-dpo (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "nous-hermes-2-mixtral-8x7b-dpo"),
"doubao-lite-4k (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "doubao-lite-4k"),
"doubao-lite-32k (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "doubao-lite-32k"),
"doubao-pro-4k (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "doubao-pro-4k"),
"doubao-pro-32k (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "doubao-pro-32k"),
"ernie-lite-8k (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "ernie-lite-8k"),
"ernie-tiny-8k (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "ernie-tiny-8k"),
"ernie-speed-8k (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "ernie-speed-8k"),
"ernie-speed-128k (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "ernie-speed-128k"),
"hunyuan-lite (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "hunyuan-lite"),
"hunyuan-standard-2025-02-10 (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "hunyuan-standard-2025-02-10"),
"hunyuan-large-2025-02-10 (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "hunyuan-large-2025-02-10"),
"glm-3-130b (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "glm-3-130b"),
"glm-4-flash (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "glm-4-flash"),
"glm-4-long (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "glm-4-long"),
"glm-4-airx (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "glm-4-airx"),
"glm-4-air (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "glm-4-air"),
"glm-4-plus (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "glm-4-plus"),
"glm-4-alltools (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "glm-4-alltools"),
"yi-vl-plus (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "yi-vl-plus"),
"yi-large (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "yi-large"),
"yi-large-turbo (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "yi-large-turbo"),
"yi-large-rag (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "yi-large-rag"),
"yi-medium (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "yi-medium"),
"yi-34b-chat-200k (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "yi-34b-chat-200k"),
"spark-desk-v1.5 (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "spark-desk-v1.5"),
"step-2-16k-exp-202412 (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "step-2-16k-exp-202412"),
"granite-3.1-2b-instruct (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "granite-3.1-2b-instruct"),
"granite-3.1-8b-instruct (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "granite-3.1-8b-instruct"),
"solar-0-70b-16bit (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "solar-0-70b-16bit"),
"mistral-nemo-inferor-12b (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "mistral-nemo-inferor-12b"),
"unslopnemo-12b (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "unslopnemo-12b"),
"rocinante-12b-v1.1 (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "rocinante-12b-v1.1"),
"rocinante-12b-v1 (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "rocinante-12b-v1"),
"sky-t1-32b-preview (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "sky-t1-32b-preview"),
"lfm-3b (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "lfm-3b"),
"lfm-7b (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "lfm-7b"),
"lfm-40b (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "lfm-40b"),
"rogue-rose-103b-v0.2 (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "rogue-rose-103b-v0.2"),
"eva-llama-3.33-70b-v0.0 (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "eva-llama-3.33-70b-v0.0"),
"eva-llama-3.33-70b-v0.1 (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "eva-llama-3.33-70b-v0.1"),
"eva-qwen2.5-72b (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "eva-qwen2.5-72b"),
"eva-qwen2.5-32b-v0.2 (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "eva-qwen2.5-32b-v0.2"),
"sorcererlm-8x22b (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "sorcererlm-8x22b"),
"mythalion-13b (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "mythalion-13b"),
"zephyr-7b-beta (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "zephyr-7b-beta"),
"zephyr-7b-alpha (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "zephyr-7b-alpha"),
"toppy-m-7b (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "toppy-m-7b"),
"openhermes-2.5-mistral-7b (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "openhermes-2.5-mistral-7b"),
"l3-lunaris-8b (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "l3-lunaris-8b"),
"llama-3.1-lumimaid-8b (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "llama-3.1-lumimaid-8b"),
"llama-3.1-lumimaid-70b (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "llama-3.1-lumimaid-70b"),
"llama-3-lumimaid-8b (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "llama-3-lumimaid-8b"),
"llama-3-lumimaid-70b (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "llama-3-lumimaid-70b"),
"llama3-openbiollm-70b (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "llama3-openbiollm-70b"),
"l3.1-70b-hanami-x1 (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "l3.1-70b-hanami-x1"),
"magnum-v4-72b (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "magnum-v4-72b"),
"magnum-v2-72b (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "magnum-v2-72b"),
"magnum-72b (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "magnum-72b"),
"mini-magnum-12b-v1.1 (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "mini-magnum-12b-v1.1"),
"remm-slerp-l2-13b (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "remm-slerp-l2-13b"),
"midnight-rose-70b (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "midnight-rose-70b"),
"athene-v2-chat (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "athene-v2-chat"),
"airoboros-l2-70b (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "airoboros-l2-70b"),
"xwin-lm-70b (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "xwin-lm-70b"),
"noromaid-20b (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "noromaid-20b"),
"violet-twilight-v0.2 (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "violet-twilight-v0.2"),
"saiga-nemo-12b (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "saiga-nemo-12b"),
"l3-8b-stheno-v3.2 (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "l3-8b-stheno-v3.2"),
"llama-3.1-8b-lexi-uncensored-v2 (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "llama-3.1-8b-lexi-uncensored-v2"),
"l3.3-70b-euryale-v2.3 (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "l3.3-70b-euryale-v2.3"),
"l3.3-ms-evayale-70b (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "l3.3-ms-evayale-70b"),
"70b-l3.3-cirrus-x1 (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "70b-l3.3-cirrus-x1"),
"l31-70b-euryale-v2.2 (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "l31-70b-euryale-v2.2"),
"l3-70b-euryale-v2.1 (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "l3-70b-euryale-v2.1"),
"fimbulvetr-11b-v2 (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "fimbulvetr-11b-v2"),
"goliath-120b (ElectronHub)": lambda user_input: communicate_with_ElectronHub(user_input, "goliath-120b"),
"r1-1776 (PerplexityLabs)": lambda user_input: communicate_with_PerplexityLabs(user_input, "r1-1776"),
"sonar-pro (PerplexityLabs)": lambda user_input: communicate_with_PerplexityLabs(user_input, "sonar-pro"),
"sonar (PerplexityLabs)": lambda user_input: communicate_with_PerplexityLabs(user_input, "sonar"),
"sonar-reasoning-pro (PerplexityLabs)": lambda user_input: communicate_with_PerplexityLabs(user_input, "sonar-reasoning-pro"),
"sonar-reasoning (PerplexityLabs)": lambda user_input: communicate_with_PerplexityLabs(user_input, "sonar-reasoning"),
"gpt-4o (JadveOpenAI)": lambda user_input: communicate_with_JadveOpenAI(user_input, "gpt-4o"),
"gpt-4o-mini (JadveOpenAI)": lambda user_input: communicate_with_JadveOpenAI(user_input, "gpt-4o-mini"),
"claude-3-7-sonnet-20250219 (JadveOpenAI)": lambda user_input: communicate_with_JadveOpenAI(user_input, "claude-3-7-sonnet-20250219"),
"claude-3-5-sonnet-20240620 (JadveOpenAI)": lambda user_input: communicate_with_JadveOpenAI(user_input, "claude-3-5-sonnet-20240620"),
"o1-mini (JadveOpenAI)": lambda user_input: communicate_with_JadveOpenAI(user_input, "o1-mini"),
"deepseek-chat (JadveOpenAI)": lambda user_input: communicate_with_JadveOpenAI(user_input, "deepseek-chat"),
"claude-3-5-haiku-20241022 (JadveOpenAI)": lambda user_input: communicate_with_JadveOpenAI(user_input, "claude-3-5-haiku-20241022"),
"chat-model-small (WiseCat)": lambda user_input: communicate_with_WiseCat(user_input, "chat-model-small"),
"chat-model-large (WiseCat)": lambda user_input: communicate_with_WiseCat(user_input, "chat-model-large"),
"chat-model-reasoning (WiseCat)": lambda user_input: communicate_with_WiseCat(user_input, "chat-model-reasoning"),
"tulu3-405b (AllenAI)": lambda user_input: communicate_with_AllenAI(user_input, "tulu3-405b"),
"OLMo-2-1124-13B-Instruct (AllenAI)": lambda user_input: communicate_with_AllenAI(user_input, "OLMo-2-1124-13B-Instruct"),
"tulu-3-1-8b (AllenAI)": lambda user_input: communicate_with_AllenAI(user_input, "tulu-3-1-8b"),
"Llama-3-1-Tulu-3-70B (AllenAI)": lambda user_input: communicate_with_AllenAI(user_input, "Llama-3-1-Tulu-3-70B"),
"olmoe-0125 (AllenAI)": lambda user_input: communicate_with_AllenAI(user_input, "olmoe-0125"),
"deepseek-chat (HeckAI)": lambda user_input: communicate_with_HeckAI(user_input, "deepseek/deepseek-chat"),
"gpt-4o-mini (HeckAI)": lambda user_input: communicate_with_HeckAI(user_input, "openai/gpt-4o-mini"),
"deepseek-r1 (HeckAI)": lambda user_input: communicate_with_HeckAI(user_input, "deepseek/deepseek-r1"),
"gemini-2.0-flash-001 (HeckAI)": lambda user_input: communicate_with_HeckAI(user_input, "google/gemini-2.0-flash-001"),
"mistral-nemo (FreeAiChat)": lambda user_input: communicate_with_FreeAIChat(user_input, "mistral-nemo"),
"mistral-large (FreeAiChat)": lambda user_input: communicate_with_FreeAIChat(user_input, "mistral-large"),
"gemini-2.0-flash (FreeAiChat)": lambda user_input: communicate_with_FreeAIChat(user_input, "gemini-2.0-flash"),
"gemini-1.5-pro (FreeAiChat)": lambda user_input: communicate_with_FreeAIChat(user_input, "gemini-1.5-pro"),
"gemini-1.5-flash (FreeAiChat)": lambda user_input: communicate_with_FreeAIChat(user_input, "gemini-1.5-flash"),
"gemini-2.0-pro-exp-02-05 (FreeAiChat)": lambda user_input: communicate_with_FreeAIChat(user_input, "gemini-2.0-pro-exp-02-05"),
"deepseek-r1 (FreeAiChat)": lambda user_input: communicate_with_FreeAIChat(user_input, "deepseek-r1"),
"deepseek-v3 (FreeAiChat)": lambda user_input: communicate_with_FreeAIChat(user_input, "deepseek-v3"),
"Deepseek r1 14B (FreeAiChat)": lambda user_input: communicate_with_FreeAIChat(user_input, "Deepseek r1 14B"),
"Deepseek r1 32B (FreeAiChat)": lambda user_input: communicate_with_FreeAIChat(user_input, "Deepseek r1 32B"),
"o3-mini-high (FreeAiChat)": lambda user_input: communicate_with_FreeAIChat(user_input, "o3-mini-high"),
"o3-mini-medium (FreeAiChat)": lambda user_input: communicate_with_FreeAIChat(user_input, "o3-mini-medium"),
"o3-mini-low (FreeAiChat)": lambda user_input: communicate_with_FreeAIChat(user_input, "o3-mini-low"),
"o3-mini (FreeAiChat)": lambda user_input: communicate_with_FreeAIChat(user_input, "o3-mini"),
"GPT-4o-mini (FreeAiChat)": lambda user_input: communicate_with_FreeAIChat(user_input, "GPT-4o-mini"),
"o1 (FreeAiChat)": lambda user_input: communicate_with_FreeAIChat(user_input, "o1"),
"o1-mini (FreeAiChat)": lambda user_input: communicate_with_FreeAIChat(user_input, "o1-mini"),
"GPT-4o (FreeAiChat)": lambda user_input: communicate_with_FreeAIChat(user_input, "GPT-4o"),
"Qwen coder (FreeAiChat)": lambda user_input: communicate_with_FreeAIChat(user_input, "Qwen coder"),
"Qwen 2.5 72B (FreeAiChat)": lambda user_input: communicate_with_FreeAIChat(user_input, "Qwen 2.5 72B"),
"Llama 3.1 405B (FreeAiChat)": lambda user_input: communicate_with_FreeAIChat(user_input, "Llama 3.1 405B"),
"llama3.1-70b-fast (FreeAiChat)": lambda user_input: communicate_with_FreeAIChat(user_input, "llama3.1-70b-fast"),
"Llama 3.3 70B (FreeAiChat)": lambda user_input: communicate_with_FreeAIChat(user_input, "Llama 3.3 70B"),
"claude 3.5 haiku (FreeAiChat)": lambda user_input: communicate_with_FreeAIChat(user_input, "claude 3.5 haiku"),
"claude 3.5 sonnet (FreeAiChat)": lambda user_input: communicate_with_FreeAIChat(user_input, "claude 3.5 sonnet"),
"(Venice) llama-3.3-70b_Web": lambda user_input: communicate_with_Venice(user_input, "llama-3.3-70b"),
"(Venice) llama-3.2-3b-akash_Web": lambda user_input: communicate_with_Venice(user_input, "llama-3.2-3b-akash"),
"(Venice) qwen2dot5-coder-32b_Web": lambda user_input: communicate_with_Venice(user_input, "qwen2dot5-coder-32b"),
"(Isou) DeepSeek-R1-Distill-Qwen-32B_Web":lambda user_input: communicate_with_ISou(user_input, "siliconflow:deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"),
"(Isou) Qwen2.5-72B-Instruct-128K_Web":lambda user_input: communicate_with_ISou(user_input, "siliconflow:Qwen/Qwen2.5-72B-Instruct-128K"),
"(Isou) Deepseek-reasoner_Web":lambda user_input: communicate_with_ISou(user_input, "deepseek-reasoner"),
"KoboldAI": communicate_with_KoboldAI,
"Phind": communicate_with_Phind,
"Felo_Web": communicate_with_Felo,
"TurboSeek_Web":communicate_with_TurboSeek,
"Marcus_Web":communicate_with_Marcus,
"Searchgpt_Web(Polinations)": lambda user_input: communicate_with_Pollinations_chat(user_input, "searchgpt"),
"Llama 3.3-70B (DDG)": lambda user_input: communicate_with_DuckDuckGO(user_input, "llama-3-70b"),
"MiniCPM-Llama3-V-2_5(Photo Analyze)(VLM)":lambda user_input: communicate_with_VLM(user_input, "openbmb/MiniCPM-Llama3-V-2_5"),
"Llava-1.5-7b-hf(Photo Analyze)(VLM)":lambda user_input: communicate_with_VLM(user_input, "llava-hf/llava-1.5-7b-hf"),
"AiArta_flux_img": lambda user_input: communicate_with_AiArta(user_input, "Flux"),
"AiArta_medieval_img": lambda user_input: communicate_with_AiArta(user_input, "Medieval"),
"AiArta_vincent_van_gogh_img": lambda user_input: communicate_with_AiArta(user_input, "Vincent Van Gogh"),
"AiArta_f_dev_img": lambda user_input: communicate_with_AiArta(user_input, "F Dev"),
"AiArta_low_poly_img": lambda user_input: communicate_with_AiArta(user_input, "Low Poly"),
"AiArta_dreamshaper_xl_img": lambda user_input: communicate_with_AiArta(user_input, "Dreamshaper-xl"),
"AiArta_anima_pencil_xl_img": lambda user_input: communicate_with_AiArta(user_input, "Anima-pencil-xl"),
"AiArta_biomech_img": lambda user_input: communicate_with_AiArta(user_input, "Biomech"),
"AiArta_trash_polka_img": lambda user_input: communicate_with_AiArta(user_input, "Trash Polka"),
"AiArta_no_style_img": lambda user_input: communicate_with_AiArta(user_input, "No Style"),
"AiArta_cheyenne_xl_img": lambda user_input: communicate_with_AiArta(user_input, "Cheyenne-xl"),
"AiArta_chicano_img": lambda user_input: communicate_with_AiArta(user_input, "Chicano"),
"AiArta_embroidery_tattoo_img": lambda user_input: communicate_with_AiArta(user_input, "Embroidery tattoo"),
"AiArta_red_and_black_img": lambda user_input: communicate_with_AiArta(user_input, "Red and Black"),
"AiArta_fantasy_art_img": lambda user_input: communicate_with_AiArta(user_input, "Fantasy Art"),
"AiArta_watercolor_img": lambda user_input: communicate_with_AiArta(user_input, "Watercolor"),
"AiArta_dotwork_img": lambda user_input: communicate_with_AiArta(user_input, "Dotwork"),
"AiArta_old_school_colored_img": lambda user_input: communicate_with_AiArta(user_input, "Old school colored"),
"AiArta_realistic_tattoo_img": lambda user_input: communicate_with_AiArta(user_input, "Realistic tattoo"),
"AiArta_japanese_2_img": lambda user_input: communicate_with_AiArta(user_input, "Japanese_2"),
"AiArta_realistic_stock_xl_img": lambda user_input: communicate_with_AiArta(user_input, "Realistic-stock-xl"),
"AiArta_f_pro_img": lambda user_input: communicate_with_AiArta(user_input, "F Pro"),
"AiArta_revanimated_img": lambda user_input: communicate_with_AiArta(user_input, "RevAnimated"),
"AiArta_katayama_mix_xl_img": lambda user_input: communicate_with_AiArta(user_input, "Katayama-mix-xl"),
"AiArta_sdxl_l_img": lambda user_input: communicate_with_AiArta(user_input, "SDXL L"),
"AiArta_cor_epica_xl_img": lambda user_input: communicate_with_AiArta(user_input, "Cor-epica-xl"),
"AiArta_anime_tattoo_img": lambda user_input: communicate_with_AiArta(user_input, "Anime tattoo"),
"AiArta_new_school_img": lambda user_input: communicate_with_AiArta(user_input, "New School"),
"AiArta_death_metal_img": lambda user_input: communicate_with_AiArta(user_input, "Death metal"),
"AiArta_old_school_img": lambda user_input: communicate_with_AiArta(user_input, "Old School"),
"AiArta_juggernaut_xl_img": lambda user_input: communicate_with_AiArta(user_input, "Juggernaut-xl"),
"AiArta_photographic_img": lambda user_input: communicate_with_AiArta(user_input, "Photographic"),
"AiArta_sdxl_1_0_img": lambda user_input: communicate_with_AiArta(user_input, "SDXL 1.0"),
"AiArta_graffiti_img": lambda user_input: communicate_with_AiArta(user_input, "Graffiti"),
"AiArta_mini_tattoo_img": lambda user_input: communicate_with_AiArta(user_input, "Mini tattoo"),
"AiArta_surrealism_img": lambda user_input: communicate_with_AiArta(user_input, "Surrealism"),
"AiArta_neo_traditional_img": lambda user_input: communicate_with_AiArta(user_input, "Neo-traditional"),
"AiArta_on_limbs_black_img": lambda user_input: communicate_with_AiArta(user_input, "On limbs black"),
"AiArta_yamers_realistic_xl_img": lambda user_input: communicate_with_AiArta(user_input, "Yamers-realistic-xl"),
"AiArta_pony_xl_img": lambda user_input: communicate_with_AiArta(user_input, "Pony-xl"),
"AiArta_playground_xl_img": lambda user_input: communicate_with_AiArta(user_input, "Playground-xl"),
"AiArta_anything_xl_img": lambda user_input: communicate_with_AiArta(user_input, "Anything-xl"),
"AiArta_flame_design_img": lambda user_input: communicate_with_AiArta(user_input, "Flame design"),
"AiArta_kawaii_img": lambda user_input: communicate_with_AiArta(user_input, "Kawaii"),
"AiArta_cinematic_art_img": lambda user_input: communicate_with_AiArta(user_input, "Cinematic Art"),
"AiArta_professional_img": lambda user_input: communicate_with_AiArta(user_input, "Professional"),
"AiArta_flux_black_ink_img": lambda user_input: communicate_with_AiArta(user_input, "Flux Black Ink"),
"FastFlux_flux_1_schnell_img": lambda user_input: communicate_with_FastFluxImager(user_input, "flux_1_schnell"),
"FastFlux_flux_1_dev_img": lambda user_input: communicate_with_FastFluxImager(user_input, "flux_1_dev"),
"FastFlux_sana_1_6b_img": lambda user_input: communicate_with_FastFluxImager(user_input, "sana_1_6b")
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

def get_bool_val(val):
    if val in ["True", "true"]:
        return True
    else:
        return False

class ChatApp(ctk.CTk):
    def __init__(self):
        try:
            super().__init__()
            if mode_val is not None:
                ctk.set_appearance_mode(mode_val)
            else:
                ctk.set_appearance_mode("dark")

            if def_color_them_val is not None:
                ctk.set_default_color_theme(def_color_them_val)
            else:
                ctk.set_default_color_theme("green")

            # Определение пути к Tesseract в зависимости от платформы
            if platform.system() == "Windows":
                self.tesseract_cmd = resource_path('tesseract.exe')
            else:
                self.tesseract_cmd = "/usr/bin/tesseract"

            pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd
            self.is_listening = False  # Флаг для отслеживания состояния прослушивания
            self.stop_listening = None  # Объект для остановки прослушивания

            if isTranslate_val is not None:
                self.isTranslate = get_bool_val(isTranslate_val)
            else:
                self.isTranslate = False

            self.server_process = None
            self.uvicorn_server = None
            self.api_running = False
            self.tray_icon = None
            self.tray_icon_thread = None  # Добавляем явную инициализацию
            self.local_ip = self.get_local_ip()
            if self.isTranslate:
                self.title("AI Chat")
            else:
                self.title("Чат с ИИ")
            self.geometry("{}x{}+0+0".format(self.winfo_screenwidth(), self.winfo_screenheight()))
            if font_size_val is not None:
                font_size = int(font_size_val)
            else:
                font_size = int(min(self.winfo_screenwidth(), self.winfo_screenheight()) / 100) + 8
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
            if model_val is not None:
                self.model_combobox.set(list(model_functions.keys())[int(model_val)])
            else:
                self.model_combobox.set(list(model_functions.keys())[0])  # Модель по умолчанию
            # Создаем новый фрейм для выбора категории
            self.category_frame = ctk.CTkFrame(self.input_frame)
            self.category_frame.pack(side="top", padx=6)  # Устанавливаем фрейм ниже

            # Метка для категории
            self.category_label = ctk.CTkLabel(self.category_frame, text="Выберите категорию:", font=("Consolas", font_size))
            self.category_label.pack(side="left", padx=6)

            self.values=[]
            if self.isTranslate:
                self.values = ["All", "Text", "Img", "Photo Analyze", "Web"]
            else:
                self.values = ["Все", "Текст", "Фото", "Анализ фото", "Поиск в Интернете"]
            # Комбобокс для выбора категории моделей
            self.category_var = tk.StringVar()
            self.category_combobox = ctk.CTkOptionMenu(self.category_frame, variable=self.category_var,
                                                       font=("Consolas", font_size),
                                                       values=self.values,
                                                       command=self.update_model_list)
            self.category_combobox.pack(side="left", padx=6)

            # Установка "All" как модели по умолчанию
            self.category_combobox.set(self.values[0])

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
            self.update_model_list(self.values[0])

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

            if mode_val =="light":
                self.theme_button = ctk.CTkButton(self.button_frame, text="Тёмная тема", command=self.toggle_theme,
                                                  font=("Consolas", font_size), text_color="white",
                                                  height=button_height)
            else:
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

            self.QR_reco_button = ctk.CTkButton(self.button_frame, text="Прочитать QR Code",
                                                 command=self.read_qrcode_data,
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

                self.QR_reco_button.pack(side="top", padx=5, pady=10)

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
            if mode_val == "light":
                self.theme_button.configure(text="Тёмная тема")
                self.chat_history.tag_config("user_input", foreground="orange")
                self.chat_history.tag_config("response", foreground="#1a237e")
                self.chat_history.tag_config("system_line", foreground="#000080")
            else:
                self.theme_button.configure(text="Светлая тема")
                self.chat_history.tag_config("user_input", foreground="orange")
                self.chat_history.tag_config("response", foreground="yellow")
                self.chat_history.tag_config("system_line", foreground="cyan")

            # Переменная для отслеживания активного виджета
            self.active_widget = None

            # Привязываем события фокуса к виджетам
            self.chat_history.bind("<FocusIn>", self.set_active_widget)
            self.input_entry.bind("<FocusIn>", self.set_active_widget)
            if get_bool_val(isTranslate_val):
                self.toggle_lang()

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

    def read_qrcode_data(self):
        # Открываем диалог выбора изображения
        image_path = filedialog.askopenfilename(
            title="Выберите изображение",
            filetypes=(("Изображения", "*.jpg;*.png;*.jpeg"), ("Все файлы", "*.*"))
        )

        if not image_path:
            return

        try:
            pil_img = Image.open(image_path)
        except Exception as e:
            messagebox.showerror("Ошибка", f"Невозможно открыть изображение: {e}")

        decoded_objects = decode(pil_img)
        if decoded_objects:
            data = decoded_objects[0].data.decode('utf-8')
            self.input_entry.delete("1.0", tk.END)
            self.input_entry.insert("1.0", data)

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
            tools_menu.add_command(label="Прочитать QR Code", command=self.read_qr_code_data)
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
                allow_origins=["*"],  # Для тестирования
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
            if host_val is not None:
                config = uvicorn.Config(app, host=host_val, port=port, log_level="info")
            else:
                config = uvicorn.Config(app, host=self.local_ip, port=port, log_level="info")
            self.uvicorn_server = uvicorn.Server(config=config)
            self.uvicorn_server.run()

        port = 8000

        if port_val is not None:
            port = port_val

        self.server_process = threading.Thread(target=run_fastapi_app, daemon=True)
        self.server_process.start()
        server_url = f"http://{host_val}:{port}/chat" if host_val else f"http://{self.local_ip}:{port}/chat"
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
            if category == self.values[0]:
                filtered_models.append(model)
            elif category == self.values[1] and not model.endswith("_img"):
                filtered_models.append(model)
            elif category == self.values[2] and model.endswith("_img"):
                filtered_models.append(model)
            elif category ==self.values[3] and "(Photo Analyze)" in model:
                filtered_models.append(model)
            elif category == self.values[4] and "_Web" in model:
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
        if self.isTranslate:

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

                    if self.history_var.get() or write_history_val is not None and get_bool_val(write_history_val):
                        self.history_var.set(True)
                        self.history_checkbox.select()
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
            if not self.isTranslate:
                self.theme_button.configure(text="Dark theme")
            else:
                self.theme_button.configure(text="Тёмная тема")
            self.chat_history.tag_config("user_input", foreground="orange")
            self.chat_history.tag_config("response", foreground="#1a237e")
            self.chat_history.tag_config("system_line", foreground="#000080")
        else:
            ctk.set_appearance_mode("dark")
            if not self.isTranslate:
                self.theme_button.configure(text="Light theme")
            else:
                self.theme_button.configure(text="Светлая тема")
            self.chat_history.tag_config("user_input", foreground="orange")
            self.chat_history.tag_config("response", foreground="yellow")
            self.chat_history.tag_config("system_line", foreground="cyan")

    def toggle_lang(self):
        if self.isTranslate:  # Переключаем на английский
            self.values = ["All", "Text", "Img", "Photo Analyze", "Web"]
            self.title("AI Chat")
            self.model_label.configure(text="Select model:")
            self.category_label.configure(text="Select category:")
            self.send_button.configure(text="Send")
            self.clear_button.configure(text="Clear chat")
            self.theme_button.configure(text="Light theme")
            self.lang_button.configure(text="Русский")
            self.history_checkbox.configure(text="Keep history")
            self.read_file_button.configure(text="Open file")
            self.exit_button.configure(text="Exit")
            self.context_menu.entryconfigure(0, label="Copy")
            self.context_menu.entryconfigure(1, label="Select All")
            self.context_menu.entryconfigure(2, label="Paste")
            self.context_menu.entryconfigure(3, label="Undo")
            self.chat_history_context_menu.entryconfigure(0, label="Copy")
            self.chat_history_context_menu.entryconfigure(1, label="Select All")
            self.img_reco_button.configure(text="Recognize text")
            self.search_label.configure(text="Model Search:")
            self.QR_reco_button.configure(text="Read QR Code")
            self.speech_reco_button.configure(text="Voice input")
            self.read_checkbox.configure(text="Read text")
        else:  # Переключаем на русский
            self.values = ["Все", "Текст", "Фото", "Анализ фото", "Поиск в Интернете"]
            self.title("Чат с ИИ")
            self.model_label.configure(text="Выберите модель:")
            self.category_label.configure(text="Выберите категорию:")
            self.send_button.configure(text="Отправить")
            self.clear_button.configure(text="Очистить чат")
            self.theme_button.configure(text="Светлая тема")
            self.lang_button.configure(text="English")
            self.read_file_button.configure(text="Открыть файл")
            self.history_checkbox.configure(text="Вести историю")
            self.exit_button.configure(text="Выход")
            self.context_menu.entryconfigure(0, label="Копировать")
            self.context_menu.entryconfigure(1, label="Выделить всё")
            self.context_menu.entryconfigure(2, label="Вставить")
            self.context_menu.entryconfigure(3, label="Отменить действие")
            self.chat_history_context_menu.entryconfigure(0, label="Копировать")
            self.chat_history_context_menu.entryconfigure(1, label="Выделить всё")
            self.img_reco_button.configure(text="Распознать текст")
            self.QR_reco_button.configure(text="Прочитать QR Code")
            self.search_label.configure(text="Поиск модели:")
            self.speech_reco_button.configure(text="Голосовой ввод")
            self.read_checkbox.configure(text="Прочитать текст")
        # Обновляем список значений в комбобоксе:
        self.category_combobox.configure(values=self.values)
        # Устанавливаем значение по умолчанию.
        self.category_combobox.set(self.values[0])

        self.isTranslate = not self.isTranslate  # Переключаем состояние

        if self.winfo_screenheight() < 900:
            self.create_menu()  # Пересоздаем меню после изменения языка
        if self.tray_icon is not None:
            self.tray_icon.stop()  # останавливаем старую иконку
        self.create_tray_icon()  # создаём новую иконку с обновлённым меню


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