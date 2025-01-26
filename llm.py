import os
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

from webscout import KOBOLDAI, BLACKBOXAI, YouChat, Felo, BlackboxAIImager, Bing, PhindSearch, DeepInfra, Julius, DARKAI, Bagoodex, RUBIKSAI, VLM, DiscordRocks, NexraImager, ChatGPTES, AmigoChat, TurboSeek, Netwrck, Qwenlm, WEBS as w
from webscout import Marcus, AskMyAI
from freeGPT import Client
from datetime import datetime
from tkinter import messagebox, filedialog
from PIL import Image
from io import BytesIO
from packaging import version
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse


# Скрываем сообщения от Pygame
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

CURRENT_VERSION = "1.43"

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
""" # Добавление навыков ИИ и другие тонкие настройки

def download_tesserat():
    try:
        # Получение информации о последнем релизе на GitHub
        response = requests.get("https://api.github.com/repos/Processori7/llm/releases/latest")
        response.raise_for_status()
        latest_release = response.json()

        # Получение ссылки на файл llm.exe последней версии
        download_url = None
        assets = latest_release["assets"]
        for asset in assets:
            if asset["name"] == "tesseract.exe":  # Ищем только tesseract.exe
                download_url = asset["browser_download_url"]
                break
        webbrowser.open(download_url)
    except requests.exceptions.RequestException as e:
        messagebox.showerror("Tesseract error", str(e))

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
            # Предложение пользователю обновление
            if messagebox.showwarning("Доступно обновление",
                                      f"Доступна новая версия {latest_version}. Хотите обновить?", icon='warning',
                                      type='yesno') == 'yes':
                update_app(download_url)
    except requests.exceptions.RequestException as e:
        messagebox.showerror("Error", str(e))

def communicate_with_Qwenlm(user_input, model):
    try:
        ai = Qwenlm(timeout=5000)
        ai.model = model
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

def communicate_with_AskMyAI(user_input):
    try:
        ai = AskMyAI()
        response = ai.chat(user_input)
        return response
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

def communicate_with_Bagoodex(user_input):
    try:
        ai = Bagoodex()
        response = ai.chat(user_input)
        return response.encode('latin1').decode('utf-8')
    except Exception as e:
        return f"{get_error_message(app.isTranslate)}: {str(e)}"

def communicate_with_Amigo(user_input, model):
    try:
        ai = AmigoChat()
        ai.model = model
        response = ai.chat(user_input)
        return response
    except Exception as e:
        return f"{get_error_message(app.isTranslate)}: {str(e)}"

def communicate_with_ChatGPTES(user_input, model):
    try:
        ai = ChatGPTES()
        ai.model = model
        response = ai.chat(user_input)
        return response
    except Exception as e:
        return f"{get_error_message(app.isTranslate)}: {str(e)}"

def communicate_with_Bing(user_input, model):
    try:
        ai = Bing()
        ai.model = model
        response = ai.chat(user_input)
        return response
    except Exception as e:
        return f"{get_error_message(app.isTranslate)}: {str(e)}"

def communicate_with_DiscordRocks(user_input, model):
    try:
        ai = DiscordRocks()
        ai.model = model
        response = ai.chat(user_input)
        return response
    except Exception as e:
        return f"{get_error_message(app.isTranslate)}: {str(e)}"

def communicate_with_RubiksAi(user_input, model):
    try:
        ai = RUBIKSAI()
        ai.model = model
        response = ai.chat(user_input)
        return response
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

def communicate_with_Julius(user_input):
    try:
        ai = Julius()
        ai.model = "GPT-4o"
        response = ai.chat(user_input)
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

def communicate_with_DeepInfra(user_input, model):
    try:
        ai = DeepInfra()
        ai.model=model
        prompt = user_input
        response = ai.ask(prompt)
        # Извлекаем только content из ответа
        content = response['choices'][0]['message']['content']
        return content
    except Exception as e:
        return f"{get_error_message(app.isTranslate)}: {str(e)}"

model_functions = {
"GPT-4o-mini(DDG)": lambda user_input: communicate_with_DuckDuckGO(user_input, "gpt-4o-mini"),
"Bing(Balanced)": lambda user_input: communicate_with_Bing(user_input, "Balanced"),
"Bing(Creative)": lambda user_input: communicate_with_Bing(user_input, "Creative"),
"Bing(Precise)": lambda user_input: communicate_with_Bing(user_input, "Precise"),
"GPT-4o(Julius)": lambda user_input: communicate_with_Julius(user_input),
"gpt-4o(DarkAi)": lambda user_input: communicate_with_DarkAi(user_input, "gpt-4o"),
"gpt-4o-mini(RUBIKSAI)": lambda user_input: communicate_with_RubiksAi(user_input, "gpt-4o-mini"),
"Gpt-4(DiscordRocks)": lambda user_input: communicate_with_DiscordRocks(user_input, "gpt-4"),
"Gpt-4-0613(DiscordRocks)": lambda user_input: communicate_with_DiscordRocks(user_input, "gpt-4-0613"),
"Gpt-4-turbo(DiscordRocks)": lambda user_input: communicate_with_DiscordRocks(user_input, "gpt-4-turbo"),
"Gpt-4o-mini-2024-07-18(DiscordRocks)": lambda user_input: communicate_with_DiscordRocks(user_input, "gpt-4o-mini-2024-07-18"),
"gpt-4o(ChatGPTES)": lambda user_input: communicate_with_ChatGPTES(user_input, "gpt-4o"),
"Gpt-4o(Blackbox)": lambda user_input: communicate_with_BlackboxAI(user_input, "gpt-4o"),
"gpt-4o-mini(ChatGPTES)": lambda user_input: communicate_with_ChatGPTES(user_input, "gpt-4o-mini"),
"chatgpt-4o-latest(ChatGPTES)": lambda user_input: communicate_with_ChatGPTES(user_input, "chatgpt-4o-latest"),
"Gpt-4o1-mini(Amigo)": lambda user_input: communicate_with_Amigo(user_input, "o1-mini"),
"Gpt-4o1-preview(Amigo)": lambda user_input: communicate_with_Amigo(user_input, "o1-preview"),
"Claude-3-haiku(DDG)": lambda user_input: communicate_with_DuckDuckGO(user_input, "claude-3-haiku"),
"Nemotron-4-340B-Instruct(DeepInfra)": lambda user_input: communicate_with_DeepInfra(user_input, "nvidia/Nemotron-4-340B-Instruct"),
"Qwen2-72B-Instruct(DeepInfra)": lambda user_input: communicate_with_DeepInfra(user_input, "Qwen/Qwen2-72B-Instruct"),
"BlackboxAI": lambda user_input: communicate_with_BlackboxAI(user_input, "blackboxai"),
"gpt4mini(Netwrck)": lambda user_input: communicate_with_Netwrck(user_input, "gpt4mini"),
"llama-3.1-lumimaid-8b(Netwrck)": lambda user_input: communicate_with_Netwrck(user_input, "lumimaid"),
"grok-2(Netwrck)": lambda user_input: communicate_with_Netwrck(user_input, "grok"),
"claude-3.5-sonnet(Netwrck)": lambda user_input: communicate_with_Netwrck(user_input, "claude"),
"l3-euryale-70b(Netwrck)": lambda user_input: communicate_with_Netwrck(user_input, "euryale"),
"mythomax-l2-13b(Netwrck)": lambda user_input: communicate_with_Netwrck(user_input, "mythomax"),
"gemini-pro-1.5(Netwrck)": lambda user_input: communicate_with_Netwrck(user_input, "gemini"),
"llama-3.1-lumimaid-70b(Netwrck)": lambda user_input: communicate_with_Netwrck(user_input, "lumimaid70b"),
"llama-3.1-nemotron-70b(Netwrck)": lambda user_input: communicate_with_Netwrck(user_input, "nemotron"),
"openai-o1(YouChat)": lambda user_input: communicate_with_YouChat(user_input, "openai_o1"),
"openai-o1-mini(YouChat)": lambda user_input: communicate_with_YouChat(user_input, "openai_o1_mini"),
"gpt-4o-mini(YouChat)": lambda user_input: communicate_with_YouChat(user_input, "gpt_4o_mini"),
"gpt-4o(YouChat)": lambda user_input: communicate_with_YouChat(user_input, "gpt_4o"),
"gpt-4-turbo(YouChat)": lambda user_input: communicate_with_YouChat(user_input, "gpt_4_turbo"),
"gpt-4(YouChat)": lambda user_input: communicate_with_YouChat(user_input, "gpt_4"),
"claude-3.5-sonnet(YouChat)": lambda user_input: communicate_with_YouChat(user_input, "claude_3_5_sonnet"),
"claude-3-opus(YouChat)": lambda user_input: communicate_with_YouChat(user_input, "claude_3_opus"),
"claude-3-sonnet(YouChat)": lambda user_input: communicate_with_YouChat(user_input, "claude_3_sonnet"),
"claude-3.5-haiku(YouChat)": lambda user_input: communicate_with_YouChat(user_input, "claude_3_5_haiku"),
"claude-3-haiku(YouChat)": lambda user_input: communicate_with_YouChat(user_input, "claude_3_haiku"),
"llama3-3.70b(YouChat)": lambda user_input: communicate_with_YouChat(user_input, "llama3_3_70b"),
"llama3-2.90b(YouChat)": lambda user_input: communicate_with_YouChat(user_input, "llama3_2_90b"),
"llama3-2.11b(YouChat)": lambda user_input: communicate_with_YouChat(user_input, "llama3_2_11b"),
"llama3-1.405b(YouChat)": lambda user_input: communicate_with_YouChat(user_input, "llama3_1_405b"),
"llama3-1.70b(YouChat)": lambda user_input: communicate_with_YouChat(user_input, "llama3_1_70b"),
"llama3(YouChat)": lambda user_input: communicate_with_YouChat(user_input, "llama3"),
"mistral-large-2(YouChat)": lambda user_input: communicate_with_YouChat(user_input, "mistral_large_2"),
"gemini-1.5-flash(YouChat)": lambda user_input: communicate_with_YouChat(user_input, "gemini_1_5_flash"),
"gemini-1.5-pro(YouChat)": lambda user_input: communicate_with_YouChat(user_input, "gemini_1_5_pro"),
"databricks-dbrx-instruct(YouChat)": lambda user_input: communicate_with_YouChat(user_input, "databricks_dbrx_instruct"),
"qwen2.5-72b(YouChat)": lambda user_input: communicate_with_YouChat(user_input, "qwen2p5_72b"),
"qwen2.5-coder-32b(YouChat)": lambda user_input: communicate_with_YouChat(user_input, "qwen2p5_coder_32b"),
"qwen2.5-coder-32b-instruct(Qwenlm)":lambda user_input: communicate_with_Qwenlm(user_input, "qwen2.5-coder-32b-instruct"),
"qwen-plus-latest(Qwenlm)":lambda user_input: communicate_with_Qwenlm(user_input, "qwen-plus-latest"),
"qvq-72b-preview(Qwenlm)":lambda user_input: communicate_with_Qwenlm(user_input, "qvq-72b-preview"),
"qvq-32b-preview(Qwenlm)":lambda user_input: communicate_with_Qwenlm(user_input, "qvq-32b-preview"),
"qwen-vl-max-latest(Qwenlm)":lambda user_input: communicate_with_Qwenlm(user_input, "qwen-vl-max-latest"),
"command-r(YouChat)": lambda user_input: communicate_with_YouChat(user_input, "command_r"),
"command-r-plus(YouChat)": lambda user_input: communicate_with_YouChat(user_input, "command_r_plus"),
"solar-1-mini(YouChat)": lambda user_input: communicate_with_YouChat(user_input, "solar_1_mini"),
"dolphin-2.5(YouChat)": lambda user_input: communicate_with_YouChat(user_input, "dolphin_2_5"),
"KoboldAI": communicate_with_KoboldAI,
"Phind": communicate_with_Phind,
"Felo": communicate_with_Felo,
"Bagoodex":communicate_with_Bagoodex,
"TurboSeek":communicate_with_TurboSeek,
"Marcus":communicate_with_Marcus,
"AskMyAI":communicate_with_AskMyAI,
"Chatgpt-4o-latest(DiscordRocks)": lambda user_input: communicate_with_DiscordRocks(user_input, "chatgpt-4o-latest"),
"Claude-3-haiku-20240307(DiscordRocks)": lambda user_input: communicate_with_DiscordRocks(user_input, "claude-3-haiku-20240307"),
"Claude-3-sonnet-20240229(DiscordRocks)": lambda user_input: communicate_with_DiscordRocks(user_input, "claude-3-sonnet-20240229"),
"Claude-3-sonnet(Amigo)": lambda user_input: communicate_with_Amigo(user_input, "claude-3-sonnet-20240229"),
"Claude-3-5-sonnet-20240620(DiscordRocks)": lambda user_input: communicate_with_DiscordRocks(user_input, "claude-3-5-sonnet-20240620"),
"Claude-sonnet-3.5(BlackboxAI)": lambda user_input: communicate_with_BlackboxAI(user_input, "claude-sonnet-3.5"),
"Claude-3-opus-20240229(DiscordRocks)": lambda user_input: communicate_with_DiscordRocks(user_input, "claude-3-opus-20240229"),
"Gpt-3.5-turbo(DiscordRocks)": lambda user_input: communicate_with_DiscordRocks(user_input, "gpt-3.5-turbo"),
"Gpt-3.5-turbo-0125(DiscordRocks)": lambda user_input: communicate_with_DiscordRocks(user_input, "gpt-3.5-turbo-0125"),
"Gpt-3.5-turbo-1106(DiscordRocks)": lambda user_input: communicate_with_DiscordRocks(user_input, "gpt-3.5-turbo-1106"),
"Gpt-3.5-turbo-16k(DiscordRocks)": lambda user_input: communicate_with_DiscordRocks(user_input, "gpt-3.5-turbo-16k"),
"Gpt-3.5-turbo-0613(DiscordRocks)": lambda user_input: communicate_with_DiscordRocks(user_input, "gpt-3.5-turbo-0613"),
"Gpt-3.5-turbo-16k-0613(DiscordRocks)": lambda user_input: communicate_with_DiscordRocks(user_input, "gpt-3.5-turbo-16k-0613"),
"Llama-3-70b-chat(DiscordRocks)": lambda user_input: communicate_with_DiscordRocks(user_input, "llama-3-70b-chat"),
"Llama-3-70b-chat-turbo(DiscordRocks)": lambda user_input: communicate_with_DiscordRocks(user_input, "llama-3-70b-chat-turbo"),
"Llama-3-8b-chat(DiscordRocks)": lambda user_input: communicate_with_DiscordRocks(user_input, "llama-3-8b-chat"),
"Llama-3-8b-chat-turbo(DiscordRocks)": lambda user_input: communicate_with_DiscordRocks(user_input, "llama-3-8b-chat-turbo"),
"Llama-3-70b-chat-lite(DiscordRocks)": lambda user_input: communicate_with_DiscordRocks(user_input, "llama-3-70b-chat-lite"),
"Llama-3-8b-chat-lite(DiscordRocks)": lambda user_input: communicate_with_DiscordRocks(user_input, "llama-3-8b-chat-lite"),
"Llama-2-13b-chat(DiscordRocks)": lambda user_input: communicate_with_DiscordRocks(user_input, "llama-2-13b-chat"),
"Llama-3.1-405b-turbo(DiscordRocks)": lambda user_input: communicate_with_DiscordRocks(user_input, "llama-3.1-405b-turbo"),
"Llama-3.1-405B(Amigo)": lambda user_input: communicate_with_Amigo(user_input, "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo"),
"Llama-3.2-90B(Amigo)": lambda user_input: communicate_with_Amigo(user_input, "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo"),
"Llama-3.1-70b-turbo(DiscordRocks)": lambda user_input: communicate_with_DiscordRocks(user_input, "llama-3.1-70b-turbo"),
"Llama-3.1-8b-turbo(DiscordRocks)": lambda user_input: communicate_with_DiscordRocks(user_input, "llama-3.1-8b-turbo"),
"LlamaGuard-2-8b(DiscordRocks)": lambda user_input: communicate_with_DiscordRocks(user_input, "LlamaGuard-2-8b"),
"Llama-Guard-7b(DiscordRocks)": lambda user_input: communicate_with_DiscordRocks(user_input, "Llama-Guard-7b"),
"Meta-Llama-Guard-3-8B(DiscordRocks)": lambda user_input: communicate_with_DiscordRocks(user_input, "Meta-Llama-Guard-3-8B"),
"Mixtral-8x7B-v0.1(DiscordRocks)": lambda user_input: communicate_with_DiscordRocks(user_input, "Mixtral-8x7B-v0.1"),
"Mixtral-8x7B-Instruct-v0.1(DiscordRocks)": lambda user_input: communicate_with_DiscordRocks(user_input, "Mixtral-8x7B-Instruct-v0.1"),
"Mixtral-8x22B-Instruct-v0.1(DiscordRocks)": lambda user_input: communicate_with_DiscordRocks(user_input, "Mixtral-8x22B-Instruct-v0.1"),
"Mistral-7B-Instruct-v0.1(DiscordRocks)": lambda user_input: communicate_with_DiscordRocks(user_input, "Mistral-7B-Instruct-v0.1"),
"Mistral-7B-Instruct-v0.2(DiscordRocks)": lambda user_input: communicate_with_DiscordRocks(user_input, "Mistral-7B-Instruct-v0.2"),
"Mistral-7B-Instruct-v0.3(DiscordRocks)": lambda user_input: communicate_with_DiscordRocks(user_input, "Mistral-7B-Instruct-v0.3"),
"Qwen1.5-72B-Chat(DiscordRocks)": lambda user_input: communicate_with_DiscordRocks(user_input, "Qwen1.5-72B-Chat"),
"Qwen1.5-110B-Chat(DiscordRocks)": lambda user_input: communicate_with_DiscordRocks(user_input, "Qwen1.5-110B-Chat"),
"Qwen2-72B-Instruct(DiscordRocks)": lambda user_input: communicate_with_DiscordRocks(user_input, "Qwen2-72B-Instruct"),
"Gemma-2b-it(DiscordRocks)": lambda user_input: communicate_with_DiscordRocks(user_input, "gemma-2b-it"),
"Dbrx-instruct(DiscordRocks)": lambda user_input: communicate_with_DiscordRocks(user_input, "dbrx-instruct"),
"Deepseek-coder-33b-instruct(DiscordRocks)": lambda user_input: communicate_with_DiscordRocks(user_input, "deepseek-coder-33b-instruct"),
"Deepseek-llm-67b-chat(DiscordRocks)": lambda user_input: communicate_with_DiscordRocks(user_input, "deepseek-llm-67b-chat"),
"Nous-Hermes-2-Mixtral-8x7B-DPO(DiscordRocks)": lambda user_input: communicate_with_DiscordRocks(user_input, "Nous-Hermes-2-Mixtral-8x7B-DPO"),
"Nous-Hermes-2-Yi-34B(DiscordRocks)": lambda user_input: communicate_with_DiscordRocks(user_input, "Nous-Hermes-2-Yi-34B"),
"WizardLM-2-8x22B(DiscordRocks)": lambda user_input: communicate_with_DiscordRocks(user_input, "WizardLM-2-8x22B"),
"CodeLlama-7b-Python(DiscordRocks)": lambda user_input: communicate_with_DiscordRocks(user_input, "CodeLlama-7b-Python"),
"Snowflake-arctic-instruct(DiscordRocks)": lambda user_input: communicate_with_DiscordRocks(user_input, "snowflake-arctic-instruct"),
"Solar-10.7B-Instruct-v1.0(DiscordRocks)": lambda user_input: communicate_with_DiscordRocks(user_input, "SOLAR-10.7B-Instruct-v1.0"),
"Stripedhyena-nous-7B(DiscordRocks)": lambda user_input: communicate_with_DiscordRocks(user_input, "StripedHyena-Nous-7B"),
"CodeLlama-13b-Instruct(DiscordRocks)": lambda user_input: communicate_with_DiscordRocks(user_input, "CodeLlama-13b-Instruct"),
"Mythomax-L2-13b(DiscordRocks)": lambda user_input: communicate_with_DiscordRocks(user_input, "MythoMax-L2-13b"),
"Gemma-2-9b-it(DiscordRocks)": lambda user_input: communicate_with_DiscordRocks(user_input, "gemma-2-9b-it"),
"Gemma-2-27b-it(DiscordRocks)": lambda user_input: communicate_with_DiscordRocks(user_input, "gemma-2-27b-it"),
"Gemini-1.5-flash(DiscordRocks)": lambda user_input: communicate_with_DiscordRocks(user_input, "gemini-1.5-flash"),
"Gemini-1.5-pro(DiscordRocks)": lambda user_input: communicate_with_DiscordRocks(user_input, "gemini-1.5-pro"),
"Gemini-1.5-pro(Amigo)": lambda user_input: communicate_with_Amigo(user_input, "gemini-1.5-pro"),
"Gemini-1-5-flash(Amigo)": lambda user_input: communicate_with_Amigo(user_input, "gemini-1-5-flash"),
"Sparkdesk(DiscordRocks)": lambda user_input: communicate_with_DiscordRocks(user_input, "sparkdesk"),
"Cosmosrp(DiscordRocks)": lambda user_input: communicate_with_DiscordRocks(user_input, "cosmosrp"),
"Reflection-70B (DeepInfra)": lambda user_input: communicate_with_DeepInfra(user_input, "mattshumer/Reflection-Llama-3.1-70B"),
"Llama 3-70B (DDG)": lambda user_input: communicate_with_DuckDuckGO(user_input, "llama-3-70b"),
"Llama-3.1-70B (DeepInfra)": lambda user_input: communicate_with_DeepInfra(user_input, "meta-llama/Meta-Llama-3.1-70B-Instruct"),
"Llama-3.1-70B-Instruct-Turbo(DeepInfra)": lambda user_input: communicate_with_DeepInfra(user_input, "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"),
"Llama-3.1-Nemotron-70B-Instruct(DeepInfra)": lambda user_input: communicate_with_DeepInfra(user_input, "nvidia/Llama-3.1-Nemotron-70B-Instruct"),
"Llama-3.2-90B-Vision-Instruct(DeepInfra)": lambda user_input: communicate_with_DeepInfra(user_input, "meta-llama/Llama-3.2-90B-Vision-Instruct"),
"Llama-3.1-405B(DarkAi)": lambda user_input: communicate_with_DarkAi(user_input, "llama-3-405b"),
"Meta-Llama-3.1-405B(DeepInfra)": lambda user_input: communicate_with_DeepInfra(user_input, "meta-llama/Meta-Llama-3.1-405B-Instruct"),
"Mixtral-8x7b(DDG)": lambda user_input: communicate_with_DuckDuckGO(user_input,"mixtral-8x7b"),
"Mixtral-8x22B(DeepInfra)": lambda user_input: communicate_with_DeepInfra(user_input,"mistralai/Mixtral-8x22B-Instruct-v0.1"),
"WizardLM-2-8x22B(DeepInfra)": lambda user_input: communicate_with_DeepInfra(user_input,"microsoft/WizardLM-2-8x22B"),
"Mixtral-8x7B(DeepInfra)": lambda user_input: communicate_with_DeepInfra(user_input,"mistralai/Mixtral-8x7B-Instruct-v0.1"),
"Dolphin-2.6-mixtral-8x7b(DeepInfra)": lambda user_input: communicate_with_DeepInfra(user_input,"cognitivecomputations/dolphin-2.6-mixtral-8x7b"),
"Dolphin-2.9.1-llama-3-70b(DeepInfra)": lambda user_input: communicate_with_DeepInfra(user_input,"cognitivecomputations/dolphin-2.9.1-llama-3-70b"),
"L3-70B-Euryale-v2.1(DeepInfra)": lambda user_input: communicate_with_DeepInfra(user_input,"Sao10K/L3-70B-Euryale-v2.1"),
"Phi-3-medium-4k-instruct(DeepInfra)": lambda user_input: communicate_with_DeepInfra(user_input,"microsoft/Phi-3-medium-4k-instruct"),
"MiniCPM-Llama3-V-2_5(Photo Analyze)(VLM)":lambda user_input: communicate_with_VLM(user_input, "openbmb/MiniCPM-Llama3-V-2_5"),
"Llava-1.5-7b-hf(Photo Analyze)(VLM)":lambda user_input: communicate_with_VLM(user_input, "llava-hf/llava-1.5-7b-hf"),
"Emi_img(NexraImager)":lambda user_input: communicate_with_NexraImager(user_input, "emi"),
"Stablediffusion-1.5_img(NexraImager)":lambda user_input: communicate_with_NexraImager(user_input, "stablediffusion-1.5"),
"Stablediffusion-2.1_img(NexraImager)":lambda user_input: communicate_with_NexraImager(user_input, "stablediffusion-2.1"),
"Sdxl-lora_img(NexraImager)":lambda user_input: communicate_with_NexraImager(user_input, "sdxl-lora"),
"Dalle_img(NexraImager)":lambda user_input: communicate_with_NexraImager(user_input, "dalle"),
"Dalle2_img(NexraImager)":lambda user_input: communicate_with_NexraImager(user_input, "dalle2"),
"Dalle-mini_img(NexraImager)":lambda user_input: communicate_with_NexraImager(user_input, "dalle-mini"),
"DreamshaperXL10_alpha2_img(NexraImager)":lambda user_input: communicate_with_NexraImager(user_input, "dreamshaperXL10_alpha2.safetensors [c8afe2ef]"),
"DynavisionXL_0411_img(NexraImager)":lambda user_input: communicate_with_NexraImager(user_input, "dynavisionXL_0411.safetensors [c39cc051]"),
"JuggernautXL_v45_img(NexraImager)":lambda user_input: communicate_with_NexraImager(user_input, "juggernautXL_v45.safetensors [e75f5471]"),
"RealismEngineSDXL_v10_img(NexraImager)":lambda user_input: communicate_with_NexraImager(user_input, "realismEngineSDXL_v10.safetensors [af771c3f]"),
"Sd_xl_base_1.0_img(NexraImager)":lambda user_input: communicate_with_NexraImager(user_input, "sd_xl_base_1.0.safetensors [be9edd61]"),
"AnimagineXLV3_v30_img(NexraImager)":lambda user_input: communicate_with_NexraImager(user_input, "animagineXLV3_v30.safetensors [75f2f05b]"),
"Sd_xl_base_1.0_inpainting_0.1_img(NexraImager)":lambda user_input: communicate_with_NexraImager(user_input, "sd_xl_base_1.0_inpainting_0.1.safetensors [5679a81a]"),
"TurbovisionXL_v431_img(NexraImager)":lambda user_input: communicate_with_NexraImager(user_input, "turbovisionXL_v431.safetensors [78890989]"),
"Devlishphotorealism_sdxl15_img(NexraImager)":lambda user_input: communicate_with_NexraImager(user_input, "devlishphotorealism_sdxl15.safetensors [77cba69f]"),
"RealvisxlV40_img(NexraImager)":lambda user_input: communicate_with_NexraImager(user_input, "realvisxlV40.safetensors [f7fdcb51]"),
"BlackboxAIImager_img":lambda user_input: communicate_with_BlackboxAIImager(user_input),
"Prodia_img":lambda user_input: gen_img(user_input, "prodia"),
"Pollinations_img":lambda user_input: gen_img(user_input, "pollinations")
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
        resp = Client.create_generation(model, user_input)

        img_folder = 'img'
        if not os.path.exists(img_folder):
            os.makedirs(img_folder)
        now = datetime.now()
        image_path = os.path.join(img_folder, f'{user_input}_{now.strftime('%d.%m.%Y_%H.%M.%S')}.png')
        with Image.open(BytesIO(resp)) as img:
            img.save(image_path)

        return f"{get_save_img_messages(app.isTranslate)}{image_path}"

    except Exception as e:
       return f"{get_error_message(app.isTranslate)}: {str(e)}"

def communicate_with_BlackboxAIImager(user_input):
    try:
        bot = BlackboxAIImager()
        resp = bot.generate(user_input, 1)
        img_folder = 'img'
        if not os.path.exists(img_folder):
            os.makedirs(img_folder)
        now = datetime.now()
        image_path = os.path.join(img_folder, f'{user_input}_{now.strftime('%d.%m.%Y_%H.%M.%S')}.png')
        # Объединяем список байтов в один объект байтов
        combined_bytes = b''.join(resp)

        with Image.open(BytesIO(combined_bytes)) as img:
            img.save(image_path)

        return f"{get_save_img_messages(app.isTranslate)}{image_path}"

    except Exception as e:
       return f"{get_error_message(app.isTranslate)}: {str(e)}"

def communicate_with_NexraImager(user_input, model):
    try:
        ai = NexraImager()
        ai.model = model
        resp = ai.generate(user_input, model)

        # Проверяем, что resp - это список изображений
        if isinstance(resp, list) and len(resp) > 0:
            num_images = len(resp)  # Количество изображений
        else:
            raise ValueError(get_error_gen_img_messages(app.isTranslate))

        img_folder = 'img'
        if not os.path.exists(img_folder):
            os.makedirs(img_folder)

        now = datetime.now()
        saved_images = []  # Список для хранения путей сохраненных изображений

        for i, image_data in enumerate(resp):
            image_path = os.path.join(img_folder, f'{user_input}_{now.strftime("%d.%m.%Y_%H.%M.%S")}_{i + 1}.png')
            with Image.open(BytesIO(image_data)) as img:
                img.save(image_path)
                saved_images.append(image_path)  # Добавляем путь к сохраненному изображению в список

        return f"{get_save_img_messages(app.isTranslate)}{', '.join(saved_images)}"
    except Exception as e:
        return f"{get_error_message(app.isTranslate)}: {str(e)}"

class ChatApp(ctk.CTk):
    def __init__(self):
        try:
            super().__init__()
            ctk.set_appearance_mode("dark")
            ctk.set_default_color_theme("green")

            pytesseract.tesseract_cmd = 'tesseract.exe'
            self.tesseract_cmd = 'tesseract.exe'
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

            self.image = Image.open("icon.ico")
            self.iconbitmap("icon.ico")
            self.menu = (
                pystray.MenuItem("Открыть", self.show_window, default=True),
                pystray.MenuItem("Закрыть", self.on_exit)
            )
            self.icon = pystray.Icon("name", self.image, "Free Ai Services", self.menu)

            # Привязываем событие закрытия окна
            self.protocol("WM_DELETE_WINDOW", self.hide_window)

            # Создание виджета chat_history с прокруткой
            self.chat_history_frame = ctk.CTkFrame(self)
            self.chat_history_frame.pack(fill="both", expand=True, padx=10, pady=10)

            self.chat_history = ctk.CTkTextbox(self.chat_history_frame, font=("Consolas", 18))
            self.chat_history.pack(fill="both", expand=True)
            self.chat_history.configure(state="disabled")

            # Добавление контекстного меню для chat_history
            self.chat_history.bind("<Button-3>", self.show_context_menu)
            self.chat_history_context_menu = tk.Menu(self, tearoff=0)
            self.chat_history_context_menu.add_command(label="Выделить всё", command=self.select_all)
            self.chat_history_context_menu.add_command(label="Копировать", command=self.copy_text)

            self.input_frame = ctk.CTkFrame(self)
            self.input_frame.pack(fill="x", padx=10, pady=10)

            # Метка для выбора модели
            self.model_label = ctk.CTkLabel(self.input_frame, text="Выберите модель:", font=("Consolas", 18))
            self.model_label.pack(side="left", padx=5)

            # Комбобокс для выбора модели
            self.model_var = tk.StringVar()
            self.model_combobox = ctk.CTkOptionMenu(self.input_frame, variable=self.model_var, font=("Consolas", 16),
                                                    values=list(model_functions.keys()))
            self.model_combobox.pack(side="left", padx=5)
            self.model_combobox.set(list(model_functions.keys())[0])  # Модель по умолчанию

            # Создаем новый фрейм для выбора категории
            self.category_frame = ctk.CTkFrame(self.input_frame)
            self.category_frame.pack(side="top", padx=6)  # Устанавливаем фрейм ниже

            # Метка для категории
            self.category_label = ctk.CTkLabel(self.category_frame, text="Выберите категорию:", font=("Consolas", 18))
            self.category_label.pack(side="left", padx=6)

            # Комбобокс для выбора категории моделей
            self.category_var = tk.StringVar()
            self.category_combobox = ctk.CTkOptionMenu(self.category_frame, variable=self.category_var,
                                                       font=("Consolas", 16),
                                                       values=["All", "Text", "Img", "Photo Analyze"],
                                                       command=self.update_model_list)
            self.category_combobox.pack(side="left", padx=6)

            # Установка "All" как модели по умолчанию
            self.category_combobox.set("All")

            self.search_label = ctk.CTkLabel(self.category_frame, text="Поиск модели:", font=("Consolas", 18))
            self.search_label.pack(side="left", padx=6)

            self.search_var = tk.StringVar()
            self.search_entry = ctk.CTkEntry(self.category_frame, textvariable=self.search_var, font=("Consolas", 18))
            self.search_entry.pack(side="left", padx=6)

            # Установка trace для отслеживания изменений в строке поиска
            self.search_var.trace("w", self.filter_models)

            # Обновление списка моделей при инициализации
            self.update_model_list("All")

            self.input_entry = ctk.CTkTextbox(self.input_frame, font=("Consolas", 16), height=200, width=180, wrap="word", text_color="orange")
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

            # Создаем фрейм для кнопок
            self.button_frame = ctk.CTkFrame(self.input_frame)
            self.button_frame.pack(side="top", fill="x")  # Упаковываем фрейм сверху и растягиваем по ширине

            # Кнопки
            self.send_button = ctk.CTkButton(self.button_frame, text="Отправить", command=self.send_message,
                                             font=("Consolas", 14), text_color="white")
            self.send_button.pack(side="top", padx=5, pady=10)

            self.clear_button = ctk.CTkButton(self.button_frame, text="Очистить чат", command=self.clear_chat,
                                              font=("Consolas", 14), text_color="white")
            self.clear_button.pack(side="top", padx=5, pady=10)

            # Кнопка голосового ввода
            self.speech_reco_button = ctk.CTkButton(self.button_frame, text="Голосовой ввод",
                                                    command=self.toggle_recognition,
                                                    font=("Consolas", 14), text_color="white")
            self.speech_reco_button.pack(side="top", padx=5, pady=10)

            # Кнопка распознавания текста с картинки
            self.img_reco_button = ctk.CTkButton(self.button_frame, text="Распознать текст", command=self.recognize_text,
                                                 font=("Consolas", 14), text_color="white")
            self.img_reco_button.pack(side="top", padx=5, pady=10)

            # Кнопка переключения темы
            self.theme_button = ctk.CTkButton(self.button_frame, text="Светлая тема", command=self.toggle_theme,
                                              font=("Consolas", 14), text_color="white")
            self.theme_button.pack(side="top", padx=5, pady=10)

            # Кнопка переключения языка
            self.lang_button = ctk.CTkButton(self.button_frame, text="English", command=self.toggle_lang,
                                             font=("Consolas", 14), text_color="white")
            self.lang_button.pack(side="top", padx=5, pady=10)

            # Кнопка API Mode
            self.api_mode_button = ctk.CTkButton(self.button_frame, text="API Mode", command=self.toggle_api_mode,
                                                 font=("Consolas", 14), text_color="white")

            self.api_mode_button.pack(side="top", padx=5, pady=10)

            # Кнопка закрытия программы
            self.exit_button = ctk.CTkButton(self.button_frame, text="Выход", command=self.on_exit,
                                             font=("Consolas", 14), text_color="white")
            self.exit_button.pack(side="top", padx=5, pady=10)

            # Определение тегов для цветного текста
            self.chat_history.tag_add("user_input", "1.0")
            self.chat_history.tag_add("response", "1.0")
            self.chat_history.tag_add("system_line", "1.0")

            # Определение тегов для цветного текста
            self.chat_history.tag_config("user_input", foreground="orange")
            self.chat_history.tag_config("response", foreground="yellow")
            self.chat_history.tag_config("system_line", foreground="cyan")

            # Чекбокс "Вести историю"
            self.history_var = tk.BooleanVar()
            self.history_checkbox = ctk.CTkCheckBox(self.input_frame, text="Вести историю", variable=self.history_var)
            self.history_checkbox.pack(side="top", padx=5, pady=5)

            # Переменная для отслеживания активного виджета
            self.active_widget = None

            # Привязываем события фокуса к виджетам
            self.chat_history.bind("<FocusIn>", self.set_active_widget)
            self.input_entry.bind("<FocusIn>", self.set_active_widget)

        except Exception as e:
            messagebox.showerror("Возникла ошибка", e)

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
            app.mount("/static", StaticFiles(directory="static"), name="static")

            # Главная страница чата
            @app.get("/chat")
            def read_chat():
                with open("static/chat.html", "r", encoding="utf-8") as file:
                    return HTMLResponse(content=file.read())

            config = uvicorn.Config(app, host=self.local_ip, port=8000, log_level="info")
            self.uvicorn_server = uvicorn.Server(config=config)
            self.uvicorn_server.run()

        self.server_process = threading.Thread(target=run_fastapi_app)
        self.server_process.start()
        server_url = f"http://{self.local_ip}:8000/chat"
        webbrowser.open(server_url)
        self.api_running = True
        self.api_mode_button.configure(text="Stop API Mode")

    def stop_api_mode(self):
        if self.uvicorn_server is not None:
            self.uvicorn_server.should_exit = True
            self.uvicorn_server.force_exit = True
            self.uvicorn_server.shutdown()
            self.uvicorn_server = None
        self.api_running = False
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
                except Exception as e:
                    self.input_entry.delete("1.0", tk.END)
                    self.input_entry.insert("1.0", "An error has occurred.")


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
            if os.path.exists(self.tesseract_cmd):

                image_path = filedialog.askopenfilename(title="Выберите изображение",
                                                        filetypes=(("Изображения", "*.jpg;*.png;*.gif"),
                                                                   ("Все файлы", "*.*")))
                if image_path:
                    # Загрузка изображения
                    image = cv2.imread(image_path)

                    # Проверка, было ли изображение загружено успешно
                    if image is None:
                        messagebox.showerror(get_image_title_errors(app.isTranslate),
                                             get_image_load_error_message(app.isTranslate))

                    # Преобразование изображения в оттенки серого
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                    # Применение порогового значения для выделения текста
                    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

                    # Использование pytesseract для распознавания текста
                    recognized_text = pytesseract.image_to_string(thresh, lang='rus+eng')

                    if recognized_text:
                        self.input_entry.delete("1.0", tk.END)
                        self.input_entry.insert("1.0", recognized_text)
                    else:
                        messagebox.showinfo("Result", get_no_text_recognized_message(app.isTranslate))
                else:
                    messagebox.showerror(get_image_title_errors(app.isTranslate),
                                         get_select_image_message_errors(app.isTranslate))
            else:
                ask = messagebox.askquestion("tesseract.exe", get_tesseract_not_found_messages(app.isTranslate))
                if ask ==True:
                    download_tesserat()
        except Exception as e:
            # Получаем информацию об ошибке
            error_message = f"{get_text_recognition_error_message(app.isTranslate)}\n{str(e)}\n\n"
            error_message +=traceback.format_exc()
            # Показываем сообщение об ошибке
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

        # Обновление комбобокса с моделями
        self.model_combobox.configure(values=filtered_models)
        self.model_combobox.set(filtered_models[0] if filtered_models else "")

    def set_active_widget(self, event):
        # Устанавливаем активный виджет
        self.active_widget = event.widget

    def create_tray_icon(self):
        """Создает новый экземпляр иконки трея"""
        menu = (
            pystray.MenuItem("Открыть", self.show_window, default=True),
            pystray.MenuItem("API Mode", self.toggle_api_mode()),
            pystray.MenuItem("Закрыть", self.on_exit)
        )
        image = Image.open("icon.ico")
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
        """Восстановление окна из трея"""
        if self.tray_icon:
            self.tray_icon.stop()  # Останавливаем текущую иконку
        self.deiconify()
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
            messagebox.showerror("Возникла ошибка", e)

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
            with open("llm_history.txt", "a") as f:
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
        else:
            # Переключаем на английский
            self.model_label.configure(text="Select model:")
            self.category_label.configure(text="Select category:")
            self.send_button.configure(text="Send")
            self.clear_button.configure(text="Clear chat")
            self.theme_button.configure(text="Light theme")
            self.lang_button.configure(text="Русский")
            self.history_checkbox.configure(text="Keep history")
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

        self.isTranslate = not self.isTranslate  # Переключаем состояние


if __name__ == "__main__":
    check_for_updates()
    app = ChatApp()
    app.mainloop()
