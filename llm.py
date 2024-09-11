import os
# Скрываем сообщения от Pygame
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import re
import requests
import webbrowser
import datetime
import tkinter.font as tkFont
import customtkinter as ctk
import tkinter as tk
import pystray
import ctypes
from webscout import KOBOLDAI, BLACKBOXAI, ThinkAnyAI, PhindSearch, DeepInfra, Julius, DARKAI, RUBIKSAI, VLM, DeepInfraImager, WEBS as w
from freeGPT import Client
from datetime import datetime
from tkinter import messagebox, filedialog
from PIL import Image
from io import BytesIO
from packaging import version


CURRENT_VERSION = "1.27"

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

def update_app(update_url):
   webbrowser.open(update_url)

def check_for_updates():
    try:
        # Получение информации о последнем релизе на GitHub
        response = requests.get("https://api.github.com/repos/Processori7/llm/releases/latest")
        response.raise_for_status()
        latest_release = response.json()

        # Получение ссылки на файл exe последней версии
        assets = latest_release["assets"]
        for asset in assets:
            if asset["name"].endswith(".exe"):
                download_url = asset["browser_download_url"]
                break
        else:
            messagebox.showerror("Ошибка обновления", "Не удалось найти файл exe для последней версии.")
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
            messagebox.showerror("Ошибка при проверке обновлений", e)

def communicate_with_RUBIKSAI(user_input, model):
    ai = RUBIKSAI()
    ai.model = model
    response = ai.chat(user_input)
    return response

def communicate_with_DarkAi(user_input, model):
    ai = DARKAI()
    ai.model = model
    response = ai.chat(user_input)
    return response

def communicate_with_DuckDuckGO(user_input, model):
    response = w().chat(user_input, model=model)  # GPT-4.o mini, mixtral-8x7b, llama-3-70b, claude-3-haiku
    return response

def communicate_with_Julius(user_input):
    ai = Julius()
    ai.model = "GPT-4o"
    response = ai.chat(user_input)
    return response

def communicate_with_KoboldAI(user_input):
    try:
        koboldai = KOBOLDAI()
        response = koboldai.chat(user_input)
        return response
    except Exception as e:
        return f"Ошибка при общении с KoboldAI: {e}"

def communicate_with_BlackboxAI(user_input):
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
            model=None
        )
        response = ai.chat(user_input)
        return response
    except Exception as e:
        return f"Ошибка при общении с BLACKBOXAI: {e}"

def communicate_with_ThinkAnyAI(user_input, model):
    try:
        opengpt = ThinkAnyAI(locale="ru", model=model, max_tokens=1500)
        response = opengpt.chat(user_input)
        return response
    except Exception as e:
        return f"Ошибка при общении с ThinkAnyAI: {e}"

def communicate_with_Phind(user_input):
    try:
        ph = PhindSearch()
        response = ph.chat(user_input)
        return response
    except Exception as e:
        return f"Ошибка при общении с PhindAI: {e}"

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
        return f"Ошибка при общении с DeepInfraAI: {e}"

model_functions = {
                "GPT-4o-mini(DDG)": lambda user_input: communicate_with_DuckDuckGO(user_input, "gpt-4o-mini"),
                "GPT-4o-mini": lambda user_input: communicate_with_ThinkAnyAI(user_input, "gpt-4o-mini"),
                "GPT-4o": lambda user_input: communicate_with_Julius(user_input),
                "gpt-4o(DarkAi)": lambda user_input: communicate_with_DarkAi(user_input, "gpt-4o"),
                "gpt-4o-mini(RUBIKSAI)": lambda user_input: communicate_with_RUBIKSAI(user_input, "gpt-4o-mini"),
                "KoboldAI": communicate_with_KoboldAI,
                "BlackboxAI": communicate_with_BlackboxAI,
                "Claude-3-haiku(ThinkAny)": lambda user_input: communicate_with_ThinkAnyAI(user_input, "claude-3-haiku"),
                "Claude-3-haiku(DDG)": lambda user_input: communicate_with_DuckDuckGO(user_input, "claude-3-haiku"),
                "Nemotron-4-340B-Instruct": lambda user_input: communicate_with_DeepInfra(user_input, "nvidia/Nemotron-4-340B-Instruct"),
                "Qwen2-72B-Instruct": lambda user_input: communicate_with_DeepInfra(user_input, "Qwen/Qwen2-72B-Instruct"),
                "Phind": communicate_with_Phind,
                "Reflection-70B (DeepInfra)": lambda user_input: communicate_with_DeepInfra(user_input, "mattshumer/Reflection-Llama-3.1-70B"),
                "Llama 3-70B (DDG)": lambda user_input: communicate_with_DuckDuckGO(user_input, "llama-3-70b"),
                "Llama-3.1-8B-instruct": lambda user_input: communicate_with_ThinkAnyAI(user_input,"llama-3.1-8b-instruct"),
                "Llama-3.1-70B (DeepInfra)": lambda user_input: communicate_with_DeepInfra(user_input, "meta-llama/Meta-Llama-3.1-70B-Instruct"),
                "Llama-3.1-405B(DarkAi)": lambda user_input: communicate_with_DarkAi(user_input, "llama-3-405b"),
                "Meta-Llama-3.1-405B": lambda user_input: communicate_with_DeepInfra(user_input, "meta-llama/Meta-Llama-3.1-405B-Instruct"),
                "Gemini-pro": lambda user_input: communicate_with_ThinkAnyAI(user_input, "gemini-pro"),
                "Gemma-2-27b-it": lambda user_input: communicate_with_ThinkAnyAI(user_input, "google/gemma-2-27b-it"),
                "Mistral-7b-instruct": lambda user_input: communicate_with_ThinkAnyAI(user_input,"mistral-7b-instruct"),
                "Mixtral-8x7b": lambda user_input: communicate_with_DuckDuckGO(user_input,"mixtral-8x7b"),
                "Mixtral-8x22B": lambda user_input: communicate_with_DeepInfra(user_input,"mistralai/Mixtral-8x22B-Instruct-v0.1"),
                "WizardLM-2-8x22B": lambda user_input: communicate_with_DeepInfra(user_input,"microsoft/WizardLM-2-8x22B"),
                "Mixtral-8x7B": lambda user_input: communicate_with_DeepInfra(user_input,"mistralai/Mixtral-8x7B-Instruct-v0.1"),
                "Dolphin-2.6-mixtral-8x7b": lambda user_input: communicate_with_DeepInfra(user_input,"cognitivecomputations/dolphin-2.6-mixtral-8x7b"),
                "Dolphin-2.9.1-llama-3-70b": lambda user_input: communicate_with_DeepInfra(user_input,"cognitivecomputations/dolphin-2.9.1-llama-3-70b"),
                "L3-70B-Euryale-v2.1": lambda user_input: communicate_with_DeepInfra(user_input,"Sao10K/L3-70B-Euryale-v2.1"),
                "Phi-3-medium-4k-instruct": lambda user_input: communicate_with_DeepInfra(user_input,"microsoft/Phi-3-medium-4k-instruct"),
                "MiniCPM-Llama3-V-2_5(Photo Analyze)":lambda user_input: communicate_with_VLM(user_input, "openbmb/MiniCPM-Llama3-V-2_5"),
                "FLUX-1-dev_img":lambda user_input: communicate_with_DeepInfraImager(user_input, "black-forest-labs/FLUX-1-dev"),
                "FLUX-1-schnell_img":lambda user_input: communicate_with_DeepInfraImager(user_input, "black-forest-labs/FLUX-1-schnell"),
                "Stable-diffusion-2-1_img":lambda user_input: communicate_with_DeepInfraImager(user_input, "stabilityai/stable-diffusion-2-1"),
                "Stable-diffusion-v1-5_img":lambda user_input: communicate_with_DeepInfraImager(user_input, "runwayml/stable-diffusion-v1-5"),
                "Stable-diffusion-v1-4_img":lambda user_input: communicate_with_DeepInfraImager(user_input, "CompVis/stable-diffusion-v1-4"),
                "Deliberate_img":lambda user_input: communicate_with_DeepInfraImager(user_input, "XpucT/Deliberate"),
                "Openjourney_img":lambda user_input: communicate_with_DeepInfraImager(user_input, "prompthero/openjourney"),
                "Sdxl_img":lambda user_input: communicate_with_DeepInfraImager(user_input, "stability-ai/sdxl"),
                "Custom-diffusion_img":lambda user_input: communicate_with_DeepInfraImager(user_input, "uwulewd/custom-diffusion"),
                "Prodia_img":lambda user_input: gen_img(user_input, "prodia"),
                "Pollinations_img":lambda user_input: gen_img(user_input, "pollinations")}

def communicate_with_VLM(user_input, model):
    try:
        image_path =filedialog.askopenfilename(
            title="Выберите изображение",
            filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.gif;*.bmp")]
        )
        if image_path:
            vlm_instance = VLM(model=model, is_conversation=True, max_tokens=600, timeout=30)
            image_base64 = vlm_instance.encode_image_to_base64(image_path)

            prompt = {
                "content": f"{user_input}",
                "image": image_base64
            }

            # Generate a response
            response = vlm_instance.chat(prompt)
            return response
    except Exception as e:
        return f"Ошибка при общении с DeepInfraAI: {e}"

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

        return f"Картинка сохранена в: {image_path}"

    except Exception as e:
       return f"Ошибка при генерации картинки: {e}"


def communicate_with_DeepInfraImager(user_input, model):
    try:
        ai = DeepInfraImager()
        ai.model = model
        resp = ai.generate(user_input, 1)

        # Проверяем, что resp - это список изображений
        if isinstance(resp, list) and len(resp) > 0:
            num_images = len(resp)  # Количество изображений
        else:
            raise ValueError("Генерация изображения не удалась, получен пустой ответ.")

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

        return f"Сохранено {num_images} изображений: {', '.join(saved_images)}"
    except Exception as e:
        return f"Ошибка при генерации картинки: {e}"

class ChatApp(ctk.CTk):
    def __init__(self):
        try:
            super().__init__()
            ctk.set_appearance_mode("dark")
            ctk.set_default_color_theme("green")

            self.isTranslate = False

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

            # Метка
            self.model_label = ctk.CTkLabel(self.input_frame, text="Выберите модель:", font=("Consolas", 18))
            self.model_label.pack(side="left", padx=5)

            # Комбобокс
            self.model_var = tk.StringVar()
            self.model_combobox = ctk.CTkOptionMenu(self.input_frame, variable=self.model_var, font=("Consolas", 16),
                                                    values=list(model_functions.keys()))
            self.model_combobox.pack(side="left", padx=5)
            self.model_combobox.set(list(model_functions.keys())[0])  # Модель по умолчанию


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

            # Кнопка переключения темы
            self.theme_button = ctk.CTkButton(self.button_frame, text="Светлая тема", command=self.toggle_theme,
                                              font=("Consolas", 14), text_color="white")
            self.theme_button.pack(side="top", padx=5, pady=10)

            # Кнопка переключения языка
            self.lang_button = ctk.CTkButton(self.button_frame, text="English", command=self.toggle_lang,
                                             font=("Consolas", 14), text_color="white")
            self.lang_button.pack(side="top", padx=5, pady=10)

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

    def set_active_widget(self, event):
        # Устанавливаем активный виджет
        self.active_widget = event.widget

    def hide_window(self):
        # Скрываем окно
        hwnd = ctypes.windll.user32.FindWindowW(None, self.title())
        ctypes.windll.user32.ShowWindow(hwnd, 0)  # 0 - SW_HIDE
        self.icon.run_detached()  # Запускаем иконку в отдельном потоке

    def show_window(self):
        # Показываем окно
        hwnd = ctypes.windll.user32.FindWindowW(None, self.title())
        ctypes.windll.user32.ShowWindow(hwnd, 5)  # 5 - SW_SHOW
        ctypes.windll.user32.SetForegroundWindow(hwnd)  # Устанавливаем окно на передний план
        # self.after(0, self.deiconify)

    def on_exit(self):
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
                    self.chat_history.insert(tk.END, f"Вы: {user_input}\n", "user_input")

                    response = model_functions[model](user_input)
                    self.chat_history.insert(tk.END, f"\nОтвет от {model}: {response}\n", "response")

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
        else:
            # Переключаем на английский
            self.model_label.configure(text="Select model:")
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

        self.isTranslate = not self.isTranslate  # Переключаем состояние

if __name__ == "__main__":
    check_for_updates()
    app = ChatApp()
    app.mainloop()
