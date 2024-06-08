from ai4free import KOBOLDAI, BLACKBOXAI, ThinkAnyAI, PhindSearch, DeepInfra
from freeGPT import Client
import tkinter as tk
from tkinter import ttk
import datetime


def communicate_with_model(message):
    """Взаимодействует с моделью для генерации ответа."""
    try:
        resp = Client.create_completion("gpt3", message)
        return resp
    except Exception as e:
        return f"Ошибка при общении с моделью: {e}"

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
        responce = ai.chat(user_input)
        return responce
    except Exception as e:
        return f"Ошибка при общении с BLACKBOXAI: {e}"

def communicate_with_ThinkAnyAI(user_input, model):
    try:
        opengpt = ThinkAnyAI(locale="ru", model=model)
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
        ai = DeepInfra(
            model="meta-llama/Meta-Llama-3-70B-Instruct",
            is_conversation=True,
            max_tokens=800,
            timeout=30,
            intro=None,
            filepath=None,
            update_file=True,
            proxies={},
            history_offset=10250,
            act=None,
        )
        message = ai.ask(user_input)
        responce = ai.get_message(message)
        return responce
    except Exception as e:
        return f"Ошибка при общении с DeepInfraAI: {e}"

class ChatApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AI Chat")
        self.geometry("{}x{}+0+0".format(self.winfo_screenwidth(), self.winfo_screenheight()))
        self.configure(bg="black")

        # Создание виджетов
        self.chat_history = tk.Text(self, bg="black", fg="green", state="disabled", font=("Consolas", 14))
        self.chat_history.pack(fill="both", expand=True, padx=10, pady=10)

        self.input_frame = tk.Frame(self, bg="black")
        self.input_frame.pack(fill="x", padx=10, pady=10)

        self.model_label = ttk.Label(self.input_frame, text="Выберите модель:", foreground="green", background="black", font=("Consolas", 14))
        self.model_label.pack(side="left", padx=5)

        self.model_var = tk.StringVar()
        self.model_combobox = ttk.Combobox(self.input_frame, textvariable=self.model_var, values=["GPT-3.5", "gpt-3.5-turbo", "KoboldAI", "BlackboxAI", "claude-3-haiku", "Phind", "Llama-2-70b-chat-hf", "llama-3-8b-instruct", "Llama-70b", "mistral-7b-instruct", "Mixtral-8x22B-Instruct-v0.1", "Mixtral-8x7B-Instruct-v0.1", "WizardLM-2-8x22B", "dolphin-2.6-mixtral-8x7b", "lzlv_70b_fp16_hf", "openchat_3.5", "MythoMax-L2-13b-turbo", "Phind-CodeLlama-34B-v2", "starcoder2-15b-instruct-v0.1", "airoboros-70b", "gemini-pro"], state="readonly")
        self.model_combobox.pack(side="left", padx=5)
        self.model_combobox.current(0)

        self.input_entry = tk.Text(self.input_frame, bg="black", fg="green", insertbackground="green", font=("Consolas", 14), height=10, width=50, wrap="word")
        self.input_entry.pack(side="left", fill="x", expand=True, padx=5)

        # Кнопки
        self.send_button = tk.Button(self.input_frame, text="Отправить", bg="black", fg="green", font=("Consolas", 14),
                                     command=self.send_message, borderwidth=0)
        self.send_button.pack(side="top", padx=5, pady=10)

        self.clear_button = tk.Button(self.input_frame, text="Очистить чат", bg="black", fg="green",
                                      font=("Consolas", 14), command=self.clear_chat, borderwidth=0)
        self.clear_button.pack(side="top", padx=5, pady=15)

        # Чекбокс "Вести историю"
        style = ttk.Style()
        style.configure("CustomCheckbutton.TCheckbutton",
                        font=("Consolas", 14),
                        background="black",
                        foreground="yellow",
                        relief="flat")

        input_frame = tk.Frame(self, bg="black")
        input_frame.pack()

        self.history_var = tk.BooleanVar()
        self.history_checkbox = ttk.Checkbutton(
            input_frame,
            text="Вести историю",
            variable=self.history_var,
            onvalue=True,
            offvalue=False,
            style="CustomCheckbutton.TCheckbutton"
        )
        self.history_checkbox.pack(side="top", padx=5, pady=5)

        self.input_entry.bind("<Return>", self.send_message)

    def write_history(self, user_input, responce):
        now = datetime.datetime.now()
        text = f"Дата и время: {now.strftime('%d.%m.%Y %H:%M:%S')}\nЗапрос пользователя: {user_input}\nОтвет ИИ: {responce}\n\n{100*"_"}"
        with open("history.txt", "a") as f:
            f.write(text)
    def send_message(self, event=None):
        user_input = self.input_entry.get("1.0", "end-1c").strip()
        if user_input:
            model = self.model_var.get()
            if model == "GPT-3.5":
                response = communicate_with_model(user_input)
            elif model == "KoboldAI":
                response = communicate_with_KoboldAI(user_input)
            elif model == "BlackboxAI":
                response = communicate_with_BlackboxAI(user_input)
            elif model == "claude-3-haiku":
                response = communicate_with_ThinkAnyAI(user_input, "claude-3-haiku")
            elif model == "Phind":
                response = communicate_with_Phind(user_input)
            elif model == "Llama-70b":
                response = communicate_with_DeepInfra(user_input)
            elif model == "mistral-7b-instruct":
                response = communicate_with_ThinkAnyAI(user_input, "mistral-7b-instruct")
            elif model == "llama-3-8b-instruct":
                response = communicate_with_ThinkAnyAI(user_input, "llama-3-8b-instruct")
            elif model == "gemini-pro":
                response = communicate_with_ThinkAnyAI(user_input, "gemini-pro")
            elif model == "gpt-3.5-turbo":
                response = communicate_with_ThinkAnyAI(user_input, "gpt-3.5-turbo")
            elif model == "Mixtral-8x22B-Instruct-v0.1":
                response = communicate_with_DeepInfra(user_input, "mistralai/Mixtral-8x22B-Instruct-v0.1")
            elif model == "WizardLM-2-8x22B":
                response = communicate_with_DeepInfra(user_input, "microsoft/WizardLM-2-8x22B")
            elif model == "Mixtral-8x7B-Instruct-v0.1":
                response = communicate_with_DeepInfra(user_input, "mistralai/Mixtral-8x7B-Instruct-v0.1")
            elif model == "dolphin-2.6-mixtral-8x7b":
                response = communicate_with_DeepInfra(user_input, "cognitivecomputations/dolphin-2.6-mixtral-8x7b")
            elif model == "lzlv_70b_fp16_hf":
                response = communicate_with_DeepInfra(user_input, "lizpreciatior/lzlv_70b_fp16_hf")
            elif model == "openchat_3.5":
                response = communicate_with_DeepInfra(user_input, "openchat/openchat_3.5")
            elif model == "MythoMax-L2-13b-turbo":
                response = communicate_with_DeepInfra(user_input, "Gryphe/MythoMax-L2-13b-turbo")
            elif model == "Phind-CodeLlama-34B-v2":
                response = communicate_with_DeepInfra(user_input, "Phind/Phind-CodeLlama-34B-v2")
            elif model == "starcoder2-15b-instruct-v0.1":
                response = communicate_with_DeepInfra(user_input, "bigcode/starcoder2-15b-instruct-v0.1")
            elif model == "airoboros-70b":
                response = communicate_with_DeepInfra(user_input, "deepinfra/airoboros-70b")
            elif model == "Llama-2-70b-chat-hf":
                response = communicate_with_DeepInfra(user_input, "meta-llama/Llama-2-70b-chat-hf")
            else:
                response = "Пожалуйста, выберите модель"
            if self.history_checkbox:
                self.write_history(user_input, response)
            self.chat_history.configure(state="normal")
            self.chat_history.insert("end", f"Вы: {user_input}\n")
            self.chat_history.insert("end", f"Ответ: {response}\n")
            self.chat_history.configure(state="disabled")
            self.input_entry.delete("1.0", "end")  # Очистка поля ввода

    def clear_chat(self):
        self.chat_history.configure(state="normal")
        self.chat_history.delete("1.0", "end")
        self.chat_history.configure(state="disabled")

if __name__ == "__main__":
    app = ChatApp()
    app.mainloop()