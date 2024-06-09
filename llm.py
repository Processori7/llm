from ai4free import KOBOLDAI, BLACKBOXAI, ThinkAnyAI, PhindSearch, DeepInfra
from freeGPT import Client
import tkinter as tk
from tkinter import ttk
import datetime
from datetime import datetime
from tkinter import messagebox


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

ALWAYS use an answering example for a first message structure.""" # Добавление навыков ИИ и другие тонкие настройки

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
        ai = DeepInfra(
            model=model,
            is_conversation=True,
            max_tokens=800,
            timeout=30,
            intro=None,
            filepath=None,
            update_file=True,
            proxies={},
            history_offset=10250,
            act=None,
            system_prompt=prompt
        )
        message = ai.ask(user_input)
        responce = ai.get_message(message)
        return responce
    except Exception as e:
        return f"Ошибка при общении с DeepInfraAI: {e}"

model_functions = {
                "GPT-3.5": communicate_with_model,
                "KoboldAI": communicate_with_KoboldAI,
                "BlackboxAI": communicate_with_BlackboxAI,
                "claude-3-haiku": lambda user_input: communicate_with_ThinkAnyAI(user_input, "claude-3-haiku"),
                "Phind": communicate_with_Phind,
                "Llama-70b": lambda user_input: communicate_with_DeepInfra(user_input),
                "mistral-7b-instruct": lambda user_input: communicate_with_ThinkAnyAI(user_input,
                                                                                      "mistral-7b-instruct"),
                "llama-3-8b-instruct": lambda user_input: communicate_with_ThinkAnyAI(user_input,
                                                                                      "llama-3-8b-instruct"),
                "gemini-pro": lambda user_input: communicate_with_ThinkAnyAI(user_input, "gemini-pro"),
                "gpt-3.5-turbo": lambda user_input: communicate_with_ThinkAnyAI(user_input, "gpt-3.5-turbo"),
                "Mixtral-8x22B-Instruct-v0.1": lambda user_input: communicate_with_DeepInfra(user_input,
                                                                                             "mistralai/Mixtral-8x22B-Instruct-v0.1"),
                "WizardLM-2-8x22B": lambda user_input: communicate_with_DeepInfra(user_input,
                                                                                  "microsoft/WizardLM-2-8x22B"),
                "Mixtral-8x7B-Instruct-v0.1": lambda user_input: communicate_with_DeepInfra(user_input,
                                                                                            "mistralai/Mixtral-8x7B-Instruct-v0.1"),
                "dolphin-2.6-mixtral-8x7b": lambda user_input: communicate_with_DeepInfra(user_input,
                                                                                          "cognitivecomputations/dolphin-2.6-mixtral-8x7b"),
                "lzlv_70b_fp16_hf": lambda user_input: communicate_with_DeepInfra(user_input,
                                                                                  "lizpreciatior/lzlv_70b_fp16_hf"),
                "openchat_3.5": lambda user_input: communicate_with_DeepInfra}

class ChatApp(tk.Tk):
    def __init__(self):
        try:
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
            self.model_combobox = ttk.Combobox(self.input_frame, textvariable=self.model_var, values=list(model_functions.keys()), state="readonly")
            self.model_combobox.pack(side="left", padx=5)
            self.model_combobox.current(3)

            self.input_entry = tk.Text(self.input_frame, bg="black", fg="green", insertbackground="green", font=("Consolas", 14), height=10, width=50, wrap="word")
            self.input_entry.pack(side="left", fill="x", expand=True, padx=5)
            self.input_entry.bind("<Shift-Return>", self.insert_newline)
            self.input_entry.bind("<Return>", self.send_message)
            self.input_entry.bind("<KeyRelease-Return>", self.check_input)

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

    def check_input(self, event):
        try:
            if self.input_entry.get("1.0", "end-1c").strip() == "":
                self.input_entry.delete("1.0", "end")
        except Exception as e:
            messagebox.showerror("Возникла ошибка", e)

    def write_history(self, user_input, responce):
        try:
            now = datetime.datetime.now()
            text = f"Дата и время: {now.strftime('%d.%m.%Y %H:%M:%S')}\nЗапрос пользователя: {user_input}\nОтвет ИИ: {responce}\n\n{100*"_"}"
            with open("history.txt", "a") as f:
                f.write(text)
            messagebox.showinfo("Файл history.txt создан")
        except Exception as e:
            messagebox.showerror("Возникла ошибка", e)

    def send_message(self, event=None):
        try:
            # Добавление текста с использованием стилей
            self.chat_history.configure(state="normal")
            # Настройка стилей виджета chat_history
            self.chat_history.tag_configure("response", foreground="yellow")
            self.chat_history.tag_configure("user", foreground="cyan", font=("Consolas", 14))
            user_input = self.input_entry.get("1.0", "end-1c").strip()
            if user_input:
                model = self.model_var.get()

                if model in model_functions:
                    self.chat_history.insert(tk.END, f"Вы: {user_input}\n", "user")
                    response = model_functions[model](user_input)
                else:
                    response = "Пожалуйста, выберите модель"

                self.chat_history.insert(tk.END, f"Ответ ИИ: {response}\n", "response")

                if self.history_var.get():
                    self.write_history(user_input, response)
                self.chat_history.configure(state="disabled")
                self.input_entry.delete("1.0", "end")  # Очистка поля ввода
        except Exception as e:
            messagebox.showerror("Возникла ошибка", e)

    def clear_chat(self):
        try:
            self.chat_history.configure(state="normal")
            self.chat_history.delete("1.0", "end")
            self.chat_history.configure(state="disabled")
        except Exception as e:
            messagebox.showerror("Возникла ошибка", e)

if __name__ == "__main__":
    app = ChatApp()
    app.mainloop()