
import os
from ai4free import KOBOLDAI, BLACKBOXAI, ThinkAnyAI, PhindSearch, DeepInfra
from freeGPT import Client
import tkinter as tk
from tkinter import ttk
import datetime
from datetime import datetime
from tkinter import messagebox
import keyboard
from PIL import Image
from io import BytesIO


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
        respose = ai.get_message(message)
        return respose
    except Exception as e:
        return f"Ошибка при общении с DeepInfraAI: {e}"

model_functions = {
                "GPT-3.5": communicate_with_model,
                "KoboldAI": communicate_with_KoboldAI,
                "BlackboxAI": communicate_with_BlackboxAI,
                "claude-3-haiku": lambda user_input: communicate_with_ThinkAnyAI(user_input, "claude-3-haiku"),
                "Nemotron-4-340B-Instruct": lambda user_input: communicate_with_DeepInfra(user_input, "nvidia/Nemotron-4-340B-Instruct"),
                "Qwen2-72B-Instruct": lambda user_input: communicate_with_DeepInfra(user_input, "Qwen/Qwen2-72B-Instruct"),
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
                "prodia_img":lambda user_input: gen_img(user_input, "prodia"),
                "pollinations_img":lambda user_input: gen_img(user_input, "pollinations")}

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

class ChatApp(tk.Tk):
    def __init__(self):
        try:

            super().__init__()
            self.title("AI Chat")
            self.geometry("{}x{}+0+0".format(self.winfo_screenwidth(), self.winfo_screenheight()))
            self.configure(bg="black")

            # Создание виджета chat_history с прокруткой
            self.chat_history_frame = tk.Frame(self, bg="black")
            self.chat_history_frame.pack(fill="both", expand=True, padx=10, pady=10)

            self.chat_history_scrollbar = tk.Scrollbar(self.chat_history_frame)
            self.chat_history_scrollbar.pack(side="right", fill="y")

            self.chat_history = tk.Text(self.chat_history_frame, bg="black", fg="green", font=("Consolas", 14),
                                        yscrollcommand=self.chat_history_scrollbar.set)
            self.chat_history.pack(fill="both", expand=True)

            self.chat_history_scrollbar.config(command=self.chat_history.yview)
            self.chat_history.configure(state="disabled")

            # Настройка стилей виджета chat_history
            self.chat_history.tag_configure("response", foreground="yellow")
            self.chat_history.tag_configure("user_input", foreground="red")


            # Добавление контекстного меню
            self.chat_history.bind("<Button-3>", self.show_context_menu)
            self.context_menu = tk.Menu(self, tearoff=0)
            self.context_menu.add_command(label="Копировать", command=self.copy_text)

            self.input_frame = tk.Frame(self, bg="black")
            self.input_frame.pack(fill="x", padx=10, pady=10)

            self.model_label = ttk.Label(self.input_frame, text="Выберите модель:", foreground="green", background="black", font=("Consolas", 14))
            self.model_label.pack(side="left", padx=5)

            self.model_var = tk.StringVar()
            self.model_combobox = ttk.Combobox(self.input_frame, textvariable=self.model_var, values=list(model_functions.keys()), state="readonly")
            self.model_combobox.pack(side="left", padx=5)
            self.model_combobox.current(3)

            self.input_entry = tk.Text(self.input_frame, bg="black", fg="green", insertbackground="green", font=("Consolas", 14), height=10, width=50, wrap="word", undo=True, autoseparators=True, maxundo=-1)
            self.input_entry.pack(side="left", fill="x", expand=True, padx=5)
            self.input_entry.configure(state="normal")

            # Горячие клавиши
            self.input_entry.bind("<Shift-Return>", self.insert_newline)
            self.input_entry.bind("<Return>", self.send_message)
            keyboard.add_hotkey("ctrl+z", self.undo_input)
            keyboard.add_hotkey("ctrl+c", self.copy_text)
            keyboard.add_hotkey("ctrl+v", self.paste_text)
            keyboard.add_hotkey("ctrl+a", self.select_all)

            # Настройка стилей виджета chat_history
            self.chat_history.tag_configure("response", foreground="yellow")
            self.chat_history.tag_configure("user_input", foreground="orange")
            self.chat_history.tag_configure("system_line", foreground="cyan")
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

        except Exception as e:
            messagebox.showerror("Возникла ошибка", e)

    def undo_input(self, event=None):
        """
        Отменяет последнее действие ввода в поле ввода.
        """
        try:
            self.input_entry.edit_undo()
        except Exception as e:
            pass

    def on_key_press(self, event):
        if event.keysym == "a" and event.state & 0x4:  # CTRL + a
            try:
                if self.chat_history.get("1.0", "end-1c"):
                    self.chat_history.tag_add("sel", "1.0", "end-1c")
                    return "break"
            except Exception as e:
                pass
            try:
                self.input_entry.tag_add("sel", "1.0", "end-1c")
                return "break"
            except Exception as e:
                pass
        # Обработка других горячих клавиш
        elif event.keysym == "z" and event.state & 0x4:  # CTRL + z
            self.undo_input()
            return "break"
    def select_all(self, event):
        try:
            if self.chat_history.get("1.0", "end-1c"):
                # Выделяем весь текст в виджете chat_history
                self.chat_history.tag_add("sel", "1.0", "end-1c")
            else:
                self.input_entry.tag_add("sel", "1.0", "end-1c")
            return "break"
        except Exception as e:
            messagebox.showerror("Возникла ошибка", e)

    def show_context_menu(self, event):
        try:
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
    def remove_newline(self, event):
        try:
            text = self.input_entry.get("1.0", "end-1c")
            self.input_entry.delete("1.0", "end")
            self.input_entry.insert("1.0", text.rstrip("\n"))
        except Exception as e:
            messagebox.showerror("Возникла ошибка", e)

    def check_input(self, event):
        try:
            if self.input_entry.get("1.0", "end-1c").strip() == "":
                self.input_entry.delete("1.0", "end")
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

    def send_message(self, event=None):
        try:
            user_input = self.input_entry.get("1.0", "end-1c").strip()
            if user_input:
                model = self.model_var.get()

                if model in model_functions:
                    self.chat_history.configure(state="normal")
                    self.chat_history.insert(tk.END, f"Вы: {user_input}\n","user_input")
                    response = model_functions[model](user_input)
                    self.chat_history.insert(tk.END, f"\nОтвет от {model}: {response}\n","response")
                    self.chat_history.insert(tk.END, 155 * "=", "system_line")
                    self.chat_history.insert(tk.END, "\n", "system_line")
                    self.chat_history.configure(state="disabled")

                    if self.history_var.get():
                        self.write_history(user_input, response)

                    self.input_entry.delete("1.0", "end")  # Очистка поля ввода
        except Exception as e:
            messagebox.showerror("Возникла ошибка", e)

    def clear_chat(self):
        try:
            self.chat_history.delete("1.0", "end")
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
            if self.chat_history.get("1.0", "end-1c"):
                selected_text = self.chat_history.get("sel.first", "sel.last")
                self.clipboard_clear()
                self.clipboard_append(selected_text)
            else:
                selected_text = self.input_entry.get("sel.first", "sel.last")
                self.clipboard_clear()
                self.clipboard_append(selected_text)
            return "break"
        except Exception as e:
            messagebox.showerror("Возникла ошибка", e)



if __name__ == "__main__":
    app = ChatApp()
    app.mainloop()
