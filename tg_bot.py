import os
import json
import logging
import threading
import asyncio

from typing import Dict, Callable
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters, CallbackQueryHandler

# Загружаем переменные окружения
load_dotenv()

# Настройка логов
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Подключаем модели из llm.py
import llm
from llm import model_functions

class TelegramBot:
    def __init__(self, model_functions: Dict[str, Callable]):
        self.model_functions = model_functions
        self.bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        if not self.bot_token:
            raise ValueError("Токен TELEGRAM_BOT_TOKEN не найден в .env")

        self.user_data = {}  # Хранение данных пользователя (модель, язык, поиск)
        self.user_data_file = "telegram_users.json"  # Файл для сохранения данных пользователей
        self.application = None
        self.running = False
        self.bot_thread = None
        
        # Загружаем данные пользователей при инициализации
        self.load_user_data()
        
        # Локализация
        self.texts = {
            "ru": {
                "welcome": "🚀 Добро пожаловать в AI бот!",
                "current_model": "🤖 Текущая модель:",
                "send_message": "💬 Отправьте сообщение для ответа или выберите категорию:",
                "select_category": "выберите категорию:",
                "text_models": "📝 Текстовые модели",
                "img_models": "🖼️ Генерация изображений",
                "photo_models": "📷 Анализ фото",
                "web_models": "🌐 Веб-поиск",
                "all_models": "📁 Все модели",
                "refresh_models": "🔄 Обновить",
                "search_models": "🔍 Поиск",
                "language": "🌍 Язык",
                "main_menu": "🔙 Главное меню",
                "model_selected": "✅ Модель выбрана!",
                "model_type": "🏷️ Тип:",
                "model_name": "🤖 Модель:",
                "updating_models": "🔄 Обновляем модели...",
                "no_models": "Нет моделей в категории",
                "model_not_found": "❌ Модель не найдена.",
                "error": "❌ Ошибка:",
                "type_text": "📝 Текстовая",
                "type_img": "🖼️ Генерация изображений",
                "type_photo": "📷 Анализ изображений",
                "type_web": "🌐 Веб-поиск",
                "search_placeholder": "🔍 Введите запрос для поиска моделей...",
                "search_results": "🔍 Результаты поиска",
                "no_search_results": "😔 По вашему запросу ничего не найдено.",
                "search_query": "Запрос:",
                "models_found": "моделей найдено",
                "cancel_search": "❌ Отменить поиск",
                "language_selected": "🌍 Язык изменен на русский",
                "no_model_selected": "🔄 Модель не выбрана",
                "choose_model": "👆 Выберите модель с помощью кнопок ниже:"
            },
            "en": {
                "welcome": "🚀 Welcome to AI Bot!",
                "current_model": "🤖 Current model:",
                "send_message": "💬 Send a message to get a response or choose a category:",
                "select_category": "choose a category:",
                "text_models": "📝 Text models",
                "img_models": "🖼️ Image generation",
                "photo_models": "📷 Photo analysis",
                "web_models": "🌐 Web search",
                "all_models": "📁 All models",
                "refresh_models": "🔄 Refresh",
                "search_models": "🔍 Search",
                "language": "🌍 Language",
                "main_menu": "🔙 Main menu",
                "model_selected": "✅ Model selected!",
                "model_type": "🏷️ Type:",
                "model_name": "🤖 Model:",
                "updating_models": "🔄 Updating models...",
                "no_models": "No models in category",
                "model_not_found": "❌ Model not found.",
                "error": "❌ Error:",
                "type_text": "📝 Text",
                "type_img": "🖼️ Image generation",
                "type_photo": "📷 Photo analysis",
                "type_web": "🌐 Web search",
                "search_placeholder": "🔍 Enter search query for models...",
                "search_results": "🔍 Search results",
                "no_search_results": "😔 No results found for your query.",
                "search_query": "Query:",
                "models_found": "models found",
                "cancel_search": "❌ Cancel search",
                "language_selected": "🌍 Language changed to English",
                "no_model_selected": "🔄 No model selected",
                "choose_model": "👆 Choose a model using the buttons below:"
            }
        }

    def get_user_language(self, user_id: int) -> str:
        """Получить язык пользователя"""
        return self.user_data.get(user_id, {}).get("language", "ru")
    
    def get_text(self, user_id: int, key: str) -> str:
        """Получить локализованный текст"""
        lang = self.get_user_language(user_id)
        return self.texts[lang].get(key, key)
    
    def escape_markdown(self, text: str) -> str:
        """Экранирование специальных символов Markdown"""
        # Основные символы, которые нужно экранировать
        escape_chars = ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']
        
        for char in escape_chars:
            text = text.replace(char, f' {char}')
        
        return text
    
    def search_models(self, query: str) -> list:
        """Полнотекстовый поиск по моделям"""
        query_lower = query.lower()
        results = []
        
        for model_name in self.model_functions.keys():
            # Поиск по названию модели
            if query_lower in model_name.lower():
                results.append(model_name)
        
        return results

    def load_user_data(self):
        """Загружает данные пользователей из файла"""
        try:
            if os.path.exists(self.user_data_file):
                with open(self.user_data_file, 'r', encoding='utf-8') as f:
                    # Загружаем данные и конвертируем строковые ключи обратно в int
                    data = json.load(f)
                    self.user_data = {int(k): v for k, v in data.items()}
                    print(f"📚 Загружены данные {len(self.user_data)} пользователей из {self.user_data_file}")
            else:
                print(f"📝 Файл {self.user_data_file} не найден, создается новая база пользователей")
                self.user_data = {}
        except Exception as e:
            print(f"⚠️ Ошибка загрузки данных пользователей: {e}")
            self.user_data = {}
    
    def save_user_data(self):
        """Сохраняет данные пользователей в файл"""
        try:
            # Конвертируем int ключи в строки для JSON
            data_to_save = {str(k): v for k, v in self.user_data.items()}
            with open(self.user_data_file, 'w', encoding='utf-8') as f:
                json.dump(data_to_save, f, ensure_ascii=False, indent=2)
            print(f"💾 Данные {len(self.user_data)} пользователей сохранены в {self.user_data_file}")
        except Exception as e:
            print(f"⚠️ Ошибка сохранения данных пользователей: {e}")

    def filter_models(self, category="all"):
        """Фильтрация моделей по категориям"""
        categories = {
            "all": lambda m: True,
            "text": lambda m: not m.endswith("_img") and "(Photo Analyze)" not in m and "_Web" not in m and "(Polination)" not in m,
            "img": lambda m: "_img" in m or "(Polination)" in m,
            "photo": lambda m: "(Photo Analyze)" in m,
            "web": lambda m: "_Web" in m,
        }
        return [m for m in self.model_functions if categories.get(category, lambda m: False)(m)]

    def get_main_menu_keyboard(self, user_id: int = None):
        """Создает кнопки главного меню"""
        if user_id:
            text_models = self.get_text(user_id, "text_models")
            img_models = self.get_text(user_id, "img_models")
            photo_models = self.get_text(user_id, "photo_models")
            web_models = self.get_text(user_id, "web_models")
            all_models = self.get_text(user_id, "all_models")
            refresh_models = self.get_text(user_id, "refresh_models")
            search_models = self.get_text(user_id, "search_models")
            language = self.get_text(user_id, "language")
        else:
            text_models = "📝 Текст"
            img_models = "🖼️ Изображения"
            photo_models = "📷 Анализ фото"
            web_models = "🌐 Веб-поиск"
            all_models = "📁 Все модели"
            refresh_models = "🔄 Обновить"
            search_models = "🔍 Поиск"
            language = "🌍 Язык"
        
        return [
            [InlineKeyboardButton(text_models, callback_data="category_text"),
             InlineKeyboardButton(img_models, callback_data="category_img")],
            [InlineKeyboardButton(photo_models, callback_data="category_photo"),
             InlineKeyboardButton(web_models, callback_data="category_web")],
            [InlineKeyboardButton(all_models, callback_data="category_all"),
             InlineKeyboardButton(refresh_models, callback_data="refresh_models")],
            [InlineKeyboardButton(search_models, callback_data="search_models"),
             InlineKeyboardButton(language, callback_data="change_language")]
        ]

    async def show_category_models(self, update: Update, category: str, page: int = 0):
        """Показывает модели категории с пагинацией"""
        query = update.callback_query
        user_id = query.from_user.id
        
        # Получаем название категории на языке пользователя
        category_names = {
            "all": "all_models",
            "text": "text_models",
            "img": "img_models",
            "photo": "photo_models",
            "web": "web_models"
        }
        
        title = self.get_text(user_id, category_names.get(category, "all_models"))
        
        models = self.filter_models(category)
        if not models:
            keyboard = [[InlineKeyboardButton(self.get_text(user_id, "main_menu"), callback_data="back_to_menu")]]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await query.edit_message_text(text=f"{self.get_text(user_id, 'no_models')} '{title}'", reply_markup=reply_markup)
            return
        
        # Пагинация - показываем по 15 моделей на страницу
        models_per_page = 15
        start_idx = page * models_per_page
        end_idx = start_idx + models_per_page
        page_models = models[start_idx:end_idx]
        
        total_pages = (len(models) + models_per_page - 1) // models_per_page
        
        # Создаем кнопки для моделей на текущей странице
        keyboard = []
        for model in page_models:
            display_name = model if len(model) <= 40 else model[:37] + "..."
            keyboard.append([InlineKeyboardButton(display_name, callback_data=f"model_{model}")])
        
        # Кнопки навигации по страницам
        if total_pages > 1:
            nav_buttons = []
            if page > 0:
                nav_buttons.append(InlineKeyboardButton("⬅️ Назад", callback_data=f"page_{category}_{page-1}"))
            
            nav_buttons.append(InlineKeyboardButton(f"{page+1}/{total_pages}", callback_data="noop"))
            
            if page < total_pages - 1:
                nav_buttons.append(InlineKeyboardButton("Вперёд ➡️", callback_data=f"page_{category}_{page+1}"))
            
            keyboard.append(nav_buttons)
        
        # Кнопка возврата в главное меню
        keyboard.append([InlineKeyboardButton(self.get_text(user_id, "main_menu"), callback_data="back_to_menu")])
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        message_text = f"🎯 {title}\n\n"
        message_text += f"📊 {self.get_text(user_id, 'models_found')}: {len(models)}\n"
        if total_pages > 1:
            message_text += f"📄 Страница: {page+1}/{total_pages}\n\n"
        else:
            message_text += "\n"
        message_text += f"Выберите модель:"
        
        try:
            await query.edit_message_text(text=message_text, reply_markup=reply_markup)
        except Exception as e:
            # Если не можем отредактировать сообщение, отправляем новое
            await query.message.reply_text(text=message_text, reply_markup=reply_markup)

        async def show_search_results_page(self, update: Update, query_text: str, page: int):
            """Показывает страницу с результатами поиска с пагинацией"""
            query = update.callback_query
            user_id = query.from_user.id

            search_results = self.search_models(query_text)
            if not search_results:
                keyboard = [[InlineKeyboardButton(self.get_text(user_id, "main_menu"), callback_data="back_to_menu")]]
                reply_markup = InlineKeyboardMarkup(keyboard)
                await query.edit_message_text(text="Результаты не найдены.", reply_markup=reply_markup)
                return

            # Пагинация - показываем по 15 моделей на страницу
            models_per_page = 15
            start_idx = page * models_per_page
            end_idx = start_idx + models_per_page
            page_models = search_results[start_idx:end_idx]

            total_pages = (len(search_results) + models_per_page - 1) // models_per_page

            # Создаем кнопки для моделей на текущей странице
            keyboard = []
            for model in page_models:
                display_name = model if len(model) <= 40 else model[:37] + "..."
                keyboard.append([InlineKeyboardButton(display_name, callback_data=f"model_{model}")])

            # Кнопки навигации по страницам
            if total_pages > 1:
                nav_buttons = []
                if page > 0:
                    nav_buttons.append(InlineKeyboardButton("⬅️ Назад", callback_data=f"search_page_{query_text}_{page-1}"))

                nav_buttons.append(InlineKeyboardButton(f"{page+1}/{total_pages}", callback_data="noop"))

                if page < total_pages - 1:
                    nav_buttons.append(InlineKeyboardButton("Вперёд ➡️", callback_data=f"search_page_{query_text}_{page+1}"))

                keyboard.append(nav_buttons)

            # Кнопка возврата в главное меню
            keyboard.append([InlineKeyboardButton(self.get_text(user_id, "main_menu"), callback_data="back_to_menu")])

            reply_markup = InlineKeyboardMarkup(keyboard)

            message_text = f"🔍 {self.get_text(user_id, 'search_results')} ({len(search_results)} {self.get_text(user_id, 'models_found')})\n\n"
            message_text += f"{self.get_text(user_id, 'search_query')}: {query_text}\n\n"
            if total_pages > 1:
                message_text += f"📄 Страница: {page+1}/{total_pages}\n\n"
            message_text += "Выберите модель:"

            try:
                await query.edit_message_text(text=message_text, reply_markup=reply_markup)
            except Exception as e:
                await query.message.reply_text(text=message_text, reply_markup=reply_markup)

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await self.show_main_menu(update)

    async def show_main_menu(self, update: Update, edit_message=False):
        # Получаем информацию о пользователе
        user_id = update.effective_user.id if hasattr(update, 'effective_user') else None
        if not user_id:
            user_id = update.callback_query.from_user.id if edit_message else update.message.from_user.id
        
        # Инициализируем данные пользователя, если их нет
        if user_id not in self.user_data:
            self.user_data[user_id] = {"language": "ru"}
            # Сохраняем данные после добавления нового пользователя
            self.save_user_data()
        
        keyboard = self.get_main_menu_keyboard(user_id)
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        current_model = self.user_data.get(user_id, {}).get("model")
        
        message_text = self.get_text(user_id, "welcome") + "\n\n"
        if current_model:
            message_text += f"{self.get_text(user_id, 'current_model')} {self.escape_markdown(current_model)}\n\n"
        message_text += self.escape_markdown(self.get_text(user_id, "send_message"))
        
        if edit_message:
            try:
                await update.callback_query.edit_message_text(message_text, reply_markup=reply_markup)
            except Exception as e:
                # Если не можем отредактировать сообщение, отправляем новое
                await update.callback_query.message.reply_text(message_text, reply_markup=reply_markup)
        else:
            await update.message.reply_text(message_text, reply_markup=reply_markup)

    async def button_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()
        user_id = query.from_user.id
        
        # Инициализируем данные пользователя
        if user_id not in self.user_data:
            self.user_data[user_id] = {"language": "ru"}

        data = query.data
        if data == "refresh_models":
            # Обновляем модели
            try:
                await query.edit_message_text(self.get_text(user_id, "updating_models"))
            except Exception:
                await query.message.reply_text(self.get_text(user_id, "updating_models"))
            
            # Получаем новые модели для бота
            new_img_models = llm.get_Polinations_img_models_for_bot()
            new_chat_models = llm.get_Polinations_chat_models()
            
            # Добавляем новые модели в словарь
            if isinstance(new_img_models, dict):
                self.model_functions.update(new_img_models)
            if isinstance(new_chat_models, dict):
                self.model_functions.update(new_chat_models)
            
            # Возвращаемся к главному меню
            await self.show_main_menu(update, edit_message=True)
            
        elif data == "change_language":
            # Смена языка
            current_lang = self.get_user_language(user_id)
            new_lang = "en" if current_lang == "ru" else "ru"
            self.user_data[user_id]["language"] = new_lang
            # Сохраняем изменения
            self.save_user_data()
            
            # Показываем сообщение о смене языка
            keyboard = self.get_main_menu_keyboard(user_id)
            reply_markup = InlineKeyboardMarkup(keyboard)
            try:
                await query.edit_message_text(self.get_text(user_id, "language_selected"), reply_markup=reply_markup)
            except Exception:
                await query.message.reply_text(self.get_text(user_id, "language_selected"), reply_markup=reply_markup)
            
        elif data == "search_models":
            # Начало поиска
            self.user_data[user_id]["search_mode"] = True
            keyboard = [[InlineKeyboardButton(self.get_text(user_id, "cancel_search"), callback_data="back_to_menu")]]
            reply_markup = InlineKeyboardMarkup(keyboard)
            try:
                await query.edit_message_text(self.get_text(user_id, "search_placeholder"), reply_markup=reply_markup)
            except Exception:
                await query.message.reply_text(self.get_text(user_id, "search_placeholder"), reply_markup=reply_markup)
            
        elif data == "back_to_menu":
            # Возвращаемся к главному меню
            self.user_data[user_id].pop("search_mode", None)  # Отменяем режим поиска
            await self.show_main_menu(update, edit_message=True)
            
        elif data.startswith("category_"):
            category = data.split("_")[1]
            await self.show_category_models(update, category, page=0)
            
        elif data.startswith("page_"):
            # Обработка пагинации категорий
            parts = data.split("_")
            if len(parts) >= 3:
                category = parts[1]
                page = int(parts[2])
                await self.show_category_models(update, category, page)
                
        elif data.startswith("search_page_"):
            # Обработка пагинации поиска
            parts = data.split("_", 3)  # search_page_query_page_number
            if len(parts) >= 4:
                search_query = parts[2]
                page = int(parts[3])
                await self.show_search_results_page(update, search_query, page)
                
        elif data == "noop":
            # Не делаем ничего - это кнопка с информацией
            pass

        elif data.startswith("model_"):
            model_name = data.split("model_", 1)[1]
            user_id = query.from_user.id
            if model_name in self.model_functions:
                # Сохраняем текущие данные пользователя и добавляем модель
                if user_id not in self.user_data:
                    self.user_data[user_id] = {"language": "ru"}
                self.user_data[user_id]["model"] = model_name
                # Сохраняем изменения
                self.save_user_data()
                
                # Определяем тип модели
                model_type = self.get_text(user_id, "type_text")
                if "_img" in model_name or "(Polination)" in model_name:
                    model_type = self.get_text(user_id, "type_img")
                elif "(Photo Analyze)" in model_name:
                    model_type = self.get_text(user_id, "type_photo")
                elif "_Web" in model_name:
                    model_type = self.get_text(user_id, "type_web")
                
                # Добавляем кнопки навигации
                keyboard = self.get_main_menu_keyboard(user_id)
                reply_markup = InlineKeyboardMarkup(keyboard)
                
                message_text = f"{self.get_text(user_id, 'model_selected')}\n\n"
                message_text += f"{self.get_text(user_id, 'model_name')} {self.escape_markdown(model_name)}\n"
                message_text += f"{self.get_text(user_id, 'model_type')} {self.escape_markdown(model_type)}\n\n"
                message_text += self.escape_markdown(self.get_text(user_id, "send_message"))
                
                try:
                    await query.edit_message_text(text=message_text, reply_markup=reply_markup)
                except Exception:
                    await query.message.reply_text(text=message_text, reply_markup=reply_markup)
            else:
                keyboard = [[InlineKeyboardButton(self.get_text(user_id, "main_menu"), callback_data="back_to_menu")]]
                reply_markup = InlineKeyboardMarkup(keyboard)
                try:
                    await query.edit_message_text(text=self.get_text(user_id, "model_not_found"), reply_markup=reply_markup)
                except Exception:
                    await query.message.reply_text(text=self.get_text(user_id, "model_not_found"), reply_markup=reply_markup)

    async def list_models_with_buttons(self, update: Update, context: ContextTypes.DEFAULT_TYPE, category, title):
        models = self.filter_models(category)
        if not models:
            await update.message.reply_text(f"Нет моделей в категории '{title}'")
            return

        buttons = []
        for model in models:
            buttons.append([InlineKeyboardButton(model, callback_data=f"model_{model}")])

        reply_markup = InlineKeyboardMarkup(buttons)
        await update.message.reply_text(f"Выберите модель из категории '{title}':", reply_markup=reply_markup)

    async def text_models(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await self.list_models_with_buttons(update, context, "text", "Текстовые модели")

    async def img_models(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await self.list_models_with_buttons(update, context, "img", "Генерация изображений")

    async def photo_models(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await self.list_models_with_buttons(update, context, "photo", "Анализ фото")

    async def web_models(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await self.list_models_with_buttons(update, context, "web", "Веб-поиск")

    async def all_models(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await self.list_models_with_buttons(update, context, "all", "Все модели")

    async def set_model(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if len(context.args) == 0:
            await update.message.reply_text("Укажите название модели.")
            return

        model_name = " ".join(context.args)
        if model_name in self.model_functions:
            user_id = update.effective_user.id
            self.user_data[user_id] = {"model": model_name}
            await update.message.reply_text(f"Выбрана модель: {model_name}")
        else:
            await update.message.reply_text("Модель не найдена.")

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        user_input = update.message.text.strip()
        if not user_input:
            return

        if self.user_data.get(user_id, {}).get("search_mode"):
            # Выполняем поиск моделей
            search_results = self.search_models(user_input)
            
            if not search_results:
                message_text = f"{self.get_text(user_id, 'no_search_results')}\n\n{self.get_text(user_id, 'search_query')} {user_input}"
                keyboard = [[InlineKeyboardButton(self.get_text(user_id, "main_menu"), callback_data="back_to_menu")]]
                reply_markup = InlineKeyboardMarkup(keyboard)
                await update.message.reply_text(message_text, reply_markup=reply_markup)
            else:
                # Показываем все результаты с пагинацией
                models_per_page = 15
                total_pages = (len(search_results) + models_per_page - 1) // models_per_page
                
                # Показываем первую страницу
                page_models = search_results[:models_per_page]
                
                keyboard = []
                for model in page_models:
                    display_name = model if len(model) <= 40 else model[:37] + "..."
                    keyboard.append([InlineKeyboardButton(display_name, callback_data=f"model_{model}")])
                
                # Кнопки навигации по страницам (только если страниц больше 1)
                if total_pages > 1:
                    nav_buttons = []
                    nav_buttons.append(InlineKeyboardButton(f"1/{total_pages}", callback_data="noop"))
                    if total_pages > 1:
                        nav_buttons.append(InlineKeyboardButton("Вперёд ➜️", callback_data=f"search_page_{user_input}_1"))
                    keyboard.append(nav_buttons)
                
                keyboard.append([InlineKeyboardButton(self.get_text(user_id, "main_menu"), callback_data="back_to_menu")])
                reply_markup = InlineKeyboardMarkup(keyboard)
                
                message_text = f"{self.get_text(user_id, 'search_results')} ({len(search_results)} {self.get_text(user_id, 'models_found')})\n\n"
                message_text += f"{self.get_text(user_id, 'search_query')}: {user_input}\n\n"
                if total_pages > 1:
                    message_text += f"📄 Страница: 1/{total_pages}\n\n"
                message_text += "Выберите модель:"
                
                await update.message.reply_text(message_text, reply_markup=reply_markup)
            
            self.user_data[user_id].pop("search_mode", None)  # Завершаем режим поиска
            return

        selected_model = self.user_data.get(user_id, {}).get("model")
        if not selected_model:
            selected_model = next(iter(self.model_functions))
            if user_id not in self.user_data:
                self.user_data[user_id] = {"language": "ru"}
            self.user_data[user_id]["model"] = selected_model
            # Сохраняем изменения
            self.save_user_data()

        try:
            # Добавляем кнопки меню к каждому ответу
            keyboard = self.get_main_menu_keyboard(user_id)
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            response = self.model_functions[selected_model](user_input)

            # Проверяем, является ли ответ URL изображения от Pollinations или содержит расширение изображения
            if (response.startswith(("http://", "https://")) and 
                ("image.pollinations.ai" in response or 
                 any(ext in response for ext in [".jpg", ".png", ".jpeg", ".gif", ".webp"]))):
                await update.message.reply_photo(photo=response, reply_markup=reply_markup)
            else:
                await update.message.reply_text(f"🤖 {selected_model}:\n\n{response}", reply_markup=reply_markup)

        except Exception as e:
            keyboard = self.get_main_menu_keyboard(user_id)
            reply_markup = InlineKeyboardMarkup(keyboard)
            await update.message.reply_text(f"{self.get_text(user_id, 'error')} {str(e)}", reply_markup=reply_markup)

    async def current_model(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Показать информацию о текущей модели"""
        user_id = update.effective_user.id
        current_model = self.user_data.get(user_id, {}).get("model")
        
        keyboard = self.get_main_menu_keyboard(user_id)
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        if current_model:
            # Определяем тип модели
            model_type = self.get_text(user_id, "type_text")
            if "_img" in current_model or "(Polination)" in current_model:
                model_type = self.get_text(user_id, "type_img")
            elif "(Photo Analyze)" in current_model:
                model_type = self.get_text(user_id, "type_photo")
            elif "_Web" in current_model:
                model_type = self.get_text(user_id, "type_web")
            
            message_text = f"{self.get_text(user_id, 'current_model')}\n\n"
            message_text += f"{self.get_text(user_id, 'model_name')} {current_model}\n"
            message_text += f"{self.get_text(user_id, 'model_type')} {model_type}\n\n"
            message_text += self.get_text(user_id, "send_message")
        else:
            message_text = f"{self.get_text(user_id, 'no_model_selected')}\n\n{self.get_text(user_id, 'choose_model')}"
        
        await update.message.reply_text(message_text, reply_markup=reply_markup)

    async def search_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Команда для поиска моделей"""
        user_id = update.effective_user.id
        
        if len(context.args) == 0:
            self.user_data[user_id] = self.user_data.get(user_id, {})
            self.user_data[user_id]["search_mode"] = True
            keyboard = [[InlineKeyboardButton(self.get_text(user_id, "cancel_search"), callback_data="back_to_menu")]]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await update.message.reply_text(self.get_text(user_id, "search_placeholder"), reply_markup=reply_markup)
            return
        
        # Поиск по запросу из аргументов
        query = " ".join(context.args)
        search_results = self.search_models(query)
        
        keyboard = self.get_main_menu_keyboard(user_id)
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        if not search_results:
            message_text = f"{self.get_text(user_id, 'no_search_results')}\n\n{self.get_text(user_id, 'search_query')}: {query}"
        else:
            message_text = f"{self.get_text(user_id, 'search_results')} ({len(search_results)} {self.get_text(user_id, 'models_found')})\n\n"
            message_text += f"{self.get_text(user_id, 'search_query')}: {query}\n\n"
            message_text += "\n".join([f"• {model}" for model in search_results[:10]])  # Показываем только первые 10
            if len(search_results) > 10:
                message_text += f"\n\n... и ещё {len(search_results) - 10} моделей"
        
        await update.message.reply_text(message_text, reply_markup=reply_markup)

    def setup_handlers(self):
        self.application.add_handler(CommandHandler("start", self.start))
        self.application.add_handler(CommandHandler("menu", self.start))  # Альтернативная команда для меню
        self.application.add_handler(CommandHandler("current", self.current_model))  # Инфо о текущей модели
        self.application.add_handler(CommandHandler("search", self.search_command))  # Поиск моделей
        self.application.add_handler(CommandHandler("text", self.text_models))
        self.application.add_handler(CommandHandler("img", self.img_models))
        self.application.add_handler(CommandHandler("photo", self.photo_models))
        self.application.add_handler(CommandHandler("web", self.web_models))
        self.application.add_handler(CommandHandler("all", self.all_models))
        self.application.add_handler(CommandHandler("set_model", self.set_model))

        self.application.add_handler(CallbackQueryHandler(self.button_handler))

        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))

    async def send_startup_greeting(self):
        """Отправляет приветственное сообщение всем пользователям при запуске бота"""
        try:
            # Получаем информацию о боте
            bot_info = await self.application.bot.get_me()
            startup_message = f"🚀 **{bot_info.first_name} бот запущен!**\n\n"
            startup_message += "✨ Новый сеанс работы начат\n"
            startup_message += "🤖 Готов к работе с AI моделями\n\n"
            startup_message += "Используйте /start для начала работы"
            
            # Отправляем сообщение всем пользователям, которые ранее взаимодействовали с ботом
            for user_id in self.user_data.keys():
                try:
                    await self.application.bot.send_message(
                        chat_id=user_id, 
                        text=startup_message,
                        parse_mode="Markdown"
                    )
                except Exception as e:
                    print(f"⚠️ Не удалось отправить приветственное сообщение пользователю {user_id}: {e}")
            
            print(f"📢 Приветственное сообщение отправлено {len(self.user_data)} пользователям")
        except Exception as e:
            print(f"❌ Ошибка при отправке приветственного сообщения: {e}")

    def run(self):
        if not self.bot_token:
            print("❌ Telegram бот не запущен: отсутствует TELEGRAM_BOT_TOKEN в .env")
            return

        print("✅ Telegram бот запущен...")
        
        # Создаем новый event loop для потока
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        self.application = ApplicationBuilder().token(self.bot_token).build()
        self.setup_handlers()
        self.running = True
        
        # Планируем отправку приветственного сообщения после запуска
        async def post_init(application):
            await self.send_startup_greeting()
        
        self.application.post_init = post_init
        
        try:
            self.application.run_polling()
        finally:
            loop.close()

    def start_bot(self):
        if self.running:
            print("⚠ Бот уже запущен")
            return
        self.bot_thread = threading.Thread(target=self.run, daemon=True)
        self.bot_thread.start()

    def stop(self):
        if not self.running:
            print("⚠ Бот уже остановлен")
            return
        print("🛑 Останавливаю Telegram бота...")
        self.running = False
        
        # Удаляем файл с данными пользователей
        try:
            if os.path.exists(self.user_data_file):
                os.remove(self.user_data_file)
                print(f"🗑️ Файл {self.user_data_file} удален")
        except Exception as e:
            print(f"⚠️ Ошибка при удалении файла {self.user_data_file}: {e}")
        
        # Принудительно завершаем поток
        if self.bot_thread and self.bot_thread.is_alive():
            import ctypes
            import threading
            
            # Пытаемся корректно остановить поток
            self.bot_thread.join(timeout=2)
            
            # Если поток все еще жив, принудительно завершаем его
            if self.bot_thread.is_alive():
                try:
                    # Получаем ID потока
                    thread_id = self.bot_thread.ident
                    if thread_id:
                        # Принудительно завершаем поток
                        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
                            ctypes.c_long(thread_id), ctypes.py_object(SystemExit)
                        )
                        if res == 0:
                            print("Не удалось найти поток")
                        elif res != 1:
                            # Отменяем операцию, если что-то пошло не так
                            ctypes.pythonapi.PyThreadState_SetAsyncExc(
                                ctypes.c_long(thread_id), None
                            )
                            print("Ошибка при принудительном завершении потока")
                        else:
                            print("Поток бота принудительно завершен")
                except Exception as e:
                    print(f"Ошибка при принудительном завершении потока: {e}")
        
        # Обнуляем ссылки
        self.application = None
        self.bot_thread = None
        print("❌ Telegram бот остановлен")

# Пример использования:
if __name__ == "__main__":
    # Получаем новые модели от Polinations для бота
    chat_models = llm.get_Polinations_chat_models()
    img_models = llm.get_Polinations_img_models_for_bot()
    
    # Добавляем новые модели в основной словарь
    if isinstance(img_models, dict):
        model_functions.update(img_models)
    if isinstance(chat_models, dict):
        model_functions.update(chat_models)
    
    # Создаем и запускаем бот
    bot = TelegramBot(model_functions)
    bot.run()
