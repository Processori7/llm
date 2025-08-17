#!/usr/bin/env python3
"""
Модуль FastAPI сервера для чат-приложения.
Содержит API эндпоинты для работы с AI моделями.
"""

import os
import sys
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel


# Создаем экземпляр FastAPI приложения
app = FastAPI()

# Добавляем CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Для тестирования
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class MessageRequest(BaseModel):
    """Модель данных для запроса к /api/gpt/ans"""
    model: str
    message: str


def resource_path(relative_path):
    """Получить абсолютный путь к ресурсам, работает для разработки и PyInstaller"""
    try:
        # PyInstaller создает временную папку и сохраняет путь в _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


def setup_routes(model_functions):
    """Настройка маршрутов FastAPI с переданными функциями моделей"""
    
    @app.get("/api/ai/models", summary="Получить все доступные LLM модели")
    def get_models():
        """Возвращает список доступных моделей для чата"""
        filtered_models = [model for model in model_functions.keys() if
                          "_img" not in model and "(Photo Analyze)" not in model]
        return {"models": filtered_models}

    @app.post("/api/gpt/ans", summary="Получить ответ на текстовый запрос")
    def get_answer(request: MessageRequest):
        """Обработка запроса к AI модели и возврат ответа"""
        model = request.model
        message = request.message

        if model not in model_functions:
            raise HTTPException(status_code=404, detail="Model not found")

        response = model_functions[model](message)
        return {"response": response}

    @app.get("/chat", summary="Открыть страницу чата Web UI")
    def read_chat():
        """Возвращает HTML страницу чата"""
        html_path = resource_path(os.path.join("static", "chat.html"))
        try:
            with open(html_path, "r", encoding="utf-8") as file:
                return HTMLResponse(content=file.read())
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Chat page not found")


def setup_static_files():
    """Настройка статических файлов"""
    # Получаем абсолютный путь к директории проекта
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Путь к статическим файлам
    static_dir = os.path.join(BASE_DIR, "static")
    
    # Проверяем существование директории статических файлов
    if os.path.exists(static_dir):
        # Монтируем статические файлы
        app.mount("/static", StaticFiles(directory=static_dir), name="static")


def get_fastapi_app():
    """Возвращает настроенное FastAPI приложение"""
    setup_static_files()
    return app
