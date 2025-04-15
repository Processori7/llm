import pytest
from httpx import AsyncClient, ASGITransport
from llm import app



# Интеграционные тесты API
@pytest.mark.asyncio
async def test_api_models():
    # Создаём тестовый клиент с базовым URL
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://localhost:8000") as client:
        response = await client.get("/api/ai/models")
        assert response.status_code == 200

        data = response.json()
        assert "models" in data  # Проверяем, что ключ "models" существует
        models = data["models"]
        assert len(models) > 0  # Проверяем, что список моделей не пустой

@pytest.mark.asyncio
async def test_api_gpt_ans():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        # Подготовка данных для запроса
        request_data = {
            "model": "claude_3_7_sonnet (YouChat)",
            "message": "Hello, how are you?"
        }

        # Отправка POST-запроса
        response = await client.post("/api/gpt/ans", json=request_data)
        assert response.status_code == 200

        # Проверка структуры ответа
        data = response.json()
        assert "response" in data  # Проверяем наличие ключа "response"
        assert isinstance(data["response"], str)  # Проверяем, что ответ — строка