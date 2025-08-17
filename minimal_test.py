#!/usr/bin/env python3
"""
Минимальный тест для проверки функций изображений
"""
import requests

def test_pollinations_api():
    """Тестирование API Pollinations"""
    
    # Тестируем получение моделей
    print("Тестируем получение моделей...")
    try:
        url = "https://image.pollinations.ai/models"
        resp = requests.get(url)
        if resp.ok:
            models = resp.json()
            print(f"✅ Получено {len(models)} моделей")
            print(f"Первые 3 модели: {models[:3]}")
        else:
            print(f"❌ Ошибка получения моделей: {resp.status_code}")
    except Exception as e:
        print(f"❌ Ошибка: {e}")
    
    # Тестируем генерацию изображения
    print("\nТестируем генерацию изображения...")
    try:
        test_prompt = "nature sunset"
        test_model = "flux"
        url = f"https://image.pollinations.ai/prompt/{test_prompt}?model={test_model}"
        print(f"URL для генерации: {url}")
        
        resp = requests.get(url)
        if resp.ok:
            print("✅ Изображение успешно сгенерировано")
            print(f"Размер ответа: {len(resp.content)} байт")
            print(f"Content-Type: {resp.headers.get('Content-Type', 'не определен')}")
        else:
            print(f"❌ Ошибка генерации: {resp.status_code}")
    except Exception as e:
        print(f"❌ Ошибка: {e}")

if __name__ == "__main__":
    test_pollinations_api()
