#!/usr/bin/env python3
"""
Расширенный тест для проверки функций генерации изображений
"""
import requests
import urllib.parse

def test_url_encoding():
    """Тестирование кодирования URL"""
    print("Тестируем кодирование URL...")
    
    # Тестируем разные промпты
    test_prompts = [
        "nature sunset",
        "природа закат",
        "a beautiful landscape with mountains",
        "красивый пейзаж с горами"
    ]
    
    for prompt in test_prompts:
        encoded = urllib.parse.quote(prompt)
        print(f"'{prompt}' -> '{encoded}'")
        
        # Проверяем, что URL работает
        url = f"https://image.pollinations.ai/prompt/{encoded}?model=flux"
        try:
            resp = requests.get(url, timeout=10)
            if resp.ok:
                print(f"✅ URL работает, размер: {len(resp.content)} байт")
            else:
                print(f"❌ URL не работает: {resp.status_code}")
        except Exception as e:
            print(f"❌ Ошибка запроса: {e}")
        print()

def test_gen_img_for_bot():
    """Имитация функции gen_img_for_bot"""
    def gen_img_for_bot(user_input, model):
        """Функция для генерации изображения для бота"""
        try:
            # Кодируем промпт для URL
            encoded_prompt = urllib.parse.quote(user_input)
            
            # Формируем URL для запроса
            url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?model={model}"
            resp = requests.get(url)

            if resp.ok:
                # Для бота возвращаем URL изображения
                return url
            else:
                return f"Ошибка генерации изображения: {resp.status_code}"
        except Exception as e:
            return f"Ошибка: {str(e)}"
    
    print("Тестируем функцию gen_img_for_bot...")
    
    # Тестируем с разными промптами
    test_cases = [
        ("nature sunset", "flux"),
        ("природа закат", "flux"),
        ("beautiful landscape", "turbo"),
        ("красивый пейзаж", "kontext")
    ]
    
    for prompt, model in test_cases:
        print(f"\nТестируем: '{prompt}' с моделью '{model}'")
        result = gen_img_for_bot(prompt, model)
        print(f"Результат: {result}")
        
        if result.startswith("https://image.pollinations.ai/"):
            print("✅ Функция работает корректно!")
        else:
            print("❌ Функция не работает")

def test_bot_image_detection():
    """Тестирование детекции изображений в боте"""
    print("\nТестируем детекцию изображений в боте...")
    
    test_responses = [
        "https://image.pollinations.ai/prompt/nature%20sunset?model=flux",
        "https://example.com/image.jpg",
        "https://example.com/image.png",
        "https://example.com/document.pdf",
        "Это просто текстовый ответ",
        "Ошибка генерации изображения: 404"
    ]
    
    for response in test_responses:
        # Проверяем, является ли ответ URL изображения
        is_image = (response.startswith(("http://", "https://")) and 
                   ("image.pollinations.ai" in response or 
                    any(ext in response for ext in [".jpg", ".png", ".jpeg", ".gif", ".webp"])))
        
        print(f"'{response}' -> {'✅ Изображение' if is_image else '❌ Не изображение'}")

if __name__ == "__main__":
    test_url_encoding()
    test_gen_img_for_bot()
    test_bot_image_detection()
