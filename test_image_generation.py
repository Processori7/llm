#!/usr/bin/env python3
"""
Тестовый файл для проверки функций генерации изображений
"""
import llm

def test_image_generation():
    """Тестирование функций генерации изображений"""
    
    # Тестируем функцию для бота
    print("Тестируем функцию gen_img_for_bot...")
    result_bot = llm.gen_img_for_bot("природа закат", "flux")
    print(f"Результат для бота: {result_bot}")
    
    # Проверяем, что функция возвращает URL
    if result_bot.startswith("https://image.pollinations.ai/"):
        print("✅ Функция gen_img_for_bot работает корректно!")
        print("✅ Возвращается URL изображения для бота")
    else:
        print("❌ Проблема с функцией gen_img_for_bot")
    
    # Тестируем получение моделей для бота
    print("\nТестируем функцию get_Polinations_img_models_for_bot...")
    models_for_bot = llm.get_Polinations_img_models_for_bot()
    
    if isinstance(models_for_bot, dict) and len(models_for_bot) > 0:
        print(f"✅ Получено {len(models_for_bot)} моделей для бота")
        # Выводим первые 3 модели
        for i, model_name in enumerate(list(models_for_bot.keys())[:3]):
            print(f"  {i+1}. {model_name}")
        
        # Тестируем одну модель
        first_model = list(models_for_bot.keys())[0]
        print(f"\nТестируем модель: {first_model}")
        test_func = models_for_bot[first_model]
        test_result = test_func("nature sunset")
        print(f"Результат: {test_result}")
        
        if test_result.startswith("https://image.pollinations.ai/"):
            print("✅ Модель работает корректно!")
        else:
            print("❌ Модель не работает корректно")
    else:
        print("❌ Проблема с получением моделей")

if __name__ == "__main__":
    test_image_generation()
