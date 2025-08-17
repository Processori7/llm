document.addEventListener('DOMContentLoaded', () => {
    const chatBox = document.getElementById('chat-box');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');
    const saveChatButton = document.getElementById('save-chat-button');
    const clearChatButton = document.getElementById('clear-chat-button');
    const modelSelect = document.getElementById('model-select');
    const modelSearch = document.getElementById('model-search'); // Поле поиска
    const swaggerButton = document.getElementById('swagger-button');
    const langButton = document.getElementById('lang-button');
    let chatHistory = [];
    let isRussian = true;
    let allModels = []; // Хранение всех моделей

    // Загружаем сохраненную историю чата при загрузке страницы
    function loadChatHistory() {
        const savedHistory = localStorage.getItem('chatHistory');
        if (savedHistory) {
            try {
                chatHistory = JSON.parse(savedHistory);
                // Восстанавливаем сообщения в чате
                chatHistory.forEach(msg => {
                    appendMessage(msg.sender, msg.message);
                });
            } catch (error) {
                console.error('Ошибка при загрузке истории чата:', error);
                chatHistory = [];
            }
        }
    }

    // Сохраняем историю чата в localStorage
    function saveChatHistory() {
        try {
            localStorage.setItem('chatHistory', JSON.stringify(chatHistory));
        } catch (error) {
            console.error('Ошибка при сохранении истории чата:', error);
        }
    }

    // Загружаем сохраненную историю чата
    loadChatHistory();

    // Загрузка моделей
    fetch('/api/ai/models')
        .then(response => response.json())
        .then(data => {
            allModels = data.models; // Сохраняем все модели
            renderModels(allModels); // Отображаем все модели
        })
        .catch(error => console.error('Ошибка при загрузке моделей:', error));

    // Функция отображения моделей
    function renderModels(models) {
        modelSelect.innerHTML = ''; // Очищаем выпадающий список
        models.forEach(model => {
            const option = document.createElement('option');
            option.value = model;
            option.textContent = model;
            modelSelect.appendChild(option);
        });
    }

    // Поиск моделей
    modelSearch.addEventListener('input', () => {
        const query = modelSearch.value.toLowerCase(); // Получаем поисковый запрос
        const filteredModels = allModels.filter(model =>
            model.toLowerCase().includes(query)
        );
        renderModels(filteredModels); // Отображаем отфильтрованные модели
    });

    // Отправка сообщения
    sendButton.addEventListener('click', sendMessage);
    userInput.addEventListener('keydown', (event) => {
        if (event.key === 'Enter') {
            event.preventDefault();
            sendMessage();
        }
    });

    function sendMessage() {
        const message = userInput.value.trim();
        if (message === '') return;
        
        // Добавляем сообщение пользователя
        chatHistory.push({ sender: 'user', message: message });
        appendMessage('user', message);
        saveChatHistory(); // Сохраняем историю
        
        // Очищаем поле ввода сразу
        userInput.value = '';
        
        // Показываем сообщение "Думаю..." 
        const thinkingText = isRussian ? 'Думаю...' : 'Thinking...';
        const thinkingElement = document.createElement('div');
        thinkingElement.classList.add('message', 'bot', 'thinking');
        thinkingElement.id = 'thinking-message'; // Добавляем ID для легкого поиска
        thinkingElement.textContent = `${isRussian ? 'Бот:' : 'Bot:'} ${thinkingText}`;
        
        // Добавляем элемент в чат
        chatBox.appendChild(thinkingElement);
        chatBox.scrollTop = chatBox.scrollHeight;
        
        const selectedModel = modelSelect.value;
        
        // Отправляем запрос к API
        fetch('/api/gpt/ans', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ model: selectedModel, message: message })
        })
        .then(response => response.json())
        .then(data => {
            
            // Минимальная задержка, чтобы пользователь увидел сообщение "Думаю..."
            setTimeout(() => {
                // Удаляем сообщение "Думаю..."
                const thinkingMsg = document.getElementById('thinking-message');
                if (thinkingMsg) {
                    thinkingMsg.remove();
                }
                
                // Добавляем ответ бота
                chatHistory.push({ sender: 'bot', message: data.response });
                appendMessage('bot', data.response);
                saveChatHistory(); // Сохраняем историю
            }, 500); // Задержка 0.5 секунды
        })
        .catch(error => {
            // Удаляем сообщение "Думаю..." при ошибке
            const thinkingMsg = document.getElementById('thinking-message');
            if (thinkingMsg) {
                thinkingMsg.remove();
            }
            
            // Показываем сообщение об ошибке
            const errorMessage = isRussian ? 'Произошла ошибка при получении ответа.' : 'An error occurred while getting the response.';
            chatHistory.push({ sender: 'bot', message: errorMessage });
            appendMessage('bot', errorMessage);
            saveChatHistory(); // Сохраняем историю
            console.error('Ошибка при отправке сообщения:', error);
        });
    }

    // Сохранение чата
    saveChatButton.addEventListener('click', () => {
        const chatText = chatHistory.map(msg => `${msg.sender}: ${msg.message}`).join('\n');
        const blob = new Blob([chatText], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'chat.txt';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    });

    // Очистка чата
    clearChatButton.addEventListener('click', () => {
        chatBox.innerHTML = '';
        chatHistory = [];
        // Очищаем сохраненную историю
        localStorage.removeItem('chatHistory');
    });

    // Открытие Swagger UI
    swaggerButton.addEventListener('click', () => {
        const swaggerUrl = `http://${window.location.hostname}:8000/docs`;
        window.open(swaggerUrl, '_blank');
    });

    // Перевод интерфейса
    langButton.addEventListener('click', () => {
        isRussian = !isRussian;
        updateLanguage();
    });

    function updateLanguage() {
        if (isRussian) {
            document.querySelector('.model-selector label').textContent = 'Выбор моделей:';
            modelSearch.placeholder = 'Поиск...';
            sendButton.textContent = 'Отправить';
            saveChatButton.textContent = 'Сохранить чат';
            clearChatButton.textContent = 'Очистить чат';
            langButton.textContent = 'English';
        } else {
            document.querySelector('.model-selector label').textContent = 'Model Selection:';
            modelSearch.placeholder = 'Search...';
            sendButton.textContent = 'Send';
            saveChatButton.textContent = 'Save Chat';
            clearChatButton.textContent = 'Clear Chat';
            langButton.textContent = 'Русский';
        }
    }

    // Добавление сообщения в чат
    function appendMessage(sender, message) {
        const messageElement = document.createElement('div');
        messageElement.classList.add('message', sender);
        messageElement.textContent = `${sender === 'user' ? (isRussian ? 'Вы:' : 'You:') : (isRussian ? 'Бот:' : 'Bot:')} ${message}`;
        chatBox.appendChild(messageElement);
        chatBox.scrollTop = chatBox.scrollHeight;
    }
});