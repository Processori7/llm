document.addEventListener('DOMContentLoaded', () => {
    const chatBox = document.getElementById('chat-box');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');
    const saveChatButton = document.getElementById('save-chat-button');
    const clearChatButton = document.getElementById('clear-chat-button');
    const modelSelect = document.getElementById('model-select');
    const swaggerButton = document.getElementById('swagger-button');
    const langButton = document.getElementById('lang-button');

    let chatHistory = [];
    let isRussian = true;

    // Загрузка моделей
    fetch('/api/ai/models')
        .then(response => response.json())
        .then(data => {
            data.models.forEach(model => {
                const option = document.createElement('option');
                option.value = model;
                option.textContent = model;
                modelSelect.appendChild(option);
            });
        })
        .catch(error => console.error('Ошибка при загрузке моделей:', error));

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

        chatHistory.push({ sender: 'user', message: message });
        appendMessage('user', message);

        const selectedModel = modelSelect.value;

        fetch('/api/gpt/ans', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ model: selectedModel, message: message })
        })
        .then(response => response.json())
        .then(data => {
            chatHistory.push({ sender: 'bot', message: data.response });
            appendMessage('bot', data.response);
        })
        .catch(error => {
            chatHistory.push({ sender: 'bot', message: 'Произошла ошибка при получении ответа.' });
            appendMessage('bot', 'Произошла ошибка при получении ответа.');
            console.error('Ошибка при отправке сообщения:', error);
        });

        userInput.value = '';
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
            sendButton.textContent = 'Отправить';
            saveChatButton.textContent = 'Сохранить чат';
            clearChatButton.textContent = 'Очистить чат';
            langButton.textContent = 'English';
        } else {
            document.querySelector('.model-selector label').textContent = 'Model Selection:';
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