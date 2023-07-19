document.addEventListener("DOMContentLoaded", function() {
    const messageInput = document.getElementById('message-input');
    const sendButton = document.getElementById('send-button');
    const messagesContainer = document.querySelector('.messages');
  
    sendButton.addEventListener('click', sendMessage);
    messageInput.addEventListener('keydown', function(e) {
      if (e.key === 'Enter') {
        sendMessage();
      }
    });
  
    function sendMessage() {
      const message = messageInput.value.trim();
      if (message !== '') {
        displayMessage('User', message);
        sendRequest(message);
        messageInput.value = '';
      }
    }
  
    function sendRequest(message) {
      const formData = new FormData();
      formData.append('user_input', message);
  
      fetch('/chat', {
        method: 'POST',
        body: formData
      })
        .then(response => response.json())
        .then(data => {
          const intent = data.intent;
          const response = data.response;
          displayMessage('AI Bot', response);
        })
        .catch(error => {
          console.error('Error:', error);
        });
    }
  
    function displayMessage(sender, text) {
      const messageElement = document.createElement('div');
      messageElement.classList.add('message');
      messageElement.innerHTML = `<span class="sender">${sender}: </span>${text}`;
      messagesContainer.appendChild(messageElement);
    
      // Scroll to the bottom of the messages container
      messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }
    
  });
  