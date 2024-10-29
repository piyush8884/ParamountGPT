document.addEventListener('DOMContentLoaded', () => {
    const chatSection = document.getElementById('chatSection');
    const modeTitle = document.getElementById('modeTitle');
    const csvUpload = document.getElementById('csvUpload');
    const chatWithDataBtn = document.getElementById('chatWithDataBtn');
    const graspInfoBtn = document.getElementById('graspInfoBtn');
    const queryInput = document.getElementById('queryInput');
    const historyList = document.getElementById('historyList');
    const clearBtn = document.getElementById('clearBtn');

    chatWithDataBtn.addEventListener('click', () => {
        modeTitle.textContent = 'Chat with Data';
        csvUpload.classList.remove('hidden');
        chatSection.classList.remove('hidden');
    });

    graspInfoBtn.addEventListener('click', () => {
        modeTitle.textContent = 'Grasp Information';
        csvUpload.classList.add('hidden');
        chatSection.classList.remove('hidden');
    });

    clearBtn.addEventListener('click', () => {
        historyList.innerHTML = '';
    });
});
