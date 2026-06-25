(function() {
    // Avoid double instantiation
    if (window.SOLIXChatbotPopup) return;

    // Detect locale and direction
    const isRtl = document.documentElement.dir === 'rtl' || document.documentElement.lang === 'ar';
    const dict = {
        title: isRtl ? 'مساعد SOL الذكي' : 'SOL AI Copilot',
        status: isRtl ? 'مهندس البيانات الافتراضي' : 'Virtual Data Engineer',
        placeholder: isRtl ? 'اكتب سؤالك هنا...' : 'Type your question here...',
        welcome: isRtl ? 'أهلاً بك! كيف يمكنني مساعدتك اليوم في منصة SOLIX؟' : 'Hello! How can I help you today with the SOLIX platform?',
        cacheText: isRtl ? 'مسترجع من الذاكرة' : 'From cache',
        connError: isRtl ? 'عذراً، حدث خطأ أثناء الاتصال بالخادم.' : 'Sorry, a connection error occurred.'
    };

    // Inject Styles
    const style = document.createElement('style');
    style.innerHTML = `
        /* Floating Trigger */
        .sol-chat-trigger {
            position: fixed;
            bottom: 24px;
            ${isRtl ? 'left: 24px;' : 'right: 24px;'}
            width: 56px;
            height: 56px;
            border-radius: 50%;
            background: linear-gradient(135deg, var(--primary, #4361ee) 0%, #1d4ed8 100%);
            box-shadow: 0 8px 24px rgba(67, 97, 238, 0.4);
            display: flex;
            align-items: center;
            justify-content: center;
            color: #ffffff;
            cursor: pointer;
            z-index: 9999;
            transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            border: 2px solid rgba(255, 255, 255, 0.1);
        }
        .sol-chat-trigger:hover {
            transform: scale(1.1) rotate(8deg);
            box-shadow: 0 12px 30px rgba(67, 97, 238, 0.55);
        }
        .sol-chat-trigger:active {
            transform: scale(0.95);
        }
        .sol-chat-trigger .material-symbols-outlined {
            font-size: 28px;
            font-variation-settings: 'FILL' 1;
        }

        /* Chat Panel Container */
        .sol-chat-panel {
            position: fixed;
            bottom: 96px;
            ${isRtl ? 'left: 24px;' : 'right: 24px;'}
            width: 380px;
            height: 500px;
            border-radius: 1.25rem;
            background: var(--surface-container-low, #181920);
            border: 1px solid var(--outline-variant, #2a2c38);
            box-shadow: 0 12px 40px rgba(0, 0, 0, 0.25);
            display: flex;
            flex-direction: column;
            z-index: 9998;
            overflow: hidden;
            opacity: 0;
            transform: translateY(20px) scale(0.95);
            pointer-events: none;
            transition: all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1);
            backdrop-filter: blur(20px);
        }
        .sol-chat-panel.open {
            opacity: 1;
            transform: translateY(0) scale(1);
            pointer-events: auto;
        }

        @media (max-width: 480px) {
            .sol-chat-panel {
                width: calc(100% - 32px);
                height: 70vh;
                bottom: 90px;
                right: 16px;
                left: 16px;
            }
        }

        /* Header */
        .sol-chat-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 12px 16px;
            border-bottom: 1px solid var(--outline-variant, #2a2c38);
            background: rgba(0, 0, 0, 0.1);
        }
        .sol-chat-header-info {
            display: flex;
            align-items: center;
            gap: 10px;
            direction: ${isRtl ? 'rtl' : 'ltr'};
        }
        .sol-chat-avatar {
            width: 32px;
            height: 32px;
            border-radius: 50%;
            background: linear-gradient(135deg, var(--primary, #4361ee) 0%, #a855f7 100%);
            display: flex;
            align-items: center;
            justify-content: center;
            color: #fff;
        }
        .sol-chat-avatar .material-symbols-outlined {
            font-size: 18px;
        }
        .sol-chat-title {
            font-size: 13.5px;
            font-weight: bold;
            color: var(--on-surface, #fff);
            line-height: 1.2;
        }
        .sol-chat-status {
            font-size: 10px;
            color: var(--outline, #94a3b8);
        }

        .sol-chat-header-actions {
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .sol-chat-action-btn {
            width: 28px;
            height: 28px;
            border-radius: 6px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: var(--outline, #94a3b8);
            cursor: pointer;
            transition: all 0.2s ease;
            border: none;
            background: transparent;
        }
        .sol-chat-action-btn:hover {
            background: var(--surface-container-highest, #2e303f);
            color: var(--on-surface, #fff);
        }

        /* Messages Body */
        .sol-chat-body {
            flex: 1;
            overflow-y: auto;
            padding: 16px;
            display: flex;
            flex-direction: column;
            gap: 12px;
            background: rgba(0,0,0,0.02);
        }
        .sol-chat-body::-webkit-scrollbar {
            width: 4px;
        }
        .sol-chat-body::-webkit-scrollbar-thumb {
            background: var(--outline-variant, #2a2c38);
            border-radius: 10px;
        }

        /* Message Bubbles */
        .sol-bubble-wrap {
            display: flex;
            width: 100%;
            direction: ${isRtl ? 'rtl' : 'ltr'};
        }
        .sol-bubble-wrap.user {
            justify-content: flex-end;
        }
        .sol-bubble-wrap.bot {
            justify-content: flex-start;
        }
        .sol-bubble-box {
            max-width: 80%;
            display: flex;
            flex-direction: column;
            gap: 4px;
        }
        .sol-bubble {
            padding: 8px 12px;
            border-radius: 14px;
            font-size: 13px;
            line-height: 1.5;
            word-wrap: break-word;
            box-shadow: 0 2px 8px rgba(0,0,0,0.03);
            text-align: ${isRtl ? 'right' : 'left'};
        }
        .sol-bubble-wrap.user .sol-bubble {
            background: linear-gradient(135deg, var(--primary, #4361ee) 0%, #1d4ed8 100%);
            color: #ffffff;
            border-bottom-right-radius: 3px;
        }
        .sol-bubble-wrap.bot .sol-bubble {
            background: var(--surface-container-high, #22232c);
            color: var(--on-surface, #fff);
            border: 1px solid var(--outline-variant, #2a2c38);
            border-bottom-left-radius: 3px;
        }

        .sol-bubble-time {
            font-size: 9px;
            color: var(--outline, #94a3b8);
            align-self: ${isRtl ? 'flex-start' : 'flex-end'};
            margin: 0 4px;
        }
        .sol-bubble-wrap.user .sol-bubble-time {
            align-self: flex-end;
        }
        .sol-bubble-wrap.bot .sol-bubble-time {
            align-self: flex-start;
        }

        .sol-cache-tag {
            background: rgba(16, 185, 129, 0.08);
            border: 1px solid rgba(16, 185, 129, 0.2);
            color: #10b981;
            font-size: 9px;
            padding: 1px 6px;
            border-radius: 10px;
            align-self: flex-start;
            margin-top: 2px;
            display: inline-flex;
            align-items: center;
            gap: 2px;
        }

        /* Typing Dot Pulse */
        .sol-typing-indicator {
            display: flex;
            gap: 3px;
            align-items: center;
            padding: 4px 6px;
        }
        .sol-dot {
            width: 5px;
            height: 5px;
            background: var(--outline, #94a3b8);
            border-radius: 50%;
            animation: sol-dot-pulse 1.4s infinite ease-in-out;
        }
        .sol-dot:nth-child(1) { animation-delay: -0.32s; }
        .sol-dot:nth-child(2) { animation-delay: -0.16s; }

        @keyframes sol-dot-pulse {
            0%, 80%, 100% { transform: scale(0.6); opacity: 0.5; }
            40% { transform: scale(1.2); opacity: 1; }
        }

        /* Input Footer */
        .sol-chat-footer {
            padding: 12px;
            border-top: 1px solid var(--outline-variant, #2a2c38);
            background: rgba(0, 0, 0, 0.05);
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .sol-input-field {
            flex: 1;
            background: var(--surface-container-highest, #2a2c38);
            border: 1px solid var(--outline-variant, #2a2c38);
            border-radius: 10px;
            color: var(--on-surface, #fff);
            padding: 8px 12px;
            font-size: 13px;
            outline: none;
            direction: ${isRtl ? 'rtl' : 'ltr'};
            transition: all 0.2s ease;
        }
        .sol-input-field:focus {
            border-color: var(--primary, #4361ee);
        }
        .sol-send-btn {
            width: 32px;
            height: 32px;
            border-radius: 50%;
            background: linear-gradient(135deg, var(--primary, #4361ee) 0%, #1d4ed8 100%);
            color: #ffffff;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            border: none;
            transition: all 0.2s ease;
        }
        .sol-send-btn:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 10px rgba(67, 97, 238, 0.35);
        }
        .sol-send-btn:active {
            transform: scale(0.95);
        }
        .sol-send-btn .material-symbols-outlined {
            font-size: 16px;
        }
    `;
    document.head.appendChild(style);

    // Inject Audio Assets
    const soundSend = document.createElement('audio');
    soundSend.id = 'sol-pop-sound-send';
    soundSend.src = 'https://assets.mixkit.co/active_storage/sfx/2568/2568-84.wav';
    soundSend.preload = 'auto';
    document.body.appendChild(soundSend);

    const soundReceive = document.createElement('audio');
    soundReceive.id = 'sol-pop-sound-receive';
    soundReceive.src = 'https://assets.mixkit.co/active_storage/sfx/1435/1435-84.wav';
    soundReceive.preload = 'auto';
    document.body.appendChild(soundReceive);

    // Inject HTML Layout
    const popupContainer = document.createElement('div');
    popupContainer.innerHTML = `
        <div class="sol-chat-trigger" id="solChatTriggerBtn" title="${dict.title}">
            <span class="material-symbols-outlined">chat</span>
        </div>
        <div class="sol-chat-panel" id="solChatPanel">
            <div class="sol-chat-header">
                <div class="sol-chat-header-info">
                    <div class="sol-chat-avatar">
                        <span class="material-symbols-outlined">smart_toy</span>
                    </div>
                    <div>
                        <div class="sol-chat-title">${dict.title}</div>
                        <div class="sol-chat-status">${dict.status}</div>
                    </div>
                </div>
                <div class="sol-chat-header-actions">
                    <button class="sol-chat-action-btn" id="solChatClearBtn" title="Clear">
                        <span class="material-symbols-outlined" style="font-size: 18px;">delete</span>
                    </button>
                    <button class="sol-chat-action-btn" id="solChatCloseBtn" title="Close">
                        <span class="material-symbols-outlined" style="font-size: 18px;">close</span>
                    </button>
                </div>
            </div>
            <div class="sol-chat-body" id="solChatBody">
                <div class="sol-bubble-wrap bot">
                    <div class="sol-bubble-box">
                        <div class="sol-bubble">${dict.welcome}</div>
                        <div class="sol-bubble-time">${new Date().getHours()}:${new Date().getMinutes().toString().padStart(2, '0')}</div>
                    </div>
                </div>
            </div>
            <div class="sol-chat-footer">
                <input type="text" class="sol-input-field" id="solChatInputField" placeholder="${dict.placeholder}">
                <button class="sol-send-btn" id="solChatSendBtn">
                    <span class="material-symbols-outlined">send</span>
                </button>
            </div>
        </div>
    `;
    document.body.appendChild(popupContainer);

    // DOM Elements
    const triggerBtn = document.getElementById('solChatTriggerBtn');
    const chatPanel = document.getElementById('solChatPanel');
    const closeBtn = document.getElementById('solChatCloseBtn');
    const clearBtn = document.getElementById('solChatClearBtn');
    const inputField = document.getElementById('solChatInputField');
    const sendBtn = document.getElementById('solChatSendBtn');
    const chatBody = document.getElementById('solChatBody');

    let history = [];
    let isOpen = false;

    function playSfx(id) {
        if (localStorage.getItem('sol_chat_sound') !== 'disabled') {
            const aud = document.getElementById(id);
            if (aud) {
                aud.currentTime = 0;
                aud.play().catch(e => console.log('Sfx error:', e));
            }
        }
    }

    function togglePanel() {
        isOpen = !isOpen;
        if (isOpen) {
            chatPanel.classList.add('open');
            triggerBtn.style.transform = 'scale(0) rotate(180deg)';
            inputField.focus();
        } else {
            chatPanel.classList.remove('open');
            triggerBtn.style.transform = 'scale(1) rotate(0deg)';
        }
    }

    triggerBtn.addEventListener('click', () => {
        playSfx('sol-pop-sound-send');
        togglePanel();
    });

    closeBtn.addEventListener('click', () => {
        playSfx('sol-pop-sound-send');
        togglePanel();
    });

    clearBtn.addEventListener('click', () => {
        playSfx('sol-pop-sound-send');
        chatBody.innerHTML = `
            <div class="sol-bubble-wrap bot">
                <div class="sol-bubble-box">
                    <div class="sol-bubble">${dict.welcome}</div>
                    <div class="sol-bubble-time">${new Date().getHours()}:${new Date().getMinutes().toString().padStart(2, '0')}</div>
                </div>
            </div>
        `;
        history = [];
    });

    function formatMarkdown(text) {
        return text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                   .replace(/\n/g, '<br>');
    }

    function appendMessage(sender, text, isCached = false) {
        const wrap = document.createElement('div');
        wrap.className = `sol-bubble-wrap ${sender}`;

        const box = document.createElement('div');
        box.className = 'sol-bubble-box';

        const bubble = document.createElement('div');
        bubble.className = 'sol-bubble';
        bubble.innerHTML = formatMarkdown(text);
        box.appendChild(bubble);

        if (isCached && sender === 'bot') {
            const cacheTag = document.createElement('div');
            cacheTag.className = 'sol-cache-tag';
            cacheTag.innerHTML = `<span class="material-symbols-outlined" style="font-size: 8px;">bolt</span> ${dict.cacheText}`;
            box.appendChild(cacheTag);
        }

        const time = document.createElement('div');
        time.className = 'sol-bubble-time';
        const now = new Date();
        time.innerText = `${now.getHours()}:${now.getMinutes().toString().padStart(2, '0')}`;
        box.appendChild(time);

        wrap.appendChild(box);
        chatBody.appendChild(wrap);
        chatBody.scrollTop = chatBody.scrollHeight;
    }

    function appendTyping() {
        const wrap = document.createElement('div');
        wrap.className = 'sol-bubble-wrap bot';
        wrap.id = 'solChatTypingRow';

        const box = document.createElement('div');
        box.className = 'sol-bubble-box';

        const bubble = document.createElement('div');
        bubble.className = 'sol-bubble';
        bubble.innerHTML = `
            <div class="sol-typing-indicator">
                <div class="sol-dot"></div>
                <div class="sol-dot"></div>
                <div class="sol-dot"></div>
            </div>
        `;

        box.appendChild(bubble);
        wrap.appendChild(box);
        chatBody.appendChild(wrap);
        chatBody.scrollTop = chatBody.scrollHeight;
        return wrap;
    }

    async function sendMsg() {
        const message = inputField.value.trim();
        if (!message) return;

        playSfx('sol-pop-sound-send');
        appendMessage('user', message);
        inputField.value = '';

        inputField.disabled = true;
        sendBtn.disabled = true;

        const typingRow = appendTyping();
        const payloadHistory = [...history];
        history.push({ role: 'user', content: message });

        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: message, history: payloadHistory })
            });

            typingRow.remove();

            if (response.ok) {
                const data = await response.json();
                appendMessage('bot', data.response, data.cached);
                history.push({ role: 'bot', content: data.response });
                playSfx('sol-pop-sound-receive');
            } else {
                appendMessage('bot', dict.connError);
            }
        } catch (e) {
            if (typingRow) typingRow.remove();
            appendMessage('bot', dict.connError);
        } finally {
            inputField.disabled = false;
            sendBtn.disabled = false;
            inputField.focus();
        }
    }

    sendBtn.addEventListener('click', sendMsg);
    inputField.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            sendMsg();
        }
    });

    window.SOLIXChatbotPopup = {
        toggle: togglePanel,
        open: () => { if (!isOpen) togglePanel(); },
        close: () => { if (isOpen) togglePanel(); }
    };
})();
