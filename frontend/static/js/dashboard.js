/* dashboard.js — Frontend Controller for Stock Price Prediction Dashboard */

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const stockInput = document.getElementById('stock-input');
    const stocksDatalist = document.getElementById('stocks-datalist');
    const predictBtn = document.getElementById('predict-btn');
    const refreshBtn = document.getElementById('refresh-btn');
    const currentDateBadge = document.getElementById('current-date-badge');

    // Sidebar Nav Items
    const navDashboard = document.getElementById('nav-dashboard');
    const navPrediction = document.getElementById('nav-prediction');
    const navPortfolio = document.getElementById('nav-portfolio');
    const navBacktest = document.getElementById('nav-backtest');
    const navAlerts = document.getElementById('nav-alerts');
    const navComparison = document.getElementById('nav-comparison');
    const navHistorical = document.getElementById('nav-historical');

    // Page Views
    const viewDashboard = document.getElementById('view-dashboard');
    const viewPrediction = document.getElementById('view-prediction');
    const viewPortfolio = document.getElementById('view-portfolio');
    const viewBacktest = document.getElementById('view-backtest');
    const viewAlerts = document.getElementById('view-alerts');
    const viewComparison = document.getElementById('view-comparison');
    const viewHistorical = document.getElementById('view-historical');

    const pageTitle = document.getElementById('page-title');
    const pageSubtitle = document.getElementById('page-subtitle');

    // Portfolio Elements
    const refreshPortfolioBtn = document.getElementById('refresh-portfolio-btn');

    // Alerts Elements
    const addAlertBtn = document.getElementById('add-alert-btn');
    const alertTickerInput = document.getElementById('alert-ticker-input');
    const alertTypeSelect = document.getElementById('alert-type-select');
    const alertTargetInput = document.getElementById('alert-target-input');
    const alertNotesInput = document.getElementById('alert-notes-input');

    // Backtest Elements
    const runBacktestBtn = document.getElementById('run-backtest-btn');
    const stratSelector = document.getElementById('strat-selector');
    let backtestChart = null;

    // Table Bodies
    const predTableBody = document.getElementById('prediction-table-body');
    const histTableBody = document.getElementById('historical-table-body');
    const backtestTradeBody = document.getElementById('backtest-trade-body');
    const predViewTitle = document.getElementById('pred-view-title');
    const histViewTitle = document.getElementById('hist-view-title');
    const backtestViewTitle = document.getElementById('backtest-view-title');

    // Cards Elements
    const currentPriceEl = document.getElementById('current-price');
    const currentDateSubEl = document.getElementById('current-date-sub');
    const predictedPriceEl = document.getElementById('predicted-price');
    const predictedDeltaSubEl = document.getElementById('predicted-delta-sub');
    const expectedReturnEl = document.getElementById('expected-return');
    const predictionSignalEl = document.getElementById('prediction-signal');

    // Info Section Elements
    const infoStockEl = document.getElementById('info-stock');
    const infoDateEl = document.getElementById('info-date');
    const infoModelEl = document.getElementById('info-model');

    // Timeframe Buttons
    const tfButtons = document.querySelectorAll('.tf-btn');

    // Global State
    let mainChart = null;
    let currentTicker = (stockInput && stockInput.value.trim()) ? stockInput.value.trim().toUpperCase() : '';
    let activeTimeframe = '1M';
    let rawHistoricalData = [];
    let currentPredictionData = null;

    // 1. Initialize Date
    const today = new Date();
    const formattedDate = today.toLocaleDateString('en-US', {
        month: 'short',
        day: 'numeric',
        year: 'numeric'
    });
    const isoDate = today.toISOString().split('T')[0];
    currentDateBadge.innerHTML = `📅 ${formattedDate}`;

    // User Authentication Session State Manager
    function getCurrentUserId() {
        const userInfoRaw = localStorage.getItem('user_info');
        if (userInfoRaw) {
            try {
                const u = JSON.parse(userInfoRaw);
                if (u && u.id !== undefined && u.id !== null) return String(u.id);
                if (u && u.email) return String(u.email).toLowerCase().trim();
            } catch (e) {}
        }
        return 'guest_user';
    }

    checkUserAuthSession();

    function checkUserAuthSession() {
        const userCardContainer = document.getElementById('sidebar-user-card');
        if (!userCardContainer) return;

        const token = localStorage.getItem('jwt_token');
        const userInfoRaw = localStorage.getItem('user_info');

        if (token && userInfoRaw) {
            try {
                const user = JSON.parse(userInfoRaw);
                const displayName = user.username || (user.email ? user.email.split('@')[0] : 'User');
                const initial = displayName.charAt(0).toUpperCase();

                userCardContainer.innerHTML = `
                    <div class="sidebar-user-logged-in">
                        <div class="user-profile-info">
                            <div class="user-avatar-circle">${initial}</div>
                            <div class="user-text-details">
                                <span class="user-display-name">${displayName}</span>
                                <span class="user-plan-badge">Pro Account</span>
                            </div>
                        </div>
                        <button id="settings-btn" class="btn-chatgpt-settings" title="Account Settings & Preferences">
                            ⚙️
                        </button>
                    </div>
                `;

                document.getElementById('settings-btn').addEventListener('click', () => {
                    openUserSettingsModal(user);
                });
                return;
            } catch (e) {
                console.error('Error parsing user_info session:', e);
            }
        }


        userCardContainer.innerHTML = `
            <div class="auth-box-chatgpt">
                <div class="auth-box-title">Get Started</div>
                <div class="auth-box-subtitle">Sign in to save portfolios & alerts</div>
                <div class="auth-box-buttons">
                    <a href="/login" class="btn-chatgpt-login">Log in</a>
                    <a href="/register" class="btn-chatgpt-signup">Sign up</a>
                </div>
            </div>
        `;
    }

    function openUserSettingsModal(user) {
        let modal = document.getElementById('user-settings-modal');
        if (!modal) {
            modal = document.createElement('div');
            modal.id = 'user-settings-modal';
            modal.className = 'settings-modal-overlay';
            document.body.appendChild(modal);
        }

        const username = user ? (user.username || user.email.split('@')[0]) : 'User';
        const email = user ? (user.email || 'user@example.com') : 'user@example.com';
        const initial = username.charAt(0).toUpperCase();

        modal.innerHTML = `
            <div class="settings-modal-card">
                <div class="settings-modal-header">
                    <h3>⚙️ Account & Platform Settings</h3>
                    <button class="settings-modal-close" id="close-settings-modal-btn">✕</button>
                </div>
                <div class="settings-modal-body">
                    <div class="settings-user-info-box">
                        <div class="settings-avatar-circle">${initial}</div>
                        <div>
                            <div class="settings-user-meta-name">${username}</div>
                            <div class="settings-user-meta-email">${email}</div>
                        </div>
                    </div>

                    <div class="settings-form-group">
                        <label>Default Stock Ticker</label>
                        <input type="text" id="settings-default-ticker" class="stock-input" style="width: 100%;" value="${localStorage.getItem('pref_default_ticker') || 'RELIANCE.NS'}">
                    </div>

                    <div class="settings-form-group">
                        <label>Preferred Notification Channel</label>
                        <select id="settings-pref-channel" class="select-sm" style="width: 100%; padding: 8px;">
                            <option value="EMAIL" selected>EMAIL ONLY (Gmail SMTP)</option>
                        </select>
                    </div>

                    <div class="settings-form-group">
                        <label>Auto Refresh Interval</label>
                        <select id="settings-refresh-interval" class="select-sm" style="width: 100%; padding: 8px;">
                            <option value="10000" selected>10 Seconds (Default)</option>
                            <option value="30000">30 Seconds</option>
                            <option value="60000">60 Seconds</option>
                        </select>
                    </div>
                </div>
                <div class="settings-modal-footer">
                    <div style="display: flex; gap: 8px;">
                        <button id="modal-logout-btn" class="btn btn-secondary" style="color: #697586; border-color: #253140;">
                            🚪 Sign Out
                        </button>
                        <button id="modal-delete-account-btn" class="btn btn-secondary" style="color: #EF4444; border-color: rgba(239,68,68,0.4); background: rgba(239,68,68,0.1);">
                            🗑️ Delete Account
                        </button>
                    </div>
                    <button id="save-settings-btn" class="btn btn-primary">
                        💾 Save Preferences
                    </button>
                </div>
            </div>
        `;

        modal.style.display = 'flex';

        document.getElementById('close-settings-modal-btn').addEventListener('click', () => {
            modal.style.display = 'none';
        });

        modal.addEventListener('click', (e) => {
            if (e.target === modal) modal.style.display = 'none';
        });

        document.getElementById('save-settings-btn').addEventListener('click', () => {
            const defTicker = document.getElementById('settings-default-ticker').value.trim().toUpperCase();
            const channel = document.getElementById('settings-pref-channel').value;
            if (defTicker) localStorage.setItem('pref_default_ticker', defTicker);
            localStorage.setItem('pref_channel', channel);
            alert('Settings preferences saved successfully!');
            modal.style.display = 'none';
        });

        document.getElementById('modal-logout-btn').addEventListener('click', () => {
            localStorage.removeItem('jwt_token');
            localStorage.removeItem('user_info');
            try { localStorage.removeItem('user_portfolio_holdings'); } catch (e) {}
            window.location.reload();
        });

        const deleteBtn = document.getElementById('modal-delete-account-btn');
        let isConfirmState = false;

        deleteBtn.addEventListener('click', async () => {
            if (!isConfirmState) {
                isConfirmState = true;
                deleteBtn.style.background = '#EF4444';
                deleteBtn.style.color = '#FFFFFF';
                deleteBtn.style.borderColor = '#DC2626';
                deleteBtn.innerHTML = '⚠️ Confirm Permanent Deletion?';
                
                // Reset back to initial state after 5 seconds if not confirmed
                setTimeout(() => {
                    if (isConfirmState) {
                        isConfirmState = false;
                        deleteBtn.style.background = 'rgba(239,68,68,0.1)';
                        deleteBtn.style.color = '#EF4444';
                        deleteBtn.style.borderColor = 'rgba(239,68,68,0.4)';
                        deleteBtn.innerHTML = '🗑️ Delete Account';
                    }
                }, 5000);
                return;
            }

            // User clicked confirm: Proceed with deletion
            deleteBtn.disabled = true;
            deleteBtn.innerHTML = '⏳ Deleting...';

            const token = localStorage.getItem('jwt_token') || '';
            const userInfoRaw = localStorage.getItem('user_info');
            let parsedUser = user || {};
            if (userInfoRaw) {
                try { parsedUser = Object.assign({}, parsedUser, JSON.parse(userInfoRaw)); } catch (e) {}
            }

            try {
                const response = await fetch('/api/auth/delete-account', {
                    method: 'POST',
                    headers: {
                        'Authorization': `Bearer ${token}`,
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        token: token,
                        email: parsedUser.email || '',
                        user_id: parsedUser.id || null
                    })
                });

                const data = await response.json();
                if (response.ok && data.status === 'success') {
                    alert('✅ Account and all associated data permanently deleted.');
                    localStorage.removeItem('jwt_token');
                    localStorage.removeItem('user_info');
                    localStorage.removeItem('user_portfolio_holdings');
                    window.location.href = '/login';
                } else {
                    alert('❌ Failed to delete account: ' + (data.message || 'Unknown database error'));
                    deleteBtn.disabled = false;
                    deleteBtn.innerHTML = '🗑️ Delete Account';
                    isConfirmState = false;
                }
            } catch (err) {
                console.error('Delete account request error:', err);
                alert('❌ Network error while attempting to delete account.');
                deleteBtn.disabled = false;
                deleteBtn.innerHTML = '🗑️ Delete Account';
                isConfirmState = false;
            }
        });
    }


    // 2. Navigation View Switching
    navDashboard.addEventListener('click', (e) => {
        e.preventDefault();
        setActiveNav(navDashboard, viewDashboard, 'Stock Price Prediction', 'AI-Based Stock Price Forecasting');
    });

    navPrediction.addEventListener('click', (e) => {
        e.preventDefault();
        setActiveNav(navPrediction, viewPrediction, 'Prediction Forecast', `Detailed 7-Day Forecast for ${currentTicker}`);
        renderPredictionTable();
    });

    if (navPortfolio) {
        navPortfolio.addEventListener('click', (e) => {
            e.preventDefault();
            setActiveNav(navPortfolio, viewPortfolio, 'Multi-Asset Portfolio Analytics', 'Valuation, Sector Risk, Sharpe & Equity Performance');
            loadPortfolioData();
        });
    }

    if (refreshPortfolioBtn) {
        refreshPortfolioBtn.addEventListener('click', () => {
            refreshPortfolioBtn.disabled = true;
            const originalHTML = refreshPortfolioBtn.innerHTML;
            refreshPortfolioBtn.innerHTML = '⏳ REFRESHING...';

            loadPortfolioData()
                .catch(err => {
                    console.error('Portfolio refresh failed:', err);
                    alert('Failed to refresh portfolio. Please check your network connection.');
                })
                .finally(() => {
                    refreshPortfolioBtn.disabled = false;
                    refreshPortfolioBtn.innerHTML = originalHTML;
                });
        });
    }

    if (navBacktest) {
        navBacktest.addEventListener('click', (e) => {
            e.preventDefault();
            setActiveNav(navBacktest, viewBacktest, 'Strategy & Monte Carlo Simulation', `Quantitative Simulation Engine for ${currentTicker}`);
            runBacktest(currentTicker);
        });
    }

    if (navAlerts) {
        navAlerts.addEventListener('click', (e) => {
            e.preventDefault();
            setActiveNav(navAlerts, viewAlerts, 'Real-Time Price & Sentiment Alerts', 'Stop-Loss, Target Price & AI Trigger Alerts');
            loadAlertsData();
        });
    }

    if (addAlertBtn) {
        addAlertBtn.addEventListener('click', () => {
            createNewAlert();
        });
    }

    if (runBacktestBtn) {
        runBacktestBtn.addEventListener('click', (e) => {
            e.preventDefault();
            runBacktest(currentTicker);
        });
    }

    if (navComparison) {
        navComparison.addEventListener('click', (e) => {
            e.preventDefault();
            setActiveNav(navComparison, viewComparison, 'Multi-Stock Performance & Risk Comparison', 'Compare returns, volatility, drawdown, and AI forecasts across 2-5 stocks');
            initStockComparisonModule();
        });
    }

    navHistorical.addEventListener('click', (e) => {
        e.preventDefault();
        setActiveNav(navHistorical, viewHistorical, 'Historical Market Data', `Historical Daily Closing Prices for ${currentTicker}`);
        renderHistoricalTable();
    });

    function setActiveNav(navEl, viewEl, title, subtitle) {
        document.querySelectorAll('.sidebar-nav li').forEach(li => li.classList.remove('active'));
        document.querySelectorAll('.page-view').forEach(v => v.style.display = 'none');

        navEl.classList.add('active');
        viewEl.style.display = 'block';

        if (pageTitle) pageTitle.textContent = title;
        if (pageSubtitle) pageSubtitle.textContent = subtitle;
    }

    // 3. Fetch Stock List for Datalist Auto-Complete
    fetchStocksList();

    function fetchStocksList() {
        fetch('/api/stocks')
            .then(res => res.json())
            .then(data => {
                if (data.status === 'success' && data.stocks.length > 0) {
                    stocksDatalist.innerHTML = '';
                    data.stocks.forEach(stk => {
                        const opt = document.createElement('option');
                        opt.value = stk;
                        stocksDatalist.appendChild(opt);
                    });
                }
                if (currentTicker) {
                    loadDashboardData(currentTicker);
                }
            })
            .catch(err => {
                console.error('Error fetching stock list:', err);
                if (currentTicker) {
                    loadDashboardData(currentTicker);
                }
            });
    }

    // 4. Event Listeners & Dynamic Alpha Slider
    const alphaSlider = document.getElementById('alpha-slider');
    const alphaValDisplay = document.getElementById('alpha-val-display');

    if (alphaSlider && alphaValDisplay) {
        alphaSlider.addEventListener('input', (e) => {
            const val = parseFloat(e.target.value).toFixed(2);
            alphaValDisplay.textContent = val;
        });

        alphaSlider.addEventListener('change', (e) => {
            const val = parseFloat(e.target.value).toFixed(2);
            loadDashboardData(currentTicker, val);
        });
    }

    predictBtn.addEventListener('click', () => {
        triggerPredictionFromInput();
    });

    stockInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            triggerPredictionFromInput();
        }
    });

    if (refreshBtn) {
        refreshBtn.addEventListener('click', () => {
            const curAlpha = alphaSlider ? alphaSlider.value : 0.15;
            loadDashboardData(currentTicker, curAlpha);
        });
    }

    function triggerPredictionFromInput() {
        const val = stockInput.value.trim().toUpperCase();
        if (val) {
            currentTicker = val;
            const curAlpha = alphaSlider ? alphaSlider.value : 0.15;
            loadDashboardData(currentTicker, curAlpha);
            if (typeof loadVisualizationTerminal === 'function') {
                loadVisualizationTerminal(currentTicker);
            }
        }
    }

    tfButtons.forEach(btn => {
        btn.addEventListener('click', (e) => {
            tfButtons.forEach(b => b.classList.remove('active'));
            e.target.classList.add('active');
            activeTimeframe = e.target.getAttribute('data-tf');
            updateChartDisplay();
        });
    });

    // Live Streaming State & Elements
    const liveToggleBtn = document.getElementById('live-toggle-btn');
    const liveMarqueeContent = document.getElementById('live-marquee-content');
    const intradayTargetPriceEl = document.getElementById('intraday-target-price');
    const intradayStatusSubEl = document.getElementById('intraday-status-sub');
    const socketDot = document.getElementById('socket-dot');
    const socketText = document.getElementById('socket-status-text');

    let livePollingInterval = null;
    let isLiveStreaming = false;
    let socket = null;
    let pingInterval = null;
    let lastPingTime = 0;

    // Visualization State Variables (declared early to avoid TDZ)
    let visualizationData = null;
    let activeVizPeriod = '1y';
    let vizChartType = 'tradingview';
    let vizAutoRefreshInterval = 10000;
    let vizAutoRefreshTimer = null;

    function updateSocketBadge(status, latencyMs = null) {
        if (!socketText || !socketDot) return;
        if (status === 'connected') {
            socketDot.className = 'socket-dot pulse-green';
            socketText.textContent = latencyMs !== null ? `WS LIVE (${latencyMs}ms)` : 'WS CONNECTED';
        } else if (status === 'connecting') {
            socketDot.className = 'socket-dot pulse-orange';
            socketText.textContent = 'WS CONNECTING...';
        } else {
            socketDot.className = 'socket-dot pulse-red';
            socketText.textContent = 'WS DISCONNECTED';
        }
    }

    // Initialize Socket.IO Client if available
    initWebSocket();

    function initWebSocket() {
        if (typeof io !== 'undefined') {
            try {
                updateSocketBadge('connecting');
                socket = io({ transports: ['websocket', 'polling'] });

                socket.on('connect', () => {
                    console.log('WebSocket connected to server. Subscribing to:', currentTicker);
                    updateSocketBadge('connected');
                    socket.emit('subscribe_ticker', { ticker: currentTicker });

                    // Start ping latency check
                    if (pingInterval) clearInterval(pingInterval);
                    pingInterval = setInterval(() => {
                        if (socket && socket.connected) {
                            lastPingTime = Date.now();
                            socket.emit('ping');
                        }
                    }, 5000);
                });

                socket.on('pong', () => {
                    if (lastPingTime > 0) {
                        const latency = Date.now() - lastPingTime;
                        updateSocketBadge('connected', latency);
                    }
                });

                socket.on('market_update', (data) => {
                    if (data && data.ticker && data.ticker.toUpperCase() === currentTicker.toUpperCase()) {
                        handleLiveMarketUpdate(data);
                    }
                });

                socket.on('disconnect', () => {
                    console.warn('WebSocket disconnected. Reverting to HTTP polling fallback.');
                    updateSocketBadge('disconnected');
                    if (pingInterval) clearInterval(pingInterval);
                });
            } catch (e) {
                console.error('Socket.IO initialization error:', e);
                updateSocketBadge('disconnected');
            }
        } else {
            updateSocketBadge('disconnected');
        }
    }

    function subscribeSocketTicker(ticker) {
        if (socket && socket.connected) {
            socket.emit('subscribe_ticker', { ticker: ticker });
        }
    }

    function handleLiveMarketUpdate(data) {
        if (!data) return;
        const target5m = (data.predicted_5m_target !== undefined && data.predicted_5m_target !== null)
            ? data.predicted_5m_target
            : (data.current_price || 0.0);
        if (intradayTargetPriceEl) {
            intradayTargetPriceEl.textContent = `₹${target5m.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
        }
        if (intradayStatusSubEl) {
            const mktStatus = data.market_status || (data.is_market_open ? 'OPEN 🟢' : 'CLOSED 🔴');
            intradayStatusSubEl.textContent = `Live Socket: ${mktStatus}`;
            intradayStatusSubEl.style.color = data.is_market_open ? '#10B981' : '#64748B';
        }
    }

    // 3. Fetch Stock List for Datalist Auto-Complete & Start Marquee Continuous Auto-Refresh
    fetchStocksList();

    const MARKET_REFRESH_INTERVAL = 15000;
    let previousPricesMap = {};

    window.selectTickerFromMarquee = function(ticker) {
        if (!ticker) return;
        const cleanT = ticker.trim().toUpperCase();
        const formattedT = cleanT.endsWith('.NS') ? cleanT : `${cleanT}.NS`;
        const stockInput = document.getElementById('stock-input');
        const predictBtn = document.getElementById('predict-btn');
        if (stockInput) {
            stockInput.value = formattedT;
        }
        if (predictBtn) {
            predictBtn.click();
        }
    };

    fetchLiveMarquee();
    setInterval(fetchLiveMarquee, MARKET_REFRESH_INTERVAL);

    function fetchLiveMarquee() {
        if (!liveMarqueeContent) return;
        fetch('/api/market-data/status')
            .then(r => r.json())
            .then(data => {
                const freshnessPill = document.getElementById('marquee-freshness-pill');
                if (freshnessPill && data.market_status) {
                    const mktStatus = data.market_status || 'CLOSED';
                    let label = '🟢 Fresh';
                    let bg = 'rgba(16, 185, 129, 0.2)';
                    let border = 'rgba(16, 185, 129, 0.4)';
                    let col = '#10B981';

                    if (mktStatus === 'CLOSED') {
                        label = '⚪ Market Closed';
                        bg = 'rgba(100, 116, 139, 0.2)';
                        border = 'rgba(100, 116, 139, 0.4)';
                        col = '#94A3B8';
                    } else if (data.freshness_label === 'Delayed') {
                        label = '🟡 Delayed';
                        bg = 'rgba(245, 158, 11, 0.2)';
                        border = 'rgba(245, 158, 11, 0.4)';
                        col = '#F59E0B';
                    } else if (data.freshness_label === 'Stale') {
                        label = '🔴 Stale';
                        bg = 'rgba(239, 68, 68, 0.2)';
                        border = 'rgba(239, 68, 68, 0.4)';
                        col = '#EF4444';
                    }

                    freshnessPill.textContent = label;
                    freshnessPill.style.background = bg;
                    freshnessPill.style.borderColor = border;
                    freshnessPill.style.color = col;
                }

                let tickers = data.top_tickers || [];
                if (tickers.length === 0 && data.marquee) tickers = data.marquee;

                if (tickers && tickers.length > 0) {
                    let itemsHtml = '';
                    tickers.forEach(item => {
                        const sym = item.display_symbol || (item.symbol ? item.symbol.replace('.NS', '') : (item.ticker ? String(item.ticker).replace('.NS', '') : ''));
                        const fullSym = item.symbol || (sym ? `${sym}.NS` : '');
                        const ltp = Number(item.ltp || item.price || 0.0);
                        const change = Number(item.change || 0.0);
                        const changePct = Number(item.change_percent || item.change_pct || 0.0);
                        const isPos = changePct >= 0;
                        const colorHex = isPos ? '#10B981' : '#EF4444';
                        const arrow = isPos ? '▲' : '▼';
                        const sign = isPos ? '+' : '';

                        // Determine price flash effect
                        let flashClass = '';
                        if (previousPricesMap[sym] !== undefined) {
                            if (ltp > previousPricesMap[sym]) flashClass = 'flash-up';
                            else if (ltp < previousPricesMap[sym]) flashClass = 'flash-down';
                        }
                        previousPricesMap[sym] = ltp;

                        itemsHtml += `
                            <span class="marquee-item ${flashClass}" onclick="selectTickerFromMarquee('${fullSym}')" title="Click to view ${sym} charts and AI forecast">
                                <strong>${sym}:</strong> ₹${ltp.toFixed(2)} 
                                <span style="color: ${colorHex};">${arrow} ${sign}${changePct.toFixed(2)}%</span>
                            </span>
                        `;
                    });

                    // Duplicate items to ensure smooth infinite seamless marquee scrolling
                    liveMarqueeContent.innerHTML = itemsHtml + itemsHtml;
                }
            })
            .catch(err => {
                console.warn('Error fetching marquee status, preserving last valid data:', err);
                const freshnessPill = document.getElementById('marquee-freshness-pill');
                if (freshnessPill) {
                    freshnessPill.textContent = '🟡 Delayed (Cached)';
                    freshnessPill.style.background = 'rgba(245, 158, 11, 0.2)';
                    freshnessPill.style.borderColor = 'rgba(245, 158, 11, 0.4)';
                    freshnessPill.style.color = '#F59E0B';
                }
            });
    }

    if (liveToggleBtn) {
        liveToggleBtn.addEventListener('click', () => {
            isLiveStreaming = !isLiveStreaming;
            if (isLiveStreaming) {
                liveToggleBtn.classList.add('active');
                liveToggleBtn.innerHTML = `<span class="live-dot pulse"></span> LIVE STREAMING ON`;
                startLivePolling();
            } else {
                liveToggleBtn.classList.remove('active');
                liveToggleBtn.innerHTML = `<span class="live-dot pulse"></span> LIVE (10s)`;
                stopLivePolling();
            }
        });
    }

    function startLivePolling() {
        stopLivePolling();
        subscribeSocketTicker(currentTicker);
        const curAlpha = alphaSlider ? alphaSlider.value : 0.15;
        loadDashboardData(currentTicker, curAlpha);
        livePollingInterval = setInterval(() => {
            if (isLiveStreaming) {
                loadLiveIntradayData(currentTicker);
                fetchLiveMarquee();
            }
        }, 10000);
    }

    function stopLivePolling() {
        if (livePollingInterval) {
            clearInterval(livePollingInterval);
            livePollingInterval = null;
        }
    }

    function loadLiveIntradayData(ticker) {
        fetch(`/api/live/${ticker}`)
            .then(r => r.json())
            .then(data => {
                if (data.status === 'success') {
                    const target5m = (data.predicted_5m_target !== undefined && data.predicted_5m_target !== null) 
                        ? data.predicted_5m_target 
                        : (data.current_price || 0.0);
                    if (intradayTargetPriceEl) {
                        intradayTargetPriceEl.textContent = `₹${target5m.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
                    }
                    if (intradayStatusSubEl) {
                        const mktStatus = data.market_status || (data.is_market_open ? 'OPEN 🟢' : 'CLOSED 🔴');
                        intradayStatusSubEl.textContent = `Market: ${mktStatus}`;
                        intradayStatusSubEl.style.color = data.is_market_open ? '#10B981' : '#64748B';
                    }
                }
            })
            .catch(err => console.error('Error loading live intraday:', err));
    }

    // 5. Main Data Loader
    function loadDashboardData(ticker, alpha = 0.15) {
        predictBtn.disabled = true;
        predictBtn.innerHTML = '⏳ Loading...';

        const predPromise = fetch(`/api/prediction/${ticker}?alpha=${alpha}`).then(r => r.json());
        const histPromise = fetch(`/api/historical/${ticker}?period=1y`).then(r => r.json());

        Promise.all([predPromise, histPromise])
            .then(([predData, histData]) => {
                predictBtn.disabled = false;
                predictBtn.innerHTML = '⚡ PREDICT';

                if (predData.status === 'success') {
                    currentPredictionData = predData;
                    updatePredictionCards(predData);
                    updatePredictionInfo(predData);
                    updateIntradayTargets(predData);
                    renderModelComparisonCards(predData);
                    renderPredictionTable();
                    loadLiveIntradayData(ticker);
                    if (typeof loadVisualizationTerminal === 'function') {
                        loadVisualizationTerminal(ticker);
                    }
                } else {
                    alert(`Warning: Unable to load predictions for ${ticker}. Error: ${predData.message || 'Unknown error'}`);
                }

                if (histData.status === 'success') {
                    rawHistoricalData = histData.data;
                    updateChartDisplay();
                    renderHistoricalTable();
                }

                // Auto-run Monte Carlo risk simulation & backtest calculation for selected ticker
                runBacktest(ticker);
            })
            .catch(err => {
                predictBtn.disabled = false;
                predictBtn.innerHTML = '⚡ PREDICT';
                console.error('Error loading dashboard data:', err);
            });
    }

    // 6. Update Prediction Cards
    function updatePredictionCards(data) {
        const currP = data.current_price || 0.0;
        const predP = data.predicted_price || 0.0;
        const expRet = data.expected_return || 0.0;
        const sig = (data.signal || 'HOLD').toUpperCase();

        const diff = predP - currP;
        const diffSign = diff >= 0 ? '+' : '';

        currentPriceEl.textContent = `₹${currP.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
        currentDateSubEl.textContent = `Base Date: ${data.prediction_date || isoDate}`;

        predictedPriceEl.textContent = `₹${predP.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
        predictedDeltaSubEl.textContent = `${diffSign}₹${diff.toFixed(2)} forecast`;
        predictedDeltaSubEl.style.color = diff >= 0 ? '#10B981' : '#EF4444';

        const retSign = expRet >= 0 ? '+' : '';
        expectedReturnEl.textContent = `${retSign}${expRet.toFixed(2)}%`;
        expectedReturnEl.style.color = expRet >= 0 ? '#10B981' : '#EF4444';

        predictionSignalEl.textContent = sig;
        predictionSignalEl.className = 'badge';
        if (sig.includes('BUY')) {
            predictionSignalEl.classList.add('badge-buy');
        } else if (sig.includes('SELL')) {
            predictionSignalEl.classList.add('badge-sell');
        } else {
            predictionSignalEl.classList.add('badge-hold');
        }

        // Update Dynamic Plain-English Chart Summary Banner
        const summaryTickerEl = document.getElementById('chart-summary-ticker');
        const summaryTextEl = document.getElementById('chart-summary-text');
        if (summaryTickerEl) summaryTickerEl.textContent = data.ticker || currentTicker;
        if (summaryTextEl) {
            const trendWord = expRet >= 0 ? 'Bullish' : 'Bearish';
            const actionWord = expRet >= 0 ? 'predicts a rise' : 'predicts a decline';
            summaryTextEl.textContent = `Stock is trading at ₹${currP.toFixed(2)}. AI model ${actionWord} of ${retSign}${expRet.toFixed(2)}% to ₹${predP.toFixed(2)} (Signal: ${sig}).`;
        }
    }

    // 6.5 Update Risk Management, Probabilities & Fundamentals Cards
    function updateIntradayTargets(data) {
        const stopEl = document.getElementById('target-stop');
        const tpEl = document.getElementById('target-tp');
        const rrEl = document.getElementById('target-rr');
        const threshEl = document.getElementById('dynamic-thresh-display');

        const probRiseEl = document.getElementById('prob-rise-display');
        const probFallEl = document.getElementById('prob-fall-display');
        const confEl = document.getElementById('confidence-pct-display');

        const peEl = document.getElementById('fund-pe');
        const pbEl = document.getElementById('fund-pb');
        const epsEl = document.getElementById('fund-eps');
        const dteEl = document.getElementById('fund-dte');
        const mcapEl = document.getElementById('fund-mcap');

        // Stop-Loss & Take-Profit
        if (stopEl && data.stop_loss) stopEl.textContent = `₹${data.stop_loss.toFixed(2)}`;
        if (tpEl && data.take_profit) tpEl.textContent = `₹${data.take_profit.toFixed(2)}`;
        if (rrEl && data.risk_reward_ratio) rrEl.textContent = data.risk_reward_ratio;
        if (threshEl && data.dynamic_buy_threshold) threshEl.textContent = `±${data.dynamic_buy_threshold.toFixed(2)}%`;

        // Probabilities
        if (probRiseEl && data.prob_rise_pct !== undefined) probRiseEl.textContent = `${data.prob_rise_pct.toFixed(1)}%`;
        if (probFallEl && data.prob_fall_pct !== undefined) probFallEl.textContent = `${data.prob_fall_pct.toFixed(1)}%`;
        if (confEl && data.confidence_pct !== undefined) confEl.textContent = `${data.confidence_pct.toFixed(1)}%`;

        // Fundamentals
        const f = data.fundamentals || {};
        if (peEl && f.PE_Ratio !== undefined) peEl.textContent = f.PE_Ratio.toFixed(1);
        if (pbEl && f.PB_Ratio !== undefined) pbEl.textContent = f.PB_Ratio.toFixed(2);
        if (epsEl && f.EPS_Growth !== undefined) {
            const sign = f.EPS_Growth >= 0 ? '+' : '';
            epsEl.textContent = `${sign}${(f.EPS_Growth * 100).toFixed(1)}%`;
            epsEl.style.color = f.EPS_Growth >= 0 ? '#10B981' : '#EF4444';
        }
        if (dteEl && f.Debt_To_Equity !== undefined) dteEl.textContent = f.Debt_To_Equity.toFixed(2);
        if (mcapEl && f.Market_Cap_B !== undefined) mcapEl.textContent = `₹${f.Market_Cap_B.toFixed(1)}B`;
    }

    // 6.6 Render AI Model Multi-Model Ensemble Comparison Cards
    function renderModelComparisonCards(data) {
        const container = document.getElementById('model-comparison-container');
        if (!container) return;

        const currPrice = data.current_price || 0.0;
        const breakdown = data.model_breakdown || {};

        const models = [
            { key: 'random_forest', name: 'Random Forest', color: '#10B981', bg: '#ECFDF5', border: '#A7F3D0' },
            { key: 'xgboost', name: 'XGBoost', color: '#2563EB', bg: '#EFF6FF', border: '#BFDBFE' },
            { key: 'lstm', name: 'LSTM Deep Net', color: '#8B5CF6', bg: '#F5F3FF', border: '#DDD6FE' },
            { key: 'lightgbm', name: 'LightGBM', color: '#F59E0B', bg: '#FFFBEB', border: '#FDE68A' },
            { key: 'ensemble', name: 'Weighted Ensemble', color: '#EC4899', bg: '#FDF2F8', border: '#FBCFE8' }
        ];

        let html = '';
        models.forEach(m => {
            const prices = breakdown[m.key] || [];
            const target7d = prices.length > 0 ? prices[prices.length - 1] : currPrice;
            const retPct = currPrice > 0 ? ((target7d - currPrice) / currPrice) * 100.0 : 0.0;
            const sign = retPct >= 0 ? '+' : '';
            const sigTag = retPct >= 1.5 ? 'BUY 🚀' : (retPct <= -1.5 ? 'SELL 📉' : 'HOLD ⏸️');

            html += `
                <div style="background: ${m.bg}; border: 1px solid ${m.border}; border-radius: 10px; padding: 12px; transition: all 0.2s ease; box-shadow: 0 1px 3px rgba(0,0,0,0.03);">
                    <div style="font-size: 11.5px; font-weight: 700; color: ${m.color}; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px;">${m.name}</div>
                    <div style="font-size: 18px; font-weight: 800; color: #0F172A; margin: 4px 0;">₹${target7d.toFixed(2)}</div>
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 6px;">
                        <span style="font-size: 12px; font-weight: 700; color: ${retPct >= 0 ? '#10B981' : '#EF4444'};">${sign}${retPct.toFixed(2)}%</span>
                        <span style="font-size: 10px; font-weight: 800; padding: 2px 6px; border-radius: 4px; background: #FFFFFF; color: ${m.color}; border: 1px solid ${m.border};">${sigTag}</span>
                    </div>
                </div>
            `;
        });
        container.innerHTML = html;
    }

    // 6.8 Run Strategy & Monte Carlo Backtest
    function runBacktest(ticker) {
        if (!runBacktestBtn) return;
        runBacktestBtn.disabled = true;
        runBacktestBtn.innerHTML = '⏳ Running Backtest...';

        const selectedStrat = stratSelector ? stratSelector.value : 'momentum';

        fetch(`/api/backtest/${ticker}?period=1y&strategy=${selectedStrat}`)
            .then(r => r.json())
            .then(data => {
                runBacktestBtn.disabled = false;
                runBacktestBtn.innerHTML = '▶ RUN BACKTEST & MONTE CARLO';

                if (data.status === 'success') {
                    if (backtestViewTitle) backtestViewTitle.textContent = `⚡ Strategy & Monte Carlo Simulation — ${ticker} (${data.strategy || 'Momentum'})`;

                    const stratRetEl = document.getElementById('bt-strat-return');
                    const benchRetEl = document.getElementById('bt-bench-return');
                    const sharpeEl = document.getElementById('bt-sharpe');
                    const maxDdEl = document.getElementById('bt-max-dd');
                    const taxValEl = document.getElementById('bt-tax-val');
                    const winRateEl = document.getElementById('bt-win-rate');
                    const tradesSubEl = document.getElementById('bt-total-trades');

                    if (stratRetEl) {
                        const sign = data.strategy_return_pct >= 0 ? '+' : '';
                        stratRetEl.textContent = `${sign}${data.strategy_return_pct.toFixed(2)}%`;
                        stratRetEl.style.color = data.strategy_return_pct >= 0 ? '#10B981' : '#EF4444';
                    }
                    if (benchRetEl) {
                        const sign = data.benchmark_return_pct >= 0 ? '+' : '';
                        benchRetEl.textContent = `${sign}${data.benchmark_return_pct.toFixed(2)}%`;
                    }
                    if (sharpeEl) sharpeEl.textContent = data.sharpe_ratio.toFixed(2);
                    if (maxDdEl) maxDdEl.textContent = `-${data.max_drawdown_pct.toFixed(2)}%`;
                    if (taxValEl) taxValEl.textContent = `₹${(data.total_statutory_taxes || 0.0).toFixed(2)}`;
                    if (winRateEl) winRateEl.textContent = `${data.win_rate_pct.toFixed(1)}%`;
                    if (tradesSubEl) tradesSubEl.textContent = `${data.total_trades} Trades`;

                    // Render Monte Carlo Percentiles
                    const mc = data.monte_carlo || {};
                    const retPctiles = mc.return_percentiles || {};
                    const p5El = document.getElementById('mc-p5');
                    const p25El = document.getElementById('mc-p25');
                    const p50El = document.getElementById('mc-p50');
                    const p75El = document.getElementById('mc-p75');
                    const p95El = document.getElementById('mc-p95');

                    if (p5El) p5El.textContent = `${retPctiles.p5 >= 0 ? '+' : ''}${(retPctiles.p5 || 0).toFixed(2)}%`;
                    if (p25El) p25El.textContent = `${retPctiles.p25 >= 0 ? '+' : ''}${(retPctiles.p25 || 0).toFixed(2)}%`;
                    if (p50El) p50El.textContent = `${retPctiles.median_p50 >= 0 ? '+' : ''}${(retPctiles.median_p50 || 0).toFixed(2)}%`;
                    if (p75El) p75El.textContent = `${retPctiles.p75 >= 0 ? '+' : ''}${(retPctiles.p75 || 0).toFixed(2)}%`;
                    if (p95El) p95El.textContent = `${retPctiles.p95 >= 0 ? '+' : ''}${(retPctiles.p95 || 0).toFixed(2)}%`;

                    // Render Equity Curve Chart
                    renderBacktestChart(data.equity_curve || []);

                    // Render Trade Log Table
                    renderTradeLogTable(data.trade_log || []);
                } else {
                    alert(`Backtest error: ${data.message || 'Unknown error'}`);
                }
            })
            .catch(err => {
                runBacktestBtn.disabled = false;
                runBacktestBtn.innerHTML = '▶ RUN BACKTEST & MONTE CARLO';
                console.error('Error running backtest:', err);
            });
    }

    // 6.9 Portfolio Analytics Loader & Dynamic Holdings Management
    const defaultHoldings = [
        { ticker: "RELIANCE.NS", shares: 10, buy_price: 2400.0 },
        { ticker: "TCS.NS", shares: 5, buy_price: 3500.0 },
        { ticker: "HDFCBANK.NS", shares: 15, buy_price: 1500.0 }
    ];

    function getUserPortfolioHoldings() {
        try {
            const userId = getCurrentUserId();
            const key = `user_portfolio_holdings_${userId}`;
            const stored = localStorage.getItem(key);
            if (stored !== null && stored !== 'undefined') {
                const parsed = JSON.parse(stored);
                if (Array.isArray(parsed)) return parsed;
            }
            // Migration check for guest user legacy key
            if (userId === 'guest_user') {
                const legacy = localStorage.getItem('user_portfolio_holdings');
                if (legacy !== null && legacy !== 'undefined') {
                    const parsed = JSON.parse(legacy);
                    if (Array.isArray(parsed)) return parsed;
                }
            }
        } catch (e) {
            console.error('Error reading saved portfolio from localStorage:', e);
        }
        return [...defaultHoldings];
    }

    function saveUserPortfolioHoldings(holdings) {
        try {
            const userId = getCurrentUserId();
            const key = `user_portfolio_holdings_${userId}`;
            localStorage.setItem(key, JSON.stringify(holdings));
        } catch (e) {
            console.error('Error saving portfolio to localStorage:', e);
        }
    }

    // Attach Portfolio Manager Event Listeners
    const addStockBtn = document.getElementById('add-stock-btn');
    const resetPortfolioBtn = document.getElementById('reset-portfolio-btn');
    const portAddTicker = document.getElementById('port-add-ticker');
    const portAddShares = document.getElementById('port-add-shares');
    const portAddBuyPrice = document.getElementById('port-add-buy-price');

    if (addStockBtn) {
        addStockBtn.addEventListener('click', () => {
            const rawTicker = portAddTicker ? portAddTicker.value.trim().toUpperCase() : '';
            const sharesVal = portAddShares ? parseFloat(portAddShares.value) : 0;
            const buyPriceVal = portAddBuyPrice ? parseFloat(portAddBuyPrice.value) : 0;

            if (!rawTicker) {
                alert('Please enter a valid stock ticker (e.g., INFY.NS, TATAMOTORS.NS)');
                return;
            }
            if (isNaN(sharesVal) || sharesVal <= 0) {
                alert('Please enter a valid number of shares greater than 0');
                return;
            }
            if (isNaN(buyPriceVal) || buyPriceVal <= 0) {
                alert('Please enter a valid buy price in ₹');
                return;
            }

            let formattedTicker = rawTicker;
            if (!formattedTicker.endsWith('.NS') && !formattedTicker.endsWith('.BO') && formattedTicker.length <= 12) {
                formattedTicker = `${formattedTicker}.NS`;
            }

            let currentHoldings = getUserPortfolioHoldings();
            const existingIdx = currentHoldings.findIndex(h => h.ticker.toUpperCase() === formattedTicker);
            if (existingIdx >= 0) {
                currentHoldings[existingIdx].shares += sharesVal;
                currentHoldings[existingIdx].buy_price = buyPriceVal;
            } else {
                currentHoldings.push({ ticker: formattedTicker, shares: sharesVal, buy_price: buyPriceVal });
            }

            saveUserPortfolioHoldings(currentHoldings);
            if (portAddTicker) portAddTicker.value = '';
            loadPortfolioData();
        });
    }

    window.removeStockFromPortfolio = function (index) {
        let currentHoldings = getUserPortfolioHoldings();
        if (index >= 0 && index < currentHoldings.length) {
            currentHoldings.splice(index, 1);
            saveUserPortfolioHoldings(currentHoldings);
            loadPortfolioData();
        }
    };

    function loadPortfolioData() {
        const holdings = getUserPortfolioHoldings();

        return fetch('/api/portfolio', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ holdings: holdings, period: '1y' })
        })
            .then(r => r.json())
            .then(data => {
                if (data.status === 'success') {
                    const mktValEl = document.getElementById('port-mkt-val');
                    const costSubEl = document.getElementById('port-cost-sub');
                    const pnlValEl = document.getElementById('port-pnl-val');
                    const pnlPctSubEl = document.getElementById('port-pnl-pct-sub');
                    const annReturnEl = document.getElementById('port-ann-return');
                    const sharpeEl = document.getElementById('port-sharpe');
                    const maxDdEl = document.getElementById('port-max-dd');
                    const tableBody = document.getElementById('portfolio-table-body');
                    const sectorListEl = document.getElementById('sector-breakdown-list');

                    if (mktValEl) mktValEl.textContent = `₹${data.current_market_value.toLocaleString('en-IN', { minimumFractionDigits: 2 })}`;
                    if (costSubEl) costSubEl.textContent = `Cost Basis: ₹${data.total_cost_basis.toLocaleString('en-IN', { minimumFractionDigits: 2 })}`;

                    const pnlSign = data.total_unrealized_pnl >= 0 ? '+' : '';
                    if (pnlValEl) {
                        pnlValEl.textContent = `${pnlSign}₹${data.total_unrealized_pnl.toLocaleString('en-IN', { minimumFractionDigits: 2 })}`;
                        pnlValEl.style.color = data.total_unrealized_pnl >= 0 ? '#10B981' : '#EF4444';
                    }
                    if (pnlPctSubEl) pnlPctSubEl.textContent = `${pnlSign}${data.total_unrealized_pnl_pct.toFixed(2)}%`;

                    if (annReturnEl) {
                        const retSign = data.annualized_return_pct >= 0 ? '+' : '';
                        annReturnEl.textContent = `${retSign}${data.annualized_return_pct.toFixed(2)}%`;
                    }
                    if (sharpeEl) sharpeEl.textContent = data.sharpe_ratio.toFixed(2);
                    if (maxDdEl) maxDdEl.textContent = `-${data.max_drawdown_pct.toFixed(2)}%`;

                    // Render Advanced Quantitative Risk Metrics (Sortino, VaR, Risk Score 0-100)
                    const sortinoEl = document.getElementById('port-sortino');
                    const varEl = document.getElementById('port-var-95');
                    const riskScoreEl = document.getElementById('port-risk-score');
                    const riskLevelBadge = document.getElementById('port-risk-level-badge');
                    const sectorWarningContainer = document.getElementById('portfolio-sector-warnings');

                    if (sortinoEl) sortinoEl.textContent = (data.sortino_ratio || 0.0).toFixed(2);
                    if (varEl) varEl.textContent = `-${(data.var_95_pct || 0.0).toFixed(2)}%`;
                    if (riskScoreEl) riskScoreEl.textContent = `${data.portfolio_risk_score || 0.0}/100`;

                    if (riskLevelBadge) {
                        riskLevelBadge.textContent = data.portfolio_risk_level || 'Low Risk 🟢';
                        if ((data.portfolio_risk_score || 0) < 35) {
                            riskLevelBadge.style.background = '#ECFDF5';
                            riskLevelBadge.style.color = '#059669';
                            riskLevelBadge.style.borderColor = '#A7F3D0';
                        } else if ((data.portfolio_risk_score || 0) < 65) {
                            riskLevelBadge.style.background = '#FEF3C7';
                            riskLevelBadge.style.color = '#D97706';
                            riskLevelBadge.style.borderColor = '#FDE68A';
                        } else {
                            riskLevelBadge.style.background = '#FEF2F2';
                            riskLevelBadge.style.color = '#DC2626';
                            riskLevelBadge.style.borderColor = '#FCA5A5';
                        }
                    }

                    // Render Sector Concentration Warnings
                    if (sectorWarningContainer) {
                        const warnings = data.sector_warnings || [];
                        if (warnings.length > 0) {
                            sectorWarningContainer.style.display = 'block';
                            sectorWarningContainer.innerHTML = warnings.map(w => `
                                <div style="background: #FFFBEB; border: 1px solid #FDE68A; color: #92400E; padding: 10px 14px; border-radius: 8px; font-size: 12.5px; font-weight: 600; margin-bottom: 10px; display: flex; align-items: center; gap: 8px;">
                                    <span>${w}</span>
                                </div>
                            `).join('');
                        } else {
                            sectorWarningContainer.style.display = 'none';
                            sectorWarningContainer.innerHTML = '';
                        }
                    }

                    // Render Holdings Table
                    if (tableBody) {
                        let rowsHTML = '';
                        (data.holdings || []).forEach((h, idx) => {
                            const pnlSign = h.unrealized_pnl >= 0 ? '+' : '';
                            const pnlColor = h.unrealized_pnl >= 0 ? '#10B981' : '#EF4444';
                            const pnlBg = h.unrealized_pnl >= 0 ? '#ECFDF5' : '#FEF2F2';
                            const isEven = idx % 2 === 0;
                            const rowBg = isEven ? '#FFFFFF' : '#F8FAFC';

                            rowsHTML += `
                                <tr style="background: ${rowBg}; border-bottom: 1px solid #F1F5F9; transition: background 0.15s ease;">
                                    <td style="padding: 12px 14px; font-weight: 700; color: #0F172A; font-size: 13.5px;">${h.ticker}</td>
                                    <td style="padding: 12px 14px; color: #475569; font-size: 13px; font-weight: 500;">${h.company}</td>
                                    <td style="padding: 12px 14px;">
                                        <span style="background: #EFF6FF; color: #1E40AF; padding: 4px 10px; border-radius: 6px; font-size: 11.5px; font-weight: 600; border: 1px solid #DBEAFE;">${h.sector}</span>
                                    </td>
                                    <td style="padding: 12px 14px; text-align: right; font-weight: 600; color: #334155; font-size: 13px;">${h.shares}</td>
                                    <td style="padding: 12px 14px; text-align: right; color: #64748B; font-size: 13px;">₹${h.buy_price.toFixed(2)}</td>
                                    <td style="padding: 12px 14px; text-align: right; color: #0F172A; font-weight: 600; font-size: 13px;">₹${h.current_price.toFixed(2)}</td>
                                    <td style="padding: 12px 14px; text-align: right; font-weight: 700; color: #0F172A; font-size: 13.5px;">₹${h.current_value.toLocaleString('en-IN', { minimumFractionDigits: 2 })}</td>
                                    <td style="padding: 12px 14px; text-align: right;">
                                        <span style="background: ${pnlBg}; color: ${pnlColor}; font-weight: 700; font-size: 12px; padding: 4px 10px; border-radius: 6px; display: inline-block;">
                                            ${pnlSign}₹${Math.abs(h.unrealized_pnl).toLocaleString('en-IN', { minimumFractionDigits: 2 })} (${pnlSign}${h.unrealized_pnl_pct.toFixed(2)}%)
                                        </span>
                                    </td>
                                    <td style="padding: 12px 14px; text-align: center;">
                                        <button onclick="window.removeStockFromPortfolio(${idx})" style="padding: 5px 12px; font-size: 11.5px; font-weight: 600; color: #EF4444; background: #FEF2F2; border: 1px solid #FCA5A5; border-radius: 6px; cursor: pointer; transition: all 0.2s ease;" title="Remove ${h.ticker} from portfolio" onmouseover="this.style.background='#EF4444';this.style.color='#FFFFFF';" onmouseout="this.style.background='#FEF2F2';this.style.color='#EF4444';">
                                            🗑️ Remove
                                        </button>
                                    </td>
                                </tr>
                            `;
                        });
                        tableBody.innerHTML = rowsHTML || `<tr><td colspan="9" style="text-align:center; padding: 24px; color:#94A3B8; font-size: 13.5px;">No stocks in portfolio yet. Use the form above to add your stock holdings!</td></tr>`;
                    }

                    // Render Sector Breakdown List with rich icons, rupee values, and badges
                    if (sectorListEl) {
                        let secHTML = '';
                        const sectorThemeMap = {
                            'Information Technology': { icon: '💻', color: '#2563EB', bg: '#EFF6FF', bar: 'linear-gradient(90deg, #2563EB, #3B82F6)' },
                            'Banking & Financials': { icon: '🏦', color: '#10B981', bg: '#ECFDF5', bar: 'linear-gradient(90deg, #10B981, #34D399)' },
                            'Financials': { icon: '🏦', color: '#10B981', bg: '#ECFDF5', bar: 'linear-gradient(90deg, #10B981, #34D399)' },
                            'Energy & Conglomerate': { icon: '⚡', color: '#F59E0B', bg: '#FFFBEB', bar: 'linear-gradient(90deg, #F59E0B, #FBBF24)' },
                            'Energy': { icon: '⚡', color: '#F59E0B', bg: '#FFFBEB', bar: 'linear-gradient(90deg, #F59E0B, #FBBF24)' },
                            'Automobiles': { icon: '🚗', color: '#8B5CF6', bg: '#F5F3FF', bar: 'linear-gradient(90deg, #8B5CF6, #A78BFA)' },
                            'Healthcare': { icon: '💊', color: '#EC4899', bg: '#FDF2F8', bar: 'linear-gradient(90deg, #EC4899, #F472B6)' },
                            'Consumer Goods': { icon: '🛍️', color: '#06B6D4', bg: '#ECFEFF', bar: 'linear-gradient(90deg, #06B6D4, #22D3EE)' },
                            'Other': { icon: '📊', color: '#64748B', bg: '#F8FAFC', bar: 'linear-gradient(90deg, #64748B, #94A3B8)' }
                        };

                        const defaultTheme = { icon: '📊', color: '#3B82F6', bg: '#EFF6FF', bar: 'linear-gradient(90deg, #2563EB, #3B82F6)' };

                        (data.sector_breakdown || []).forEach((sec) => {
                            const theme = sectorThemeMap[sec.sector] || defaultTheme;
                            const secValueStr = sec.value ? `₹${sec.value.toLocaleString('en-IN', { minimumFractionDigits: 2 })}` : '';

                            secHTML += `
                                <div style="background: #FFFFFF; padding: 14px; border-radius: 12px; border: 1px solid #E2E8F0; box-shadow: 0 2px 6px rgba(15, 23, 42, 0.02);">
                                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                                        <div style="display: flex; align-items: center; gap: 8px;">
                                            <span style="font-size: 16px; background: ${theme.bg}; padding: 4px 8px; border-radius: 6px;">${theme.icon}</span>
                                            <div>
                                                <div style="font-size: 13px; font-weight: 700; color: #0F172A;">${sec.sector}</div>
                                                ${secValueStr ? `<div style="font-size: 11.5px; color: #64748B; font-weight: 500;">${secValueStr}</div>` : ''}
                                            </div>
                                        </div>
                                        <span style="background: ${theme.bg}; color: ${theme.color}; font-weight: 700; font-size: 12px; padding: 4px 10px; border-radius: 20px; border: 1px solid ${theme.color}30;">
                                            ${sec.weight_pct}%
                                        </span>
                                    </div>
                                    <div style="width: 100%; height: 7px; background: #F1F5F9; border-radius: 6px; overflow: hidden; margin-top: 4px;">
                                        <div style="width: ${sec.weight_pct}%; height: 100%; background: ${theme.bar}; border-radius: 6px; transition: width 0.4s ease;"></div>
                                    </div>
                                </div>
                            `;
                        });

                        sectorListEl.innerHTML = secHTML || `
                            <div style="text-align: center; padding: 24px; color: #94A3B8; background: #F8FAFC; border-radius: 12px; border: 1px dashed #CBD5E1;">
                                <div style="font-size: 24px; margin-bottom: 6px;">📊</div>
                                <div style="font-size: 13px; font-weight: 500;">No Sector Data</div>
                                <div style="font-size: 11.5px; color: #CBD5E1; margin-top: 2px;">Add stocks above to view sector exposure breakdown</div>
                            </div>
                        `;
                    }
                }
            })
            .catch(err => console.error('Error loading portfolio:', err));
    }

    // 6.10 Alert Management Loader
    function loadAlertsData() {
        const userId = getCurrentUserId();
        fetch(`/api/alerts?user_id=${encodeURIComponent(userId)}`)
            .then(r => r.json())
            .then(data => {
                if (data.status === 'success') {
                    const tbody = document.getElementById('alerts-table-body');
                    if (!tbody) return;
                    const alerts = data.alerts || [];

                    if (alerts.length === 0) {
                        tbody.innerHTML = `<tr><td colspan="7" style="text-align: center; color: #94A3B8;">No active alert rules. Create one above!</td></tr>`;
                        return;
                    }

                    let htmlStr = '';
                    alerts.forEach(a => {
                        const statusBadge = a.triggered ?
                            '<span style="background: #FEF2F2; color: #EF4444; padding: 2px 8px; border-radius: 4px; font-weight: 700;">TRIGGERED 🔴</span>' :
                            '<span style="background: #ECFDF5; color: #10B981; padding: 2px 8px; border-radius: 4px; font-weight: 700;">ACTIVE 🟢</span>';

                        htmlStr += `
                            <tr>
                                <td><code style="font-size: 11px;">${a.id}</code></td>
                                <td><b>${a.ticker}</b></td>
                                <td><span style="font-weight: 600; color: #2563EB;">${a.type}</span></td>
                                <td><b>₹${a.target_value}</b></td>
                                <td>${statusBadge}</td>
                                <td>${a.created_at}</td>
                                <td>
                                    <button class="btn btn-secondary" onclick="window.deleteAlertRule('${a.id}')" style="padding: 4px 8px; font-size: 11px; color: #EF4444; border-color: #FCA5A5;">
                                        🗑️ Delete
                                    </button>
                                </td>
                            </tr>
                        `;
                    });
                    tbody.innerHTML = htmlStr;
                }
            })
            .catch(err => console.error('Error loading alerts:', err));
    }

    function createNewAlert() {
        if (!alertTickerInput || !alertTargetInput) return;
        const ticker = alertTickerInput.value.trim();
        const type = alertTypeSelect.value;
        const target_value = parseFloat(alertTargetInput.value);
        const notes = alertNotesInput ? alertNotesInput.value.trim() : '';

        if (!ticker || isNaN(target_value)) {
            alert('Please enter a valid ticker and target threshold value');
            return;
        }

        const userId = getCurrentUserId();
        fetch('/api/alerts', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ user_id: userId, ticker, type, target_value, notes })
        })
            .then(r => r.json())
            .then(data => {
                if (data.status === 'success') {
                    loadAlertsData();
                    if (alertNotesInput) alertNotesInput.value = '';
                } else {
                    alert(`Error adding alert: ${data.message}`);
                }
            })
            .catch(err => console.error('Error creating alert:', err));
    }

    window.deleteAlertRule = function (id) {
        const userId = getCurrentUserId();
        fetch(`/api/alerts?id=${encodeURIComponent(id)}&user_id=${encodeURIComponent(userId)}`, { method: 'DELETE' })
            .then(r => r.json())
            .then(data => {
                if (data.status === 'success') {
                    loadAlertsData();
                } else {
                    alert(`Error deleting alert: ${data.message}`);
                }
            })
            .catch(err => console.error('Error deleting alert:', err));
    };

    function renderBacktestChart(equityData) {
        const canvas = document.getElementById('backtestChart');
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        if (backtestChart) backtestChart.destroy();

        if (!equityData || equityData.length === 0) return;

        const labels = equityData.map(e => e.date);
        const stratVals = equityData.map(e => e.portfolio_value);
        const benchVals = equityData.map(e => e.benchmark_value);
        const startCapital = equityData[0].portfolio_value || 100000.0;

        // Gradient Fill for AI Strategy Line
        const gradientStrat = ctx.createLinearGradient(0, 0, 0, 300);
        gradientStrat.addColorStop(0, 'rgba(16, 185, 129, 0.25)');
        gradientStrat.addColorStop(1, 'rgba(16, 185, 129, 0.01)');

        backtestChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: '🟢 AI Strategy Portfolio (Active Trading)',
                        data: stratVals,
                        borderColor: '#10B981',
                        backgroundColor: gradientStrat,
                        borderWidth: 2.5,
                        fill: true,
                        pointRadius: 0,
                        pointHoverRadius: 6,
                        pointHoverBackgroundColor: '#10B981',
                        pointHoverBorderColor: '#FFFFFF',
                        pointHoverBorderWidth: 2,
                        tension: 0.2
                    },
                    {
                        label: '🔷 Buy & Hold Benchmark (Passive Stock Holding)',
                        data: benchVals,
                        borderColor: '#3B82F6',
                        borderDash: [5, 5],
                        borderWidth: 2,
                        fill: false,
                        pointRadius: 0,
                        pointHoverRadius: 5,
                        pointHoverBackgroundColor: '#3B82F6',
                        tension: 0.2
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    mode: 'index',
                    intersect: false
                },
                plugins: {
                    legend: {
                        position: 'top',
                        labels: {
                            font: { family: 'Inter', size: 12, weight: '600' },
                            usePointStyle: true,
                            boxWidth: 10,
                            padding: 15
                        }
                    },
                    tooltip: {
                        backgroundColor: 'rgba(15, 23, 42, 0.95)',
                        titleFont: { family: 'Inter', size: 13, weight: '700' },
                        bodyFont: { family: 'Inter', size: 12 },
                        padding: 12,
                        cornerRadius: 8,
                        displayColors: true,
                        callbacks: {
                            title: (tooltipItems) => `📅 Date: ${tooltipItems[0].label}`,
                            label: (ctx) => {
                                const val = ctx.parsed.y;
                                const pnl = val - startCapital;
                                const pnlPct = ((val - startCapital) / startCapital) * 100;
                                const sign = pnl >= 0 ? '+' : '';
                                return ` ${ctx.dataset.label}: ₹${val.toLocaleString('en-IN', { minimumFractionDigits: 2 })} (${sign}${pnlPct.toFixed(2)}%)`;
                            },
                            afterBody: (tooltipItems) => {
                                if (tooltipItems.length >= 2) {
                                    const stratP = tooltipItems[0].parsed.y;
                                    const benchP = tooltipItems[1].parsed.y;
                                    const delta = stratP - benchP;
                                    const sign = delta >= 0 ? '+' : '';
                                    return `\n⚡ AI Outperformance: ${sign}₹${delta.toLocaleString('en-IN', { minimumFractionDigits: 2 })}`;
                                }
                                return '';
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        grid: { display: false },
                        ticks: {
                            font: { family: 'Inter', size: 11 },
                            maxTicksLimit: 8,
                            color: '#64748B'
                        }
                    },
                    y: {
                        grid: { color: 'rgba(226, 232, 240, 0.6)' },
                        ticks: {
                            font: { family: 'Inter', size: 11 },
                            color: '#64748B',
                            callback: (v) => '₹' + (v >= 100000 ? (v / 1000).toFixed(0) + 'k' : v.toLocaleString('en-IN'))
                        }
                    }
                }
            }
        });
    }

    function renderTradeLogTable(tradeLog) {
        if (!backtestTradeBody) return;
        if (!tradeLog || tradeLog.length === 0) {
            backtestTradeBody.innerHTML = `<tr><td colspan="8" style="text-align: center; color: #94A3B8;">No trades executed during this period.</td></tr>`;
            return;
        }

        let htmlStr = '';
        tradeLog.slice().reverse().forEach(t => {
            let color = '#94A3B8';
            if (t.pnl > 0) color = '#10B981';
            else if (t.pnl < 0) color = '#EF4444';

            const nominalVal = t.nominal ? `₹${t.nominal.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}` : `₹${(t.shares * t.price).toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
            const tradeFeesStr = t.trade_fees !== undefined ? `₹${t.trade_fees.toFixed(2)}` : '₹0.00';
            const pnlStr = t.pnl !== 0 ? `${t.pnl >= 0 ? '+' : ''}₹${t.pnl.toFixed(2)}` : '-';
            const pnlPctStr = t.pnl_pct !== 0 ? `${t.pnl_pct >= 0 ? '+' : ''}${t.pnl_pct.toFixed(2)}%` : '-';

            htmlStr += `
                <tr>
                    <td>${t.date}</td>
                    <td><b>${t.action}</b></td>
                    <td>${nominalVal}</td>
                    <td>₹${t.price.toFixed(2)}</td>
                    <td>${t.shares}</td>
                    <td><span style="color: #64748B; font-weight: 500;">${tradeFeesStr}</span></td>
                    <td style="color: ${color}; font-weight: 600;">${pnlStr}</td>
                    <td style="color: ${color}; font-weight: 700;">${pnlPctStr}</td>
                </tr>
            `;
        });
        backtestTradeBody.innerHTML = htmlStr;
    }

    // 7. Update Prediction Info Section & Sentiment Overlay Breakdown
    function updatePredictionInfo(data) {
        infoStockEl.textContent = data.ticker || currentTicker;
        infoDateEl.textContent = data.prediction_date || isoDate;
        infoModelEl.textContent = data.model_name || data.model || 'LSTM';

        const techPriceEl = document.getElementById('info-tech-price');
        const sentAdjEl = document.getElementById('info-sent-adj');
        const finalCompEl = document.getElementById('info-final-comp');

        const techP = data.technical_baseline_price || data.predicted_price || 0.0;
        const finalP = data.predicted_price || 0.0;
        const sentAdjPct = data.sentiment_adjustment_pct || 0.0;
        const alphaP = data.alpha_parameter || 0.15;

        if (techPriceEl) {
            techPriceEl.textContent = `₹${techP.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
        }
        if (sentAdjEl) {
            const signStr = sentAdjPct >= 0 ? '+' : '';
            sentAdjEl.textContent = `${signStr}${sentAdjPct.toFixed(2)}%`;
            sentAdjEl.style.color = sentAdjPct >= 0 ? '#10B981' : '#EF4444';
        }
        if (finalCompEl) {
            finalCompEl.textContent = `₹${finalP.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
        }

        updateNewsSentimentInfo(data);
    }

    // 7.5 Update News & Sentiment Intelligence Section (Report 4)
    function updateNewsSentimentInfo(data) {
        const sentData = data.news_sentiment || {};
        const aiCompSig = data.ai_composite_signal || 'HOLD / NEUTRAL ⚖️';

        const summarySub = document.getElementById('sentiment-summary-sub');
        const badgeWrapper = document.getElementById('sentiment-badge-wrapper');
        const compSigEl = document.getElementById('sent-composite-sig');
        const scoreValEl = document.getElementById('sent-score-val');
        const sectorValEl = document.getElementById('sent-sector-val');
        const confValEl = document.getElementById('sent-conf-val');
        const sourcesBarEl = document.getElementById('sent-sources-bar');
        const headlinesContainer = document.getElementById('headlines-list-container');

        if (!summarySub || !compSigEl) return;

        const sentSummary = sentData.impact_summary || 'Neutral market sentiment impact.';
        const overallSent = (sentData.overall_sentiment || 'NEUTRAL').toUpperCase();
        const sentScore = sentData.sentiment_score || 0.0;
        const sentConf = sentData.confidence_pct || 50.0;

        const sectorInfo = sentData.sector_sentiment || {};
        const sectorName = sentData.sector_name || 'General Market';
        const sectorSent = (sectorInfo.sector_sentiment || 'NEUTRAL').toUpperCase();
        const sectorScore = sectorInfo.sector_score || 0.0;

        const sourcesList = sentData.sources_used || ['Google News', 'Yahoo Finance', 'Economic Times', 'Moneycontrol'];
        const headlines = sentData.headlines || [];

        summarySub.textContent = sentSummary;

        if (badgeWrapper) {
            let badgeClass = 'badge-neutral';
            if (overallSent === 'BULLISH') badgeClass = 'badge-bullish';
            else if (overallSent === 'BEARISH') badgeClass = 'badge-bearish';
            badgeWrapper.innerHTML = `<span class="${badgeClass}">🟢 ${overallSent}</span>`;
        }

        compSigEl.textContent = aiCompSig;
        const signStr = sentScore >= 0 ? '+' : '';
        scoreValEl.innerHTML = `${signStr}${sentScore.toFixed(2)} <span class="range-sub">(-1 to +1)</span>`;

        let sectorColor = '#94A3B8';
        if (sectorSent === 'BULLISH') sectorColor = '#10B981';
        else if (sectorSent === 'BEARISH') sectorColor = '#EF4444';

        sectorValEl.innerHTML = `${sectorName}<br><span class="sector-sub" style="color:${sectorColor}; font-weight:700;">${sectorSent} (${sectorScore >= 0 ? '+' : ''}${sectorScore.toFixed(2)})</span>`;
        confValEl.textContent = `${sentConf.toFixed(1)}%`;

        if (sourcesBarEl) {
            sourcesBarEl.innerHTML = `🔍 <strong>Sources Aggregated:</strong> ${sourcesList.join(' • ')}`;
        }

        if (headlinesContainer) {
            if (headlines.length === 0) {
                headlinesContainer.innerHTML = `<div class="headline-placeholder" style="color:#94A3B8; padding:10px;">No recent news headlines available for this ticker.</div>`;
                return;
            }

            let htmlStr = '';
            headlines.slice(0, 6).forEach(item => {
                const tSent = (item.sentiment || 'NEUTRAL').toUpperCase();
                let bBadge = '<span style="color:#94A3B8; font-weight:700; font-size:11px;">[NEUTRAL]</span>';
                if (tSent === 'BULLISH') bBadge = '<span style="color:#10B981; font-weight:700; font-size:11px;">[BULLISH]</span>';
                else if (tSent === 'BEARISH') bBadge = '<span style="color:#EF4444; font-weight:700; font-size:11px;">[BEARISH]</span>';

                htmlStr += `
                    <div class="headline-item">
                        <div class="headline-top">
                            <div>${bBadge} <span class="headline-title-text">${item.title}</span></div>
                        </div>
                        <div class="headline-meta">
                            <span>Source: <strong class="headline-source">${item.source || 'Financial News'}</strong></span>
                            <span>${item.date || ''}</span>
                        </div>
                    </div>
                `;
            });
            headlinesContainer.innerHTML = htmlStr;
        }
    }

    // 8. Render Prediction Breakdown Table View with 95% Confidence Bounds
    function renderPredictionTable() {
        if (!currentPredictionData || (!currentPredictionData.forecast_table && !currentPredictionData.prediction_dates)) {
            predTableBody.innerHTML = `<tr><td colspan="7" style="text-align:center;">No prediction data available.</td></tr>`;
            return;
        }

        if (predViewTitle) predViewTitle.textContent = `Detailed 7-Day Price Forecast — ${currentTicker}`;

        // If backend provided pre-computed forecast_table, use it directly
        if (currentPredictionData.forecast_table && currentPredictionData.forecast_table.length > 0) {
            let rowsHTML = '';
            currentPredictionData.forecast_table.forEach(item => {
                let colorHex = item.color === 'green' ? '#10B981' : (item.color === 'red' ? '#EF4444' : '#94A3B8');
                const upperVal = item.upper_bound ? `₹${item.upper_bound.toFixed(2)}` : 'N/A';
                const lowerVal = item.lower_bound ? `₹${item.lower_bound.toFixed(2)}` : 'N/A';
                rowsHTML += `
                    <tr>
                        <td><b>${item.day}</b></td>
                        <td>${item.date} (${item.full_date})</td>
                        <td><b>₹${item.predicted_price.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</b></td>
                        <td style="color: #059669; font-weight: 600;">${upperVal}</td>
                        <td style="color: #DC2626; font-weight: 600;">${lowerVal}</td>
                        <td style="color: ${colorHex}; font-weight: 600;">${item.price_change_str}</td>
                        <td style="color: ${colorHex}; font-weight: 700;">${item.expected_return_str}</td>
                    </tr>
                `;
            });
            predTableBody.innerHTML = rowsHTML;
            return;
        }

        // Fallback consecutive day calculation
        let prevP = currentPredictionData.current_price || 0.0;
        const dates = currentPredictionData.prediction_dates || [];
        const prices = currentPredictionData.predicted_prices || [];
        const uBounds = currentPredictionData.upper_bounds || [];
        const lBounds = currentPredictionData.lower_bounds || [];

        let rowsHTML = '';
        dates.forEach((dStr, idx) => {
            const p = prices[idx];
            const upperVal = uBounds[idx] ? `₹${uBounds[idx].toFixed(2)}` : 'N/A';
            const lowerVal = lBounds[idx] ? `₹${lBounds[idx].toFixed(2)}` : 'N/A';
            const diff = p - prevP;
            const retPct = prevP > 0 ? ((diff / prevP) * 100) : 0.0;
            const diffRound = Math.round(diff * 100) / 100;

            let color = '#94A3B8';
            let chgStr = `₹0.00 ⚪`;
            let retStr = `0.00% ⚪`;

            if (diffRound > 0) {
                color = '#10B981';
                chgStr = `+₹${diff.toFixed(2)} 🟢`;
                retStr = `+${retPct.toFixed(2)}% 🟢`;
            } else if (diffRound < 0) {
                color = '#EF4444';
                chgStr = `-₹${Math.abs(diff).toFixed(2)} 🔴`;
                retStr = `${retPct.toFixed(2)}% 🔴`;
            }

            rowsHTML += `
                <tr>
                    <td><b>Day ${idx + 1}</b></td>
                    <td>${dStr}</td>
                    <td><b>₹${p.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</b></td>
                    <td style="color: #059669; font-weight: 600;">${upperVal}</td>
                    <td style="color: #DC2626; font-weight: 600;">${lowerVal}</td>
                    <td style="color: ${color}; font-weight: 600;">${chgStr}</td>
                    <td style="color: ${color}; font-weight: 700;">${retStr}</td>
                </tr>
            `;
            prevP = p;
        });

        predTableBody.innerHTML = rowsHTML;
    }

    // 9. Render Historical Data Table View
    function renderHistoricalTable() {
        if (!rawHistoricalData || rawHistoricalData.length === 0) {
            histTableBody.innerHTML = `<tr><td colspan="2" style="text-align:center;">No historical data available.</td></tr>`;
            return;
        }

        if (histViewTitle) histViewTitle.textContent = `Historical Closing Prices — ${currentTicker}`;

        const recentSlice = rawHistoricalData.slice(-60).reverse();
        let rowsHTML = '';
        recentSlice.forEach(row => {
            rowsHTML += `
                <tr>
                    <td>${row.date}</td>
                    <td><b>₹${row.close.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</b></td>
                </tr>
            `;
        });

        histTableBody.innerHTML = rowsHTML;
    }

    // 10. Render Chart.js Chart with 95% Confidence Bounds
    function updateChartDisplay() {
        if (!rawHistoricalData || rawHistoricalData.length === 0) return;

        const chartTitleEl = document.getElementById('chart-stock-title');
        const currP = currentPredictionData ? (currentPredictionData.current_price || 0.0) : 0.0;
        if (chartTitleEl) {
            chartTitleEl.textContent = `— ${currentTicker} (₹${currP.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })})`;
        }

        const daysMap = { '7D': 7, '1M': 30, '3M': 90, '6M': 180, '1Y': 365 };
        const filterDays = daysMap[activeTimeframe] || 30;

        const slicedHist = rawHistoricalData.slice(-filterDays);
        const labels = slicedHist.map(d => d.date);
        const actualPrices = slicedHist.map(d => d.close);

        const predictedDatasetPoints = new Array(actualPrices.length).fill(null);
        const upperBandPoints = new Array(actualPrices.length).fill(null);
        const lowerBandPoints = new Array(actualPrices.length).fill(null);

        if (actualPrices.length > 0) {
            const lastVal = actualPrices[actualPrices.length - 1];
            predictedDatasetPoints[actualPrices.length - 1] = lastVal;
            upperBandPoints[actualPrices.length - 1] = lastVal;
            lowerBandPoints[actualPrices.length - 1] = lastVal;
        }

        if (currentPredictionData && currentPredictionData.prediction_dates) {
            const uBounds = currentPredictionData.upper_bounds || [];
            const lBounds = currentPredictionData.lower_bounds || [];

            currentPredictionData.prediction_dates.forEach((dStr, idx) => {
                labels.push(dStr);
                actualPrices.push(null);
                predictedDatasetPoints.push(currentPredictionData.predicted_prices[idx]);
                upperBandPoints.push(uBounds[idx] || currentPredictionData.predicted_prices[idx]);
                lowerBandPoints.push(lBounds[idx] || currentPredictionData.predicted_prices[idx]);
            });
        }

        renderChart(labels, actualPrices, predictedDatasetPoints, upperBandPoints, lowerBandPoints);
    }

    function renderChart(labels, actualData, predictedData, upperBandData = [], lowerBandData = []) {
        const ctx = document.getElementById('mainChart').getContext('2d');

        if (mainChart) {
            mainChart.destroy();
        }

        mainChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'Actual Price',
                        data: actualData,
                        borderColor: '#2563EB',
                        backgroundColor: 'rgba(37, 99, 235, 0.08)',
                        fill: true,
                        borderWidth: 2.5,
                        pointRadius: 3,
                        pointHoverRadius: 6,
                        tension: 0.2
                    },
                    {
                        label: 'Predicted Price',
                        data: predictedData,
                        borderColor: '#10B981',
                        borderDash: [6, 4],
                        borderWidth: 2.5,
                        pointRadius: 4,
                        pointBackgroundColor: '#10B981',
                        pointHoverRadius: 7,
                        tension: 0.2
                    },
                    {
                        label: '95% Upper Bound',
                        data: upperBandData,
                        borderColor: 'rgba(16, 185, 129, 0.3)',
                        borderWidth: 1,
                        pointRadius: 0,
                        fill: false,
                        tension: 0.2
                    },
                    {
                        label: '95% Lower Bound',
                        data: lowerBandData,
                        borderColor: 'rgba(16, 185, 129, 0.3)',
                        backgroundColor: 'rgba(16, 185, 129, 0.12)',
                        borderWidth: 1,
                        pointRadius: 0,
                        fill: '-1', // Fill area between lower bound and upper bound
                        tension: 0.2
                    }
                ]
            },
            options: {
                animation: {
                    duration: 650,
                    easing: 'easeOutQuart'
                },
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    mode: 'index',
                    intersect: false,
                },
                plugins: {
                    legend: {
                        position: 'top',
                        labels: {
                            font: { family: 'Inter', size: 13, weight: '600' },
                            usePointStyle: true,
                            boxWidth: 8,
                            filter: function(item) {
                                return !item.text.includes('Bound');
                            }
                        }
                    },
                    tooltip: {
                        callbacks: {
                            label: function (context) {
                                let label = context.dataset.label || '';
                                if (label) label += ': ';
                                if (context.parsed.y !== null) {
                                    label += '₹' + context.parsed.y.toLocaleString('en-IN', { minimumFractionDigits: 2 });
                                }
                                return label;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        grid: { display: false },
                        ticks: { font: { family: 'Inter', size: 11 }, maxTicksLimit: 10 }
                    },
                    y: {
                        grid: { color: '#E2E8F0' },
                        ticks: {
                            font: { family: 'Inter', size: 11 },
                            callback: function (value) {
                                return '₹' + value.toLocaleString('en-IN');
                            }
                        }
                    }
                }
            }
        });
    }

    /* ==========================================================================
       8. LIVE STOCK MARKET VISUALIZATION TERMINAL CONTROLLER
       ========================================================================== */
    // (visualizationData, activeVizPeriod, vizChartType, etc. initialized at top of DOMContentLoaded)

    // Attach Event Listeners for Live Visualization Toolbar
    const vizTfButtons = document.querySelectorAll('#viz-timeframe-buttons .tf-btn');
    vizTfButtons.forEach(btn => {
        btn.addEventListener('click', (e) => {
            vizTfButtons.forEach(b => b.classList.remove('active'));
            e.target.classList.add('active');
            activeVizPeriod = e.target.getAttribute('data-period');
            loadVisualizationTerminal(currentTicker, activeVizPeriod);
        });
    });

    const btnTradingView = document.getElementById('btn-chart-tradingview');
    const btnCandlestick = document.getElementById('btn-chart-candlestick');
    const btnLine = document.getElementById('btn-chart-line');
    const tvContainer = document.getElementById('tvLightweightChartContainer');
    const plotlyContainer = document.getElementById('plotlyCandlestickChart');

    if (btnTradingView && btnCandlestick && btnLine) {
        btnTradingView.addEventListener('click', () => {
            btnTradingView.classList.add('active');
            btnCandlestick.classList.remove('active');
            btnLine.classList.remove('active');
            vizChartType = 'tradingview';
            if (tvContainer) tvContainer.style.display = 'block';
            if (plotlyContainer) plotlyContainer.style.display = 'none';
            if (visualizationData && window.TradingViewChartManager) {
                window.TradingViewChartManager.renderData(visualizationData, 'candlestick');
            }
        });

        btnCandlestick.addEventListener('click', () => {
            btnCandlestick.classList.add('active');
            btnTradingView.classList.remove('active');
            btnLine.classList.remove('active');
            vizChartType = 'candlestick';
            if (tvContainer) tvContainer.style.display = 'none';
            if (plotlyContainer) plotlyContainer.style.display = 'block';
            if (visualizationData) renderPlotlyCandlestickChart(visualizationData);
        });

        btnLine.addEventListener('click', () => {
            btnLine.classList.add('active');
            btnTradingView.classList.remove('active');
            btnCandlestick.classList.remove('active');
            vizChartType = 'line';
            if (tvContainer) tvContainer.style.display = 'block';
            if (plotlyContainer) plotlyContainer.style.display = 'none';
            if (visualizationData && window.TradingViewChartManager) {
                window.TradingViewChartManager.renderData(visualizationData, 'line');
            } else if (visualizationData) {
                renderPlotlyCandlestickChart(visualizationData);
            }
        });
    }

    // Overlay Toggles
    const overlayIds = ['chk-sma20', 'chk-sma50', 'chk-sma200', 'chk-ema20', 'chk-ema50', 'chk-bollinger', 'chk-ai-forecast'];
    overlayIds.forEach(id => {
        const el = document.getElementById(id);
        if (el) {
            el.addEventListener('change', () => {
                if (visualizationData) {
                    if (vizChartType === 'tradingview' && window.TradingViewChartManager) {
                        window.TradingViewChartManager.renderData(visualizationData, 'candlestick');
                    } else {
                        renderPlotlyCandlestickChart(visualizationData);
                    }
                }
            });
        }
    });

    // Auto-Refresh Selector
    const refreshIntervalSelect = document.getElementById('refresh-interval-select');
    if (refreshIntervalSelect) {
        refreshIntervalSelect.addEventListener('change', (e) => {
            vizAutoRefreshInterval = parseInt(e.target.value, 10);
            setupAutoRefreshTimer();
        });
    }

    const manualRefreshBtn = document.getElementById('manual-refresh-btn');
    if (manualRefreshBtn) {
        manualRefreshBtn.addEventListener('click', () => {
            loadVisualizationTerminal(currentTicker, activeVizPeriod);
        });
    }

    function setupAutoRefreshTimer() {
        if (vizAutoRefreshTimer) {
            clearInterval(vizAutoRefreshTimer);
            vizAutoRefreshTimer = null;
        }
        if (vizAutoRefreshInterval > 0) {
            vizAutoRefreshTimer = setInterval(() => {
                loadVisualizationTerminal(currentTicker, activeVizPeriod);
            }, vizAutoRefreshInterval);
        }
    }

    // ── Price & Signal Alert Handlers ─────────────────────────────────────
    const testAlertBtn = document.getElementById('test-alert-btn');
    const alertChannelSelect = document.getElementById('alert-channel-select');
    const alertEmailInput = document.getElementById('alert-email-input');
    const alertTelegramInput = document.getElementById('alert-telegram-input');

    if (addAlertBtn) {
        addAlertBtn.addEventListener('click', (e) => {
            e.preventDefault();
            const ticker = (alertTickerInput && alertTickerInput.value) ? alertTickerInput.value.trim().toUpperCase() : 'RELIANCE.NS';
            const condType = alertTypeSelect ? alertTypeSelect.value : 'PRICE_ABOVE';
            const targetVal = alertTargetInput ? parseFloat(alertTargetInput.value) : 0.0;
            const channel = alertChannelSelect ? alertChannelSelect.value : 'BOTH';
            const email = alertEmailInput ? alertEmailInput.value.trim() : '';
            const telegramId = alertTelegramInput ? alertTelegramInput.value.trim() : '';
            const notes = alertNotesInput ? alertNotesInput.value.trim() : '';

            if (!ticker) {
                alert('Please enter a valid stock ticker symbol.');
                return;
            }

            addAlertBtn.disabled = true;
            addAlertBtn.innerHTML = '⏳ SAVING...';

            fetch('/api/user/alerts', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    ticker: ticker,
                    condition_type: condType,
                    target_value: targetVal,
                    channel: channel,
                    email: email,
                    telegram_chat_id: telegramId,
                    notes: notes
                })
            })
            .then(res => res.json())
            .then(data => {
                addAlertBtn.disabled = false;
                addAlertBtn.innerHTML = '➕ SAVE ALERT RULE';
                if (data.status === 'success') {
                    alert('✅ Alert rule saved successfully!');
                    loadUserAlerts();
                } else {
                    alert('⚠️ Alert creation failed: ' + (data.message || 'Unknown error'));
                }
            })
            .catch(err => {
                addAlertBtn.disabled = false;
                addAlertBtn.innerHTML = '➕ SAVE ALERT RULE';
                alert('❌ Connection error saving alert: ' + err.message);
            });
        });
    }

    if (testAlertBtn) {
        testAlertBtn.addEventListener('click', (e) => {
            e.preventDefault();
            const channel = alertChannelSelect ? alertChannelSelect.value : 'BOTH';
            const email = alertEmailInput ? alertEmailInput.value.trim() : '';
            const telegramId = alertTelegramInput ? alertTelegramInput.value.trim() : '';

            testAlertBtn.disabled = true;
            testAlertBtn.innerHTML = '⚡ SENDING...';

            fetch('/api/user/alerts/test', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    channel: channel,
                    email: email,
                    telegram_chat_id: telegramId,
                    title: 'Test Notification Alert',
                    body: 'This is a test notification from your Enterprise NSE Quantitative Trading Cockpit.'
                })
            })
            .then(res => res.json())
            .then(data => {
                testAlertBtn.disabled = false;
                testAlertBtn.innerHTML = '⚡ TEST DISPATCH ALERT';
                if (data.status === 'success') {
                    alert('🚀 Test alert notification dispatched successfully via ' + channel + '!');
                } else {
                    alert('⚠️ Alert test dispatch returned status: ' + (data.message || 'Check server logs'));
                }
            })
            .catch(err => {
                testAlertBtn.disabled = false;
                testAlertBtn.innerHTML = '⚡ TEST DISPATCH ALERT';
                alert('❌ Error sending test alert: ' + err.message);
            });
        });
    }

    function loadUserAlerts() {
        const tbody = document.getElementById('alerts-table-body');
        if (!tbody) return;

        fetch('/api/user/alerts')
            .then(res => res.json())
            .then(data => {
                if (data.status === 'success' && Array.isArray(data.alerts)) {
                    if (data.alerts.length === 0) {
                        tbody.innerHTML = '<tr><td colspan="7" style="text-align: center; color: #94A3B8; padding: 20px;">No active alert rules configured. Create one above!</td></tr>';
                        return;
                    }
                    tbody.innerHTML = data.alerts.map(a => `
                        <tr>
                            <td><code>ALT-${String(a.id).padStart(4, '0')}</code></td>
                            <td><strong>${a.ticker}</strong></td>
                            <td><span class="badge badge-buy">${a.condition_type}</span></td>
                            <td>₹${parseFloat(a.target_value).toFixed(2)}</td>
                            <td><span class="badge ${a.triggered ? 'badge-sell' : 'badge-buy'}">${a.triggered ? 'TRIGGERED' : 'ACTIVE'}</span></td>
                            <td>${a.created_at || 'Just now'}</td>
                            <td>
                                <button class="btn btn-secondary btn-sm" onclick="deleteUserAlert(${a.id})" style="padding: 4px 8px; font-size: 11px; color: #EF4444; border-color: #FCA5A5;">
                                    🗑️ DELETE
                                </button>
                            </td>
                        </tr>
                    `).join('');
                }
            })
            .catch(err => {
                console.error('Error fetching alerts:', err);
            });
    }

    window.deleteUserAlert = function(alertId) {
        if (!confirm('Are you sure you want to delete this alert rule?')) return;
        fetch(`/api/user/alerts/${alertId}`, { method: 'DELETE' })
            .then(res => res.json())
            .then(data => {
                if (data.status === 'success') {
                    loadUserAlerts();
                } else {
                    alert('Failed to delete alert: ' + data.message);
                }
            });
    };

    // ── Live Broker Integration Handlers ──────────────────────────────────
    const modePaperBtn = document.getElementById('mode-paper-btn');
    const modeBrokerBtn = document.getElementById('mode-broker-btn');
    const brokerConfigCard = document.getElementById('broker-config-card');
    const connectBrokerBtn = document.getElementById('connect-broker-btn');
    const brokerSelect = document.getElementById('broker-select');
    const brokerApiKeyInput = document.getElementById('broker-api-key');
    const brokerSecretInput = document.getElementById('broker-secret');
    const brokerTokenInput = document.getElementById('broker-token');
    const brokerConnBadge = document.getElementById('broker-conn-badge');

    if (modePaperBtn && modeBrokerBtn) {
        modePaperBtn.addEventListener('click', () => {
            modePaperBtn.classList.add('active');
            modeBrokerBtn.classList.remove('active');
            if (brokerConfigCard) brokerConfigCard.style.display = 'none';
            switchBrokerMode('PAPER');
        });

        modeBrokerBtn.addEventListener('click', () => {
            modeBrokerBtn.classList.add('active');
            modePaperBtn.classList.remove('active');
            if (brokerConfigCard) brokerConfigCard.style.display = 'block';
            const bName = brokerSelect ? brokerSelect.value : 'ZERODHA';
            switchBrokerMode(bName);
        });
    }

    if (brokerSelect) {
        brokerSelect.addEventListener('change', (e) => {
            if (modeBrokerBtn && modeBrokerBtn.classList.contains('active')) {
                switchBrokerMode(e.target.value);
            }
        });
    }

    function switchBrokerMode(mode) {
        fetch('/api/broker/mode', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ mode: mode })
        })
        .then(res => res.json())
        .then(data => {
            const paperBadge = document.querySelector('#paper-order-form .badge');
            const submitBtn = document.getElementById('paper-order-submit-btn');
            if (mode === 'PAPER') {
                if (paperBadge) {
                    paperBadge.textContent = 'PAPER ENGINE';
                    paperBadge.style.background = '#DBEAFE';
                    paperBadge.style.color = '#1E40AF';
                }
                if (submitBtn) submitBtn.innerHTML = 'EXECUTE BUY ORDER (PAPER)';
            } else {
                if (paperBadge) {
                    paperBadge.textContent = 'LIVE BROKER: ' + mode;
                    paperBadge.style.background = '#FEE2E2';
                    paperBadge.style.color = '#991B1B';
                }
                if (submitBtn) submitBtn.innerHTML = `EXECUTE ${mode} REAL ORDER 🔴`;
            }
        });
    }

    if (connectBrokerBtn) {
        connectBrokerBtn.addEventListener('click', (e) => {
            e.preventDefault();
            const bName = brokerSelect ? brokerSelect.value : 'ZERODHA';
            const apiKey = brokerApiKeyInput ? brokerApiKeyInput.value.trim() : '';
            const secret = brokerSecretInput ? brokerSecretInput.value.trim() : '';
            const token = brokerTokenInput ? brokerTokenInput.value.trim() : '';

            connectBrokerBtn.disabled = true;
            connectBrokerBtn.innerHTML = '⏳ CONNECTING...';

            fetch('/api/broker/config', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    broker: bName,
                    api_key: apiKey,
                    api_secret: secret,
                    access_token: token,
                    client_code: secret
                })
            })
            .then(res => res.json())
            .then(data => {
                connectBrokerBtn.disabled = false;
                connectBrokerBtn.innerHTML = '🔌 CONNECT BROKER';
                if (data.connected) {
                    if (brokerConnBadge) {
                        brokerConnBadge.textContent = 'CONNECTED 🟢';
                        brokerConnBadge.style.background = '#D1FAE5';
                        brokerConnBadge.style.color = '#065F46';
                    }
                    alert(`✅ ${bName} API connected successfully! Real money orders will now route to ${bName}.`);
                } else {
                    if (brokerConnBadge) {
                        brokerConnBadge.textContent = 'DISCONNECTED 🔴';
                        brokerConnBadge.style.background = '#FEE2E2';
                        brokerConnBadge.style.color = '#991B1B';
                    }
                    alert(`⚠️ ${data.message || 'Broker connection failed. Check API key & credentials.'}`);
                }
            })
            .catch(err => {
                connectBrokerBtn.disabled = false;
                connectBrokerBtn.innerHTML = '🔌 CONNECT BROKER';
                alert('❌ Broker connection error: ' + err.message);
            });
        });
    }

    let liveTickStreamTimer = null;

    function startLiveTickStream(ticker) {
        if (liveTickStreamTimer) {
            clearInterval(liveTickStreamTimer);
            liveTickStreamTimer = null;
        }

        if (!ticker) return;
        const cleanT = ticker.trim().toUpperCase();

        // High frequency poll every 3 seconds for live streaming market ticks
        liveTickStreamTimer = setInterval(() => {
            const url = `/api/stock/${encodeURIComponent(cleanT)}/visualization?period=1d&include_prediction=false`;
            fetch(url)
                .then(res => res.json())
                .then(data => {
                    if (data.status === 'success' && data.live_quote) {
                        const lq = data.live_quote;
                        const ltp = Number(lq.ltp || 0);

                        if (ltp > 0) {
                            renderLiveStockHeader(data);

                            const todayStr = new Date().toISOString().split('T')[0];
                            const tickPayload = {
                                time: todayStr,
                                ltp: ltp,
                                open: Number(lq.open || ltp),
                                high: Number(lq.high || ltp),
                                low: Number(lq.low || ltp),
                                volume: Number(lq.volume || 0)
                            };

                            if (window.TradingViewChartManager && typeof window.TradingViewChartManager.updateLiveTick === 'function') {
                                window.TradingViewChartManager.updateLiveTick(tickPayload);
                            }
                        }
                    }
                })
                .catch(err => {
                    console.debug('Live tick stream polling skipped:', err);
                });
        }, 3000);
    }

    function loadVisualizationTerminal(ticker, period = null) {
        if (!ticker) return;
        if (!period) period = (typeof activeVizPeriod !== 'undefined' && activeVizPeriod) ? activeVizPeriod : '1y';
        const cleanT = ticker.trim().toUpperCase();
        
        const liveStockSymbolEl = document.getElementById('live-stock-symbol');
        if (liveStockSymbolEl) liveStockSymbolEl.textContent = cleanT;
        
        const url = `/api/stock/${encodeURIComponent(cleanT)}/visualization?period=${encodeURIComponent(period)}&include_prediction=true`;
        
        fetch(url)
            .then(res => res.json())
            .then(data => {
                if (data.status === 'success') {
                    visualizationData = data;
                    renderLiveStockHeader(data);

                    if (window.TradingViewChartManager && (vizChartType === 'tradingview' || vizChartType === 'line')) {
                        if (tvContainer) tvContainer.style.display = 'block';
                        if (plotlyContainer) plotlyContainer.style.display = 'none';
                        const chartMode = vizChartType === 'line' ? 'line' : 'candlestick';
                        window.TradingViewChartManager.renderData(data, chartMode);
                    } else {
                        if (tvContainer) tvContainer.style.display = 'none';
                        if (plotlyContainer) plotlyContainer.style.display = 'block';
                        renderPlotlyCandlestickChart(data);
                    }

                    renderPlotlyRSIChart(data);
                    renderPlotlyMACDChart(data);
                    renderAnalyticsSummary(data);

                    // Start high-frequency live tick poller
                    startLiveTickStream(cleanT);
                } else {
                    console.warn('Visualization API error:', data.message);
                }
            })
            .catch(err => {
                console.error('Failed to load visualization data:', err);
            });
    }


    let lastRecordedLTP = null;

    function renderLiveStockHeader(data) {
        const lq = data.live_quote || {};
        const ltpValEl = document.getElementById('live-ltp-val');
        const changeTagEl = document.getElementById('live-change-tag');
        const companyNameEl = document.getElementById('live-company-name');
        const sourceBadgeEl = document.getElementById('live-data-source-badge');
        const statusBadgeEl = document.getElementById('live-market-status-badge');
        const lastUpdatedEl = document.getElementById('live-last-updated');

        const openValEl = document.getElementById('live-open-val');
        const highValEl = document.getElementById('live-high-val');
        const lowValEl = document.getElementById('live-low-val');
        const prevCloseValEl = document.getElementById('live-prev-close-val');
        const volumeValEl = document.getElementById('live-volume-val');

        const ltp = lq.ltp || (data.ohlcv_series && data.ohlcv_series.length ? data.ohlcv_series[data.ohlcv_series.length - 1].close : 0);
        const change = lq.change || 0;
        const changePct = lq.change_percent || 0;

        if (ltpValEl) {
            ltpValEl.textContent = ltp.toLocaleString('en-IN', { minimumFractionDigits: 2 });
            
            // Trigger Live Price Pulse Animation (Option C)
            if (lastRecordedLTP !== null && ltp !== lastRecordedLTP) {
                const flashClass = ltp > lastRecordedLTP ? 'price-flash-up' : 'price-flash-down';
                ltpValEl.classList.remove('price-flash-up', 'price-flash-down');
                // Force reflow for animation restart
                void ltpValEl.offsetWidth;
                ltpValEl.classList.add(flashClass);
                setTimeout(() => {
                    ltpValEl.classList.remove('price-flash-up', 'price-flash-down');
                }, 850);
            }
            lastRecordedLTP = ltp;
        }

        if (changeTagEl) {
            const sign = change >= 0 ? '+' : '';
            changeTagEl.textContent = `${sign}${change.toFixed(2)} (${sign}${changePct.toFixed(2)}%)`;
            changeTagEl.className = change >= 0 ? 'change-tag positive' : 'change-tag negative';
        }

        if (companyNameEl) companyNameEl.textContent = lq.company_name || data.ticker;
        if (sourceBadgeEl) sourceBadgeEl.textContent = data.data_source || 'yfinance';
        
        if (statusBadgeEl) {
            const mktOpen = data.market_status ? data.market_status.is_open : true;
            statusBadgeEl.textContent = mktOpen ? 'OPEN 🟢' : 'CLOSED 🔴';
            statusBadgeEl.className = mktOpen ? 'badge market-status-badge' : 'badge market-status-badge closed';
        }

        if (openValEl) openValEl.textContent = '₹' + (lq.open || (data.ohlcv_series && data.ohlcv_series.length ? data.ohlcv_series[data.ohlcv_series.length - 1].open : ltp)).toLocaleString('en-IN');
        if (highValEl) highValEl.textContent = '₹' + (lq.high || (data.ohlcv_series && data.ohlcv_series.length ? data.ohlcv_series[data.ohlcv_series.length - 1].high : ltp)).toLocaleString('en-IN');
        if (lowValEl) lowValEl.textContent = '₹' + (lq.low || (data.ohlcv_series && data.ohlcv_series.length ? data.ohlcv_series[data.ohlcv_series.length - 1].low : ltp)).toLocaleString('en-IN');
        if (prevCloseValEl) prevCloseValEl.textContent = '₹' + (lq.previous_close || ltp).toLocaleString('en-IN');
        if (volumeValEl) volumeValEl.textContent = (lq.volume || (data.volume_summary ? data.volume_summary.current_volume : 0)).toLocaleString('en-IN');

        if (lastUpdatedEl) {
            const now = new Date();
            lastUpdatedEl.textContent = 'Updated: ' + now.toLocaleTimeString('en-US', { hour12: false }) + ' IST';
        }
    }

    function renderPlotlyCandlestickChart(data) {
        const container = document.getElementById('plotlyCandlestickChart');
        if (!container || !data.ohlcv_series || data.ohlcv_series.length === 0) return;

        const series = data.ohlcv_series;
        const dates = series.map(s => s.date);
        const opens = series.map(s => s.open);
        const highs = series.map(s => s.high);
        const lows = series.map(s => s.low);
        const closes = series.map(s => s.close);
        const volumes = series.map(s => s.volume);

        const traces = [];

        // 1. Main Price Series (Candlestick or Line)
        if (vizChartType === 'candlestick') {
            traces.push({
                x: dates,
                open: opens,
                high: highs,
                low: lows,
                close: closes,
                type: 'candlestick',
                name: data.ticker,
                increasing: { line: { color: '#34D399', width: 1.5 }, fillcolor: '#10B981' },
                decreasing: { line: { color: '#F87171', width: 1.5 }, fillcolor: '#EF4444' },
                yaxis: 'y'
            });
        } else {
            traces.push({
                x: dates,
                y: closes,
                type: 'scatter',
                mode: 'lines',
                name: `${data.ticker} Close`,
                line: { color: '#38BDF8', width: 2 },
                yaxis: 'y'
            });
        }

        // 2. Volume Sub-chart
        const volColors = series.map(s => s.close >= s.open ? 'rgba(52, 211, 153, 0.4)' : 'rgba(248, 113, 113, 0.4)');
        traces.push({
            x: dates,
            y: volumes,
            type: 'bar',
            name: 'Volume',
            marker: { color: volColors },
            yaxis: 'y2'
        });

        // 3. Technical Indicator Overlays
        const chkSma20 = document.getElementById('chk-sma20');
        if (chkSma20 && chkSma20.checked) {
            traces.push({
                x: dates,
                y: series.map(s => s.sma20),
                type: 'scatter',
                mode: 'lines',
                name: 'SMA 20',
                line: { color: '#3B82F6', width: 1.5 },
                yaxis: 'y'
            });
        }

        const chkSma50 = document.getElementById('chk-sma50');
        if (chkSma50 && chkSma50.checked) {
            traces.push({
                x: dates,
                y: series.map(s => s.sma50),
                type: 'scatter',
                mode: 'lines',
                name: 'SMA 50',
                line: { color: '#F97316', width: 1.5 },
                yaxis: 'y'
            });
        }

        const chkSma200 = document.getElementById('chk-sma200');
        if (chkSma200 && chkSma200.checked) {
            traces.push({
                x: dates,
                y: series.map(s => s.sma200),
                type: 'scatter',
                mode: 'lines',
                name: 'SMA 200',
                line: { color: '#A855F7', width: 1.5 },
                yaxis: 'y'
            });
        }

        const chkEma20 = document.getElementById('chk-ema20');
        if (chkEma20 && chkEma20.checked) {
            traces.push({
                x: dates,
                y: series.map(s => s.ema20),
                type: 'scatter',
                mode: 'lines',
                name: 'EMA 20',
                line: { color: '#14B8A6', width: 1.5 },
                yaxis: 'y'
            });
        }

        const chkEma50 = document.getElementById('chk-ema50');
        if (chkEma50 && chkEma50.checked) {
            traces.push({
                x: dates,
                y: series.map(s => s.ema50),
                type: 'scatter',
                mode: 'lines',
                name: 'EMA 50',
                line: { color: '#EC4899', width: 1.5 },
                yaxis: 'y'
            });
        }

        const chkBollinger = document.getElementById('chk-bollinger');
        if (chkBollinger && chkBollinger.checked) {
            traces.push({
                x: dates,
                y: series.map(s => s.bb_upper),
                type: 'scatter',
                mode: 'lines',
                name: 'Bollinger Upper',
                line: { color: 'rgba(148, 163, 184, 0.5)', width: 1, dash: 'dot' },
                yaxis: 'y'
            });
            traces.push({
                x: dates,
                y: series.map(s => s.bb_lower),
                type: 'scatter',
                mode: 'lines',
                name: 'Bollinger Lower',
                line: { color: 'rgba(148, 163, 184, 0.5)', width: 1, dash: 'dot' },
                fill: 'tonexty',
                fillcolor: 'rgba(148, 163, 184, 0.08)',
                yaxis: 'y'
            });
        }

        // 4. AI Forecast Overlay
        const chkAIForecast = document.getElementById('chk-ai-forecast');
        if (chkAIForecast && chkAIForecast.checked && data.ai_forecast) {
            const fc = data.ai_forecast;
            if (fc.prediction_dates && fc.predicted_prices && fc.prediction_dates.length > 0) {
                const lastActualDate = dates[dates.length - 1];
                const lastActualClose = closes[closes.length - 1];

                const fcDates = [lastActualDate, ...fc.prediction_dates];
                const fcPrices = [lastActualClose, ...fc.predicted_prices];
                const fcUpper = [lastActualClose, ...(fc.upper_bounds || fc.predicted_prices)];
                const fcLower = [lastActualClose, ...(fc.lower_bounds || fc.predicted_prices)];

                traces.push({
                    x: fcDates,
                    y: fcPrices,
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: '🔮 AI Forecast',
                    line: { color: '#10B981', width: 2.5, dash: 'dash' },
                    marker: { size: 6, color: '#10B981' },
                    yaxis: 'y'
                });

                if (fc.upper_bounds && fc.lower_bounds) {
                    traces.push({
                        x: fcDates,
                        y: fcUpper,
                        type: 'scatter',
                        mode: 'lines',
                        name: '95% Upper Bound',
                        line: { color: 'rgba(16, 185, 129, 0.3)', width: 1 },
                        yaxis: 'y',
                        showlegend: false
                    });
                    traces.push({
                        x: fcDates,
                        y: fcLower,
                        type: 'scatter',
                        mode: 'lines',
                        name: '95% Confidence Band',
                        line: { color: 'rgba(16, 185, 129, 0.3)', width: 1 },
                        fill: 'tonexty',
                        fillcolor: 'rgba(16, 185, 129, 0.12)',
                        yaxis: 'y'
                    });
                }
            }
        }

        const layout = {
            paper_bgcolor: '#0B1120',
            plot_bgcolor: '#0F172A',
            autosize: true,
            font: { family: 'Inter', color: '#94A3B8', size: 11 },
            margin: { l: 50, r: 20, t: 30, b: 30 },
            showlegend: true,
            legend: {
                orientation: 'h',
                x: 0,
                y: 1.12,
                font: { color: '#CBD5E1', size: 11 }
            },
            xaxis: {
                gridcolor: '#1E293B',
                rangeslider: { visible: false },
                type: 'date'
            },
            yaxis: {
                title: 'Price (₹)',
                gridcolor: '#1E293B',
                zerolinecolor: '#334155',
                domain: [0.25, 1.0],
                autorange: true
            },
            yaxis2: {
                title: 'Vol',
                gridcolor: '#1E293B',
                domain: [0, 0.20],
                showgrid: false,
                autorange: true
            }
        };

        const config = { responsive: true, displayModeBar: false };
        if (typeof Plotly !== 'undefined') {
            Plotly.react(container, traces, layout, config).then(() => {
                setTimeout(() => { if (typeof Plotly !== 'undefined') Plotly.Plots.resize(container); }, 100);
            });
        }
    }

    function renderPlotlyRSIChart(data) {
        const container = document.getElementById('plotlyRSIChart');
        if (!container || !data.ohlcv_series) return;

        const series = data.ohlcv_series;
        const dates = series.map(s => s.date);
        const rsiVals = series.map(s => s.rsi);

        const traces = [
            {
                x: dates,
                y: rsiVals,
                type: 'scatter',
                mode: 'lines',
                name: 'RSI (14)',
                line: { color: '#38BDF8', width: 2 }
            },
            {
                x: [dates[0], dates[dates.length - 1]],
                y: [70, 70],
                type: 'scatter',
                mode: 'lines',
                name: 'Overbought (70)',
                line: { color: '#EF4444', width: 1, dash: 'dot' },
                showlegend: false
            },
            {
                x: [dates[0], dates[dates.length - 1]],
                y: [50, 50],
                type: 'scatter',
                mode: 'lines',
                name: 'Neutral (50)',
                line: { color: '#94A3B8', width: 1, dash: 'dash' },
                showlegend: false
            },
            {
                x: [dates[0], dates[dates.length - 1]],
                y: [30, 30],
                type: 'scatter',
                mode: 'lines',
                name: 'Oversold (30)',
                line: { color: '#10B981', width: 1, dash: 'dot' },
                showlegend: false
            }
        ];

        const layout = {
            paper_bgcolor: '#1E293B',
            plot_bgcolor: '#0F172A',
            font: { family: 'Inter', color: '#94A3B8', size: 10 },
            margin: { l: 40, r: 15, t: 10, b: 25 },
            showlegend: false,
            xaxis: { gridcolor: '#1E293B' },
            yaxis: { range: [0, 100], gridcolor: '#1E293B', zeroline: false }
        };

        if (typeof Plotly !== 'undefined') {
            Plotly.react(container, traces, layout, { responsive: true, displayModeBar: false });
        }

        const rsiBadge = document.getElementById('rsi-status-badge');
        if (rsiBadge && data.analytics_summary) {
            const rsiVal = data.analytics_summary.rsi_current || 50;
            const rsiStatus = data.analytics_summary.rsi_status || 'Neutral';
            rsiBadge.textContent = `${rsiStatus} (${rsiVal.toFixed(1)})`;
            rsiBadge.className = rsiStatus === 'Overbought' ? 'badge badge-sell' : (rsiStatus === 'Oversold' ? 'badge badge-buy' : 'badge badge-hold');
        }
    }

    function renderPlotlyMACDChart(data) {
        const container = document.getElementById('plotlyMACDChart');
        if (!container || !data.ohlcv_series) return;

        const series = data.ohlcv_series;
        const dates = series.map(s => s.date);
        const macd = series.map(s => s.macd);
        const sig = series.map(s => s.macd_signal);
        const hist = series.map(s => s.macd_hist);

        const histColors = hist.map(h => (h || 0) >= 0 ? 'rgba(52, 211, 153, 0.5)' : 'rgba(248, 113, 113, 0.5)');

        const traces = [
            {
                x: dates,
                y: macd,
                type: 'scatter',
                mode: 'lines',
                name: 'MACD',
                line: { color: '#3B82F6', width: 1.5 }
            },
            {
                x: dates,
                y: sig,
                type: 'scatter',
                mode: 'lines',
                name: 'Signal',
                line: { color: '#F97316', width: 1.5 }
            },
            {
                x: dates,
                y: hist,
                type: 'bar',
                name: 'Histogram',
                marker: { color: histColors }
            }
        ];

        const layout = {
            paper_bgcolor: '#1E293B',
            plot_bgcolor: '#0F172A',
            font: { family: 'Inter', color: '#94A3B8', size: 10 },
            margin: { l: 40, r: 15, t: 10, b: 25 },
            showlegend: false,
            xaxis: { gridcolor: '#1E293B' },
            yaxis: { gridcolor: '#1E293B', zerolinecolor: '#334155' }
        };

        if (typeof Plotly !== 'undefined') {
            Plotly.react(container, traces, layout, { responsive: true, displayModeBar: false });
        }

        const macdBadge = document.getElementById('macd-status-badge');
        if (macdBadge && data.analytics_summary) {
            const status = data.analytics_summary.macd_status || 'Bullish';
            macdBadge.textContent = status;
            macdBadge.className = status.includes('Bullish') ? 'badge badge-buy' : 'badge badge-sell';
        }
    }

    function renderAnalyticsSummary(data) {
        const an = data.analytics_summary || {};
        const vs = data.volume_summary || {};

        const atrEl = document.getElementById('analytics-atr');
        const volEl = document.getElementById('analytics-volatility');
        const sma20El = document.getElementById('analytics-sma20-dist');
        const sma50El = document.getElementById('analytics-sma50-dist');
        const relVolEl = document.getElementById('analytics-rel-volume');

        if (atrEl) atrEl.textContent = '₹' + (an.atr || 0).toFixed(2);
        if (volEl) volEl.textContent = (an.volatility_pct || 0).toFixed(2) + '%';

        if (sma20El) {
            const v20 = an.price_vs_sma20_pct || 0;
            sma20El.textContent = (v20 >= 0 ? '+' : '') + v20.toFixed(2) + '%';
            sma20El.style.color = v20 >= 0 ? '#34D399' : '#F87171';
        }

        if (sma50El) {
            const v50 = an.price_vs_sma50_pct || 0;
            sma50El.textContent = (v50 >= 0 ? '+' : '') + v50.toFixed(2) + '%';
            sma50El.style.color = v50 >= 0 ? '#34D399' : '#F87171';
        }

        if (relVolEl) {
            const rv = vs.relative_volume || 1.0;
            relVolEl.textContent = rv.toFixed(2) + 'x';
            relVolEl.style.color = rv >= 1.5 ? '#F59E0B' : '#38BDF8';
        }
    }

    // ── Live Portfolio & Market Auto-Refresh Timer Engine ─────────────────
    let autoRefreshInterval = null;

    function setupAutoRefreshTimer() {
        const liveToggleBtn = document.getElementById('live-toggle-btn');

        // Start default 10s polling interval for portfolio & positions live updates
        startAutoRefreshInterval();

        if (liveToggleBtn) {
            liveToggleBtn.addEventListener('click', (e) => {
                e.preventDefault();
                if (autoRefreshInterval) {
                    stopAutoRefreshInterval();
                    liveToggleBtn.classList.remove('btn-live-stream');
                    liveToggleBtn.classList.add('btn-secondary');
                    liveToggleBtn.innerHTML = '⏸️ PAUSED (10s)';
                } else {
                    startAutoRefreshInterval();
                    liveToggleBtn.classList.remove('btn-secondary');
                    liveToggleBtn.classList.add('btn-live-stream');
                    liveToggleBtn.innerHTML = '<span class="live-dot pulse"></span> LIVE (10s)';
                }
            });
        }

        // Attach WebSocket / Socket.IO event listeners if available
        if (typeof io !== 'undefined') {
            try {
                const socket = io();
                const socketStatusDot = document.getElementById('socket-dot');
                const socketStatusText = document.getElementById('socket-status-text');

                socket.on('connect', () => {
                    if (socketStatusDot) socketStatusDot.className = 'socket-dot pulse-green';
                    if (socketStatusText) socketStatusText.textContent = 'WS CONNECTED (8ms)';
                });

                socket.on('disconnect', () => {
                    if (socketStatusDot) socketStatusDot.className = 'socket-dot pulse-red';
                    if (socketStatusText) socketStatusText.textContent = 'WS DISCONNECTED';
                });

                // Real-time market tick broadcast updates portfolio & paper positions automatically
                socket.on('market_update', (data) => {
                    console.log('⚡ Live Market Tick received via Socket.IO:', data);
                    triggerLivePortfolioRefresh();
                });
            } catch (e) {
                console.log('Socket.IO optional initialization fallback:', e.message);
            }
        }
    }

    function startAutoRefreshInterval() {
        if (autoRefreshInterval) clearInterval(autoRefreshInterval);
        autoRefreshInterval = setInterval(() => {
            triggerLivePortfolioRefresh();
        }, 10000); // Refresh every 10 seconds
    }

    function stopAutoRefreshInterval() {
        if (autoRefreshInterval) {
            clearInterval(autoRefreshInterval);
            autoRefreshInterval = null;
        }
    }

    function triggerLivePortfolioRefresh() {
        // Automatically refresh portfolio holdings breakdown if portfolio view or background active
        if (typeof loadPortfolioData === 'function') {
            loadPortfolioData().catch(err => console.error('Auto-refresh portfolio error:', err));
        }
    }

    function triggerLivePortfolioRefresh() {
        // Automatically refresh portfolio holdings breakdown if portfolio view or background active
        if (typeof loadPortfolioData === 'function') {
            loadPortfolioData().catch(err => console.error('Auto-refresh portfolio error:', err));
        }
    }

    // Trigger initial load for current ticker if specified & set up refresh timer
    if (currentTicker) {
        loadVisualizationTerminal(currentTicker);
    }
    setupAutoRefreshTimer();


    // ── STOCK COMPARISON MODULE CONTROLLER ────────────────────────────
    let selectedComparisonStocks = ['RELIANCE.NS', 'TCS.NS'];
    let compModuleInitialized = false;

    function initStockComparisonModule() {
        if (!compModuleInitialized) {
            compModuleInitialized = true;
            setupComparisonControls();
        }
        renderComparisonChips();
        fetchAndRenderStockComparison();
    }

    function setupComparisonControls() {
        const addBtn = document.getElementById('comp-add-stock-btn');
        const inputEl = document.getElementById('comp-stock-input');
        const submitBtn = document.getElementById('comp-submit-btn');
        const periodSelect = document.getElementById('comp-period-select');

        if (addBtn && inputEl) {
            addBtn.addEventListener('click', () => {
                const val = inputEl.value.trim().toUpperCase();
                if (val) addComparisonStock(val);
            });

            inputEl.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    const val = inputEl.value.trim().toUpperCase();
                    if (val) addComparisonStock(val);
                }
            });
        }

        if (submitBtn) {
            submitBtn.addEventListener('click', () => {
                fetchAndRenderStockComparison();
            });
        }

        if (periodSelect) {
            periodSelect.addEventListener('change', () => {
                fetchAndRenderStockComparison();
            });
        }
    }

    function addComparisonStock(rawSymbol) {
        if (!rawSymbol) return;
        let sym = rawSymbol.trim().toUpperCase();
        if (!sym.includes('.')) sym = `${sym}.NS`;

        const banner = document.getElementById('comp-alert-banner');
        if (banner) banner.style.display = 'none';

        if (selectedComparisonStocks.includes(sym)) {
            showComparisonAlert(`Stock ${sym} is already in the comparison selection.`, 'warning');
            return;
        }

        if (selectedComparisonStocks.length >= 5) {
            showComparisonAlert('Maximum limit reached! You can compare at most 5 stocks simultaneously.', 'warning');
            return;
        }

        selectedComparisonStocks.push(sym);
        const inputEl = document.getElementById('comp-stock-input');
        if (inputEl) inputEl.value = '';

        renderComparisonChips();
        fetchAndRenderStockComparison();
    }

    function removeComparisonStock(sym) {
        if (selectedComparisonStocks.length <= 2) {
            showComparisonAlert('Comparison requires at least 2 stocks. Select a replacement before removing.', 'warning');
            return;
        }

        selectedComparisonStocks = selectedComparisonStocks.filter(s => s !== sym);
        renderComparisonChips();
        fetchAndRenderStockComparison();
    }

    function renderComparisonChips() {
        const chipsContainer = document.getElementById('comp-selected-chips');
        const countBadge = document.getElementById('comp-count-badge');
        if (!chipsContainer) return;

        if (countBadge) {
            countBadge.textContent = `${selectedComparisonStocks.length} / 5 Selected`;
        }

        const colors = ['#38BDF8', '#34D399', '#F59E0B', '#EF4444', '#8B5CF6'];
        let html = '';

        selectedComparisonStocks.forEach((sym, idx) => {
            const col = colors[idx % colors.length];
            html += `
                <div class="comp-chip" style="display: inline-flex; align-items: center; gap: 8px; padding: 6px 14px; background: rgba(255,255,255,0.9); border: 1.5px solid ${col}; border-radius: 20px; font-size: 13px; font-weight: 700; color: #0F172A; box-shadow: 0 2px 6px rgba(0,0,0,0.05);">
                    <span style="width: 10px; height: 10px; border-radius: 50%; background: ${col}; display: inline-block;"></span>
                    <span>${sym}</span>
                    <button type="button" onclick="window.removeComparisonStockGlobal('${sym}')" style="background: none; border: none; color: #64748B; font-weight: 800; cursor: pointer; padding: 0 2px; font-size: 14px; line-height: 1;" title="Remove ${sym}">
                        ✕
                    </button>
                </div>
            `;
        });

        chipsContainer.innerHTML = html;
    }

    window.removeComparisonStockGlobal = function(sym) {
        removeComparisonStock(sym);
    };

    function showComparisonAlert(msg, type = 'warning') {
        const banner = document.getElementById('comp-alert-banner');
        if (!banner) return;

        banner.style.display = 'block';
        if (type === 'warning') {
            banner.style.background = '#FEF3C7';
            banner.style.border = '1px solid #F59E0B';
            banner.style.color = '#92400E';
        } else {
            banner.style.background = '#FEE2E2';
            banner.style.border = '1px solid #EF4444';
            banner.style.color = '#991B1B';
        }
        banner.textContent = `⚠️ ${msg}`;
    }

    function fetchAndRenderStockComparison() {
        const periodSelect = document.getElementById('comp-period-select');
        const period = periodSelect ? periodSelect.value : '1y';
        const submitBtn = document.getElementById('comp-submit-btn');

        if (selectedComparisonStocks.length < 2) {
            showComparisonAlert('Please select at least 2 stocks for comparison.', 'warning');
            return;
        }

        if (submitBtn) {
            submitBtn.disabled = true;
            submitBtn.innerHTML = '⏳ Loading...';
        }

        const queryParams = selectedComparisonStocks.map(s => `symbols=${encodeURIComponent(s)}`).join('&');

        const historyPromise = fetch(`/api/stock-comparison/history?${queryParams}&period=${period}`).then(r => r.json());
        const predictionPromise = fetch('/api/stock-comparison/predictions', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ symbols: selectedComparisonStocks })
        }).then(r => r.json()).catch(err => ({ status: 'error', message: err.message }));

        Promise.all([historyPromise, predictionPromise])
            .then(([histRes, predRes]) => {
                if (submitBtn) {
                    submitBtn.disabled = false;
                    submitBtn.innerHTML = '⚡ COMPARE NOW';
                }

                if (histRes.status === 'success') {
                    renderNormalizedChart(histRes);
                    renderPricesChart(histRes);
                    renderMetricsTable(histRes.metrics_table);
                    updateComparisonSummaryKPIs(histRes);

                    if (histRes.metadata && histRes.metadata.warning) {
                        showComparisonAlert(histRes.metadata.warning, 'warning');
                    }
                } else {
                    showComparisonAlert(histRes.message || 'Failed to fetch historical comparison data.', 'danger');
                }

                if (predRes && predRes.status === 'success') {
                    renderPredictionsChart(predRes);
                    renderPredictionsGrid(predRes);
                }
            })
            .catch(err => {
                if (submitBtn) {
                    submitBtn.disabled = false;
                    submitBtn.innerHTML = '⚡ COMPARE NOW';
                }
                console.error('Error fetching stock comparison data:', err);
                showComparisonAlert('Network or server error while executing stock comparison.', 'danger');
            });
    }

    function updateComparisonSummaryKPIs(data) {
        const table = data.metrics_table || [];
        const meta = data.metadata || {};

        if (table.length === 0) return;

        // Top Return
        const sortedByReturn = [...table].sort((a, b) => b.percentage_return - a.percentage_return);
        const topRet = sortedByReturn[0];
        const topRetEl = document.getElementById('comp-kpi-top-return');
        const topRetSub = document.getElementById('comp-kpi-top-return-sub');
        if (topRetEl) topRetEl.textContent = `${topRet.symbol} (${topRet.percentage_return > 0 ? '+' : ''}${topRet.percentage_return}%)`;
        if (topRetSub) topRetSub.textContent = `P1: ₹${topRet.starting_price} ➔ P2: ₹${topRet.latest_price}`;

        // Lowest Volatility
        const sortedByVol = [...table].sort((a, b) => a.annualized_volatility - b.annualized_volatility);
        const lowVol = sortedByVol[0];
        const lowVolEl = document.getElementById('comp-kpi-lowest-vol');
        const lowVolSub = document.getElementById('comp-kpi-lowest-vol-sub');
        if (lowVolEl) lowVolEl.textContent = `${lowVol.symbol} (${lowVol.annualized_volatility}%)`;
        if (lowVolSub) lowVolSub.textContent = `Std Dev * sqrt(252)`;

        // Lowest Max Drawdown (least negative)
        const sortedByDD = [...table].sort((a, b) => b.max_drawdown - a.max_drawdown);
        const bestDD = sortedByDD[0];
        const bestDDEl = document.getElementById('comp-kpi-best-dd');
        const bestDDSub = document.getElementById('comp-kpi-best-dd-sub');
        if (bestDDEl) bestDDEl.textContent = `${bestDD.symbol} (${bestDD.max_drawdown}%)`;
        if (bestDDSub) bestDDSub.textContent = `Peak-to-Trough Decline`;

        // Common dates
        const datesEl = document.getElementById('comp-kpi-dates-count');
        const datesSub = document.getElementById('comp-kpi-dates-sub');
        if (datesEl) datesEl.textContent = meta.common_dates_count || 0;
        if (datesSub) datesSub.textContent = meta.start_date && meta.end_date ? `${meta.start_date} to ${meta.end_date}` : 'Aligned dates';
    }

    function renderNormalizedChart(data) {
        const container = document.getElementById('plotlyCompNormalizedChart');
        if (!container || typeof Plotly === 'undefined') return;

        const dates = data.charts_data.dates || [];
        const normData = data.charts_data.normalized || {};
        const traces = [];

        Object.keys(normData).forEach(sym => {
            const item = normData[sym];
            traces.push({
                x: dates,
                y: item.values,
                mode: 'lines',
                name: `${sym} (Base 100)`,
                line: { color: item.color, width: 2.5 },
                hovertemplate: `<b>${sym}</b><br>Date: %{x}<br>Normalized Value: %{y:.2f}<extra></extra>`
            });
        });

        // Add 100 baseline dashed line
        traces.push({
            x: [dates[0], dates[dates.length - 1]],
            y: [100, 100],
            mode: 'lines',
            name: 'Base 100 Line',
            line: { color: '#94A3B8', width: 1.5, dash: 'dash' },
            showlegend: false,
            hoverinfo: 'none'
        });

        const layout = {
            autosize: true,
            margin: { l: 50, r: 20, t: 20, b: 40 },
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            xaxis: {
                gridcolor: '#F1F5F9',
                zerolinecolor: '#E2E8F0',
                tickfont: { color: '#64748B', size: 11 }
            },
            yaxis: {
                title: { text: 'Normalized Index (Base = 100)', font: { color: '#64748B', size: 12 } },
                gridcolor: '#F1F5F9',
                zerolinecolor: '#E2E8F0',
                tickfont: { color: '#64748B', size: 11 }
            },
            legend: {
                orientation: 'h',
                x: 0,
                y: 1.12,
                font: { color: '#0F172A', size: 12, weight: 600 }
            }
        };

        Plotly.newPlot(container, traces, layout, { responsive: true, displayModeBar: false });
    }

    function renderPricesChart(data) {
        const container = document.getElementById('plotlyCompPricesChart');
        if (!container || typeof Plotly === 'undefined') return;

        const dates = data.charts_data.dates || [];
        const priceData = data.charts_data.closing_prices || {};
        const traces = [];

        Object.keys(priceData).forEach(sym => {
            const item = priceData[sym];
            traces.push({
                x: dates,
                y: item.values,
                mode: 'lines',
                name: `${sym} (₹)`,
                line: { color: item.color, width: 2 },
                hovertemplate: `<b>${sym}</b><br>Date: %{x}<br>Close Price: ₹%{y:.2f}<extra></extra>`
            });
        });

        const layout = {
            autosize: true,
            margin: { l: 50, r: 20, t: 20, b: 40 },
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            xaxis: {
                gridcolor: '#F1F5F9',
                zerolinecolor: '#E2E8F0',
                tickfont: { color: '#64748B', size: 11 }
            },
            yaxis: {
                title: { text: 'Closing Price (₹)', font: { color: '#64748B', size: 12 } },
                gridcolor: '#F1F5F9',
                zerolinecolor: '#E2E8F0',
                tickfont: { color: '#64748B', size: 11 }
            },
            legend: {
                orientation: 'h',
                x: 0,
                y: 1.12,
                font: { color: '#0F172A', size: 12, weight: 600 }
            }
        };

        Plotly.newPlot(container, traces, layout, { responsive: true, displayModeBar: false });
    }

    function renderMetricsTable(metricsTable) {
        const tbody = document.getElementById('comp-metrics-table-body');
        if (!tbody) return;

        if (!metricsTable || metricsTable.length === 0) {
            tbody.innerHTML = `<tr><td colspan="8" style="text-align: center; color: #94A3B8; padding: 20px;">No comparison data available.</td></tr>`;
            return;
        }

        let html = '';
        metricsTable.forEach(row => {
            const isPos = row.percentage_return >= 0;
            const retCol = isPos ? '#10B981' : '#EF4444';
            const retSign = isPos ? '+' : '';

            html += `
                <tr style="border-bottom: 1px solid #F1F5F9;">
                    <td style="font-weight: 700; color: #0F172A; display: flex; align-items: center; gap: 8px;">
                        <span style="width: 10px; height: 10px; border-radius: 50%; background: ${row.color}; display: inline-block;"></span>
                        ${row.symbol}
                    </td>
                    <td style="text-align: right; font-weight: 700; color: #0F172A;">₹${row.latest_price.toFixed(2)}</td>
                    <td style="text-align: center; color: #64748B; font-size: 12px;">${row.latest_date}</td>
                    <td style="text-align: right; font-weight: 600; color: ${retCol};">${retSign}₹${row.absolute_change.toFixed(2)}</td>
                    <td style="text-align: right; font-weight: 700; color: ${retCol};">${retSign}${row.percentage_return.toFixed(2)}%</td>
                    <td style="text-align: right; font-weight: 600; color: #3B82F6;">${row.annualized_volatility.toFixed(2)}%</td>
                    <td style="text-align: right; font-weight: 600; color: #EF4444;">${row.max_drawdown.toFixed(2)}%</td>
                    <td style="text-align: center; font-weight: 600; color: #64748B;">${row.observation_count}</td>
                </tr>
            `;
        });

        tbody.innerHTML = html;
    }

    function renderPredictionsChart(predRes) {
        const container = document.getElementById('plotlyCompPredictionsChart');
        if (!container || typeof Plotly === 'undefined') return;

        const dates = predRes.forecast_dates || [];
        const preds = predRes.predictions || [];
        const traces = [];

        const colors = ['#38BDF8', '#34D399', '#F59E0B', '#EF4444', '#8B5CF6'];

        preds.forEach((item, idx) => {
            const col = colors[idx % colors.length];
            traces.push({
                x: item.forecast_dates || dates,
                y: item.predicted_prices || [],
                mode: 'lines+markers',
                name: `${item.symbol} AI Forecast`,
                line: { color: col, width: 2.5, dash: 'solid' },
                marker: { size: 6, color: col },
                hovertemplate: `<b>${item.symbol}</b><br>Forecast Date: %{x}<br>Predicted Price: ₹%{y:.2f}<extra></extra>`
            });
        });

        const layout = {
            autosize: true,
            margin: { l: 50, r: 20, t: 20, b: 40 },
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            xaxis: {
                gridcolor: '#E9D5FF',
                zerolinecolor: '#DDD6FE',
                tickfont: { color: '#6D28D9', size: 11 }
            },
            yaxis: {
                title: { text: '7-Day Forecast Price (₹)', font: { color: '#6D28D9', size: 12 } },
                gridcolor: '#E9D5FF',
                zerolinecolor: '#DDD6FE',
                tickfont: { color: '#6D28D9', size: 11 }
            },
            legend: {
                orientation: 'h',
                x: 0,
                y: 1.12,
                font: { color: '#4C1D95', size: 12, weight: 600 }
            }
        };

        Plotly.newPlot(container, traces, layout, { responsive: true, displayModeBar: false });
    }

    function renderPredictionsGrid(predRes) {
        const grid = document.getElementById('comp-predictions-grid');
        if (!grid) return;

        const preds = predRes.predictions || [];
        if (preds.length === 0) {
            grid.innerHTML = '<div style="color: #6D28D9; font-size: 13px;">No predictions available.</div>';
            return;
        }

        let html = '';
        preds.forEach(item => {
            const expRet = item.expected_return_pct || 0.0;
            const isPos = expRet >= 0;
            const col = isPos ? '#059669' : '#DC2626';
            const bg = isPos ? '#D1FAE5' : '#FEE2E2';

            html += `
                <div style="background: #FFFFFF; border: 1px solid #DDD6FE; border-radius: 10px; padding: 14px; box-shadow: 0 2px 8px rgba(139,92,246,0.08);">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <strong style="color: #4C1D95; font-size: 14px;">${item.symbol}</strong>
                        <span style="font-size: 11px; font-weight: 700; padding: 2px 8px; border-radius: 12px; background: ${bg}; color: ${col};">
                            ${item.signal}
                        </span>
                    </div>
                    <div style="font-size: 12px; color: #6D28D9;">Current: <strong>₹${item.current_price.toFixed(2)}</strong></div>
                    <div style="font-size: 12px; color: #4C1D95; font-weight: 700; margin-top: 2px;">
                        7-Day Target: <strong>₹${item.predicted_7d_price.toFixed(2)}</strong> (${isPos ? '+' : ''}${expRet}%)
                    </div>
                    <div style="font-size: 11px; color: #7C3AED; margin-top: 4px;">Confidence: ${item.confidence_score}%</div>
                </div>
            `;
        });

        grid.innerHTML = html;
    }
});


