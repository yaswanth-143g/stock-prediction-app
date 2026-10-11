/**
 * static/js/tradingview_chart.js
 * TradingView Lightweight Charts Integration Component
 * ===================================================
 * Renders professional financial candlestick charts, volume histograms,
 * SMA/EMA indicator overlays, AI forecast lines, confidence bands, and BUY/SELL signal markers.
 */

window.TradingViewChartManager = (function() {
    let chartInstance = null;
    let candlestickSeries = null;
    let lineSeries = null;
    let volumeSeries = null;
    let overlaySeriesMap = {};
    let aiForecastSeries = null;
    let upperBandSeries = null;
    let lowerBandSeries = null;
    let resizeObserver = null;

    /**
     * Compatibility helpers for Lightweight Charts v3 and v4 APIs
     */
    function safeAddHistogramSeries(chart, options) {
        if (typeof chart.addHistogramSeries === 'function') {
            return chart.addHistogramSeries(options);
        }
        if (typeof chart.addSeries === 'function' && typeof LightweightCharts !== 'undefined' && LightweightCharts.HistogramSeries) {
            return chart.addSeries(LightweightCharts.HistogramSeries, options);
        }
        console.error('LightweightCharts: Unable to add HistogramSeries');
        return null;
    }

    function safeAddCandlestickSeries(chart, options) {
        if (typeof chart.addCandlestickSeries === 'function') {
            return chart.addCandlestickSeries(options);
        }
        if (typeof chart.addSeries === 'function' && typeof LightweightCharts !== 'undefined' && LightweightCharts.CandlestickSeries) {
            return chart.addSeries(LightweightCharts.CandlestickSeries, options);
        }
        console.error('LightweightCharts: Unable to add CandlestickSeries');
        return null;
    }

    function safeAddLineSeries(chart, options) {
        if (typeof chart.addLineSeries === 'function') {
            return chart.addLineSeries(options);
        }
        if (typeof chart.addSeries === 'function' && typeof LightweightCharts !== 'undefined' && LightweightCharts.LineSeries) {
            return chart.addSeries(LightweightCharts.LineSeries, options);
        }
        console.error('LightweightCharts: Unable to add LineSeries');
        return null;
    }

    /**
     * Initializes or resets the TradingView Lightweight Chart container
     * @param {string} containerId Element ID of the chart container div
     */
    function initChart(containerId = 'tvLightweightChartContainer') {
        const container = document.getElementById(containerId);
        if (!container) {
            console.error('TradingView chart container not found:', containerId);
            return null;
        }

        // Clear existing chart instance if re-initializing
        if (chartInstance) {
            try {
                chartInstance.remove();
            } catch (e) {
                console.warn('Error removing prior chart instance:', e);
            }
            chartInstance = null;
        }

        container.innerHTML = ''; // Clear container children

        if (typeof LightweightCharts === 'undefined') {
            console.error('LightweightCharts library is not loaded from CDN!');
            container.innerHTML = '<div style="color: #EF4444; padding: 20px; text-align: center;">LightweightCharts CDN library loading error.</div>';
            return null;
        }

        const containerWidth = container.clientWidth || 800;
        const containerHeight = container.clientHeight || 480;

        chartInstance = LightweightCharts.createChart(container, {
            width: containerWidth,
            height: containerHeight,
            layout: {
                background: { type: 'solid', color: '#0F151D' },
                textColor: '#A8B2BF',
                fontSize: 12,
                fontFamily: 'Inter, sans-serif'
            },
            grid: {
                vertLines: { color: '#1C2632' },
                horzLines: { color: '#1C2632' }
            },
            crosshair: {
                mode: LightweightCharts.CrosshairMode ? LightweightCharts.CrosshairMode.Normal : 0,
                vertLine: {
                    color: '#38BDF8',
                    width: 1,
                    style: LightweightCharts.LineStyle ? LightweightCharts.LineStyle.Dashed : 2,
                    labelBackgroundColor: '#151E29'
                },
                horzLine: {
                    color: '#38BDF8',
                    width: 1,
                    style: LightweightCharts.LineStyle ? LightweightCharts.LineStyle.Dashed : 2,
                    labelBackgroundColor: '#151E29'
                }
            },
            rightPriceScale: {
                borderColor: '#253140',
                textColor: '#A8B2BF',
                scaleMargins: {
                    top: 0.1,
                    bottom: 0.25 // Reserve bottom 25% for volume histogram
                }
            },
            timeScale: {
                borderColor: '#253140',
                timeVisible: true,
                secondsVisible: false
            },
            handleScroll: {
                mouseWheel: true,
                pressedMouseMove: true,
                horzTouchDrag: true,
                vertTouchDrag: true
            },
            handleScale: {
                axisPressedMouseMove: true,
                mouseWheel: true,
                pinch: true
            }
        });

        // Add Volume Series (Histogram at bottom)
        volumeSeries = safeAddHistogramSeries(chartInstance, {
            color: '#26a69a',
            priceFormat: { type: 'volume' },
            priceScaleId: 'volume_scale',
            scaleMargins: {
                top: 0.75,
                bottom: 0
            }
        });

        if (chartInstance.priceScale) {
            try {
                chartInstance.priceScale('volume_scale').applyOptions({
                    scaleMargins: {
                        top: 0.75,
                        bottom: 0
                    }
                });
            } catch (e) {
                console.warn('Volume price scale applyOptions warning:', e);
            }
        }

        // Setup Responsive Resize Observer
        if (window.ResizeObserver) {
            if (resizeObserver) resizeObserver.disconnect();
            resizeObserver = new ResizeObserver(entries => {
                if (chartInstance && entries[0] && entries[0].contentRect) {
                    const { width, height } = entries[0].contentRect;
                    if (width > 0) {
                        chartInstance.applyOptions({ width: width, height: height || 480 });
                    }
                }
            });
            resizeObserver.observe(container);
        }

        return chartInstance;
    }

    /**
     * Renders OHLCV dataset onto TradingView Lightweight Chart
     * @param {Object} vizData Payload returned by /api/stock/<ticker>/visualization
     * @param {string} chartType 'candlestick' or 'line'
     */
    function renderData(vizData, chartType = 'candlestick') {
        if (!vizData || !vizData.ohlcv_series || vizData.ohlcv_series.length === 0) {
            console.warn('TradingView chart: empty OHLCV dataset provided');
            return;
        }

        const containerId = 'tvLightweightChartContainer';
        initChart(containerId);

        if (!chartInstance) return;

        const series = vizData.ohlcv_series;

        // Parse candlestick and volume arrays
        const candleData = [];
        const lineData = [];
        const volumeData = [];

        function parseTime(timeStr) {
            if (!timeStr) return null;
            if (typeof timeStr === 'number') return timeStr;
            const strVal = String(timeStr).trim();
            if (strVal.includes(':')) {
                const d = new Date(strVal.replace(' ', 'T'));
                if (!isNaN(d.getTime())) {
                    return Math.floor(d.getTime() / 1000);
                }
            }
            return strVal.split(' ')[0];
        }

        series.forEach(item => {
            const timeVal = parseTime(item.date);
            const openVal = Number(item.open);
            const highVal = Number(item.high);
            const lowVal = Number(item.low);
            const closeVal = Number(item.close);
            const volVal = Number(item.volume || 0);

            if (timeVal && !isNaN(closeVal)) {
                candleData.push({
                    time: timeVal,
                    open: openVal,
                    high: highVal,
                    low: lowVal,
                    close: closeVal
                });

                lineData.push({
                    time: timeVal,
                    value: closeVal
                });

                volumeData.push({
                    time: timeVal,
                    value: volVal,
                    color: closeVal >= openVal ? 'rgba(52, 211, 153, 0.4)' : 'rgba(248, 113, 113, 0.4)'
                });
            }
        });

        // 1. Add Main Price Series (Candlestick or Line)
        if (chartType === 'candlestick') {
            candlestickSeries = safeAddCandlestickSeries(chartInstance, {
                upColor: '#10B981',
                downColor: '#EF4444',
                borderVisible: false,
                wickUpColor: '#34D399',
                wickDownColor: '#F87171'
            });
            if (candlestickSeries) candlestickSeries.setData(candleData);
        } else {
            lineSeries = safeAddLineSeries(chartInstance, {
                color: '#38BDF8',
                lineWidth: 2
            });
            if (lineSeries) lineSeries.setData(lineData);
        }

        // Set Volume Data
        if (volumeSeries && volumeData.length > 0) {
            volumeSeries.setData(volumeData);
        }

        // 2. Add Indicator Overlays if Checked
        renderIndicators(series, vizData);

        // 3. Add AI Forecast Overlay Line & Markers
        renderAIForecast(vizData);

        // 4. Fit Content Time Scale
        chartInstance.timeScale().fitContent();
    }

    /**
     * Renders Technical Indicator Overlay lines (SMA 20/50/200, EMA 20/50, Bollinger)
     */
    function renderIndicators(series, vizData) {
        overlaySeriesMap = {};

        const mainSeries = candlestickSeries || lineSeries;
        if (!mainSeries) return;

        // SMA 20
        const chkSma20 = document.getElementById('chk-sma20');
        if (chkSma20 && chkSma20.checked) {
            const sma20Data = series.filter(s => s.sma20 !== null && !isNaN(s.sma20)).map(s => ({ time: s.date, value: s.sma20 }));
            if (sma20Data.length > 0) {
                const s = safeAddLineSeries(chartInstance, { color: '#3B82F6', lineWidth: 1.5, title: 'SMA 20' });
                if (s) {
                    s.setData(sma20Data);
                    overlaySeriesMap['sma20'] = s;
                }
            }
        }

        // SMA 50
        const chkSma50 = document.getElementById('chk-sma50');
        if (chkSma50 && chkSma50.checked) {
            const sma50Data = series.filter(s => s.sma50 !== null && !isNaN(s.sma50)).map(s => ({ time: s.date, value: s.sma50 }));
            if (sma50Data.length > 0) {
                const s = safeAddLineSeries(chartInstance, { color: '#F97316', lineWidth: 1.5, title: 'SMA 50' });
                if (s) {
                    s.setData(sma50Data);
                    overlaySeriesMap['sma50'] = s;
                }
            }
        }

        // SMA 200
        const chkSma200 = document.getElementById('chk-sma200');
        if (chkSma200 && chkSma200.checked) {
            const sma200Data = series.filter(s => s.sma200 !== null && !isNaN(s.sma200)).map(s => ({ time: s.date, value: s.sma200 }));
            if (sma200Data.length > 0) {
                const s = safeAddLineSeries(chartInstance, { color: '#A855F7', lineWidth: 1.5, title: 'SMA 200' });
                if (s) {
                    s.setData(sma200Data);
                    overlaySeriesMap['sma200'] = s;
                }
            }
        }

        // EMA 20
        const chkEma20 = document.getElementById('chk-ema20');
        if (chkEma20 && chkEma20.checked) {
            const ema20Data = series.filter(s => s.ema20 !== null && !isNaN(s.ema20)).map(s => ({ time: s.date, value: s.ema20 }));
            if (ema20Data.length > 0) {
                const s = safeAddLineSeries(chartInstance, { color: '#14B8A6', lineWidth: 1.5, title: 'EMA 20' });
                if (s) {
                    s.setData(ema20Data);
                    overlaySeriesMap['ema20'] = s;
                }
            }
        }

        // EMA 50
        const chkEma50 = document.getElementById('chk-ema50');
        if (chkEma50 && chkEma50.checked) {
            const ema50Data = series.filter(s => s.ema50 !== null && !isNaN(s.ema50)).map(s => ({ time: s.date, value: s.ema50 }));
            if (ema50Data.length > 0) {
                const s = safeAddLineSeries(chartInstance, { color: '#EC4899', lineWidth: 1.5, title: 'EMA 50' });
                if (s) {
                    s.setData(ema50Data);
                    overlaySeriesMap['ema50'] = s;
                }
            }
        }

        // Bollinger Bands
        const chkBollinger = document.getElementById('chk-bollinger');
        if (chkBollinger && chkBollinger.checked) {
            const upperData = series.filter(s => s.bb_upper !== null && !isNaN(s.bb_upper)).map(s => ({ time: s.date, value: s.bb_upper }));
            const lowerData = series.filter(s => s.bb_lower !== null && !isNaN(s.bb_lower)).map(s => ({ time: s.date, value: s.bb_lower }));
            if (upperData.length > 0) {
                const u = safeAddLineSeries(chartInstance, { color: 'rgba(148, 163, 184, 0.6)', lineWidth: 1, lineStyle: 2, title: 'BB Upper' });
                if (u) u.setData(upperData);
                const l = safeAddLineSeries(chartInstance, { color: 'rgba(148, 163, 184, 0.6)', lineWidth: 1, lineStyle: 2, title: 'BB Lower' });
                if (l) l.setData(lowerData);
                overlaySeriesMap['bb_upper'] = u;
                overlaySeriesMap['bb_lower'] = l;
            }
        }
    }

    /**
     * Renders AI Forecast Line, Shaded Confidence Interval Bands, & Signal Markers on TradingView Chart
     */
    function renderAIForecast(vizData) {
        const chkAIForecast = document.getElementById('chk-ai-forecast');
        if (!chkAIForecast || !chkAIForecast.checked || !vizData.ai_forecast) return;

        const fc = vizData.ai_forecast;
        const series = vizData.ohlcv_series;
        if (!fc.prediction_dates || fc.prediction_dates.length === 0 || !series || series.length === 0) return;

        const lastActual = series[series.length - 1];
        const forecastLineData = [
            { time: lastActual.date, value: Number(lastActual.close) }
        ];
        const upperBandData = [
            { time: lastActual.date, value: Number(lastActual.close) }
        ];
        const lowerBandData = [
            { time: lastActual.date, value: Number(lastActual.close) }
        ];

        fc.prediction_dates.forEach((dStr, idx) => {
            const pVal = Number(fc.predicted_prices ? fc.predicted_prices[idx] : 0);
            const uVal = Number(fc.upper_bounds ? fc.upper_bounds[idx] : pVal * 1.02);
            const lVal = Number(fc.lower_bounds ? fc.lower_bounds[idx] : pVal * 0.98);

            if (dStr && !isNaN(pVal)) {
                forecastLineData.push({ time: dStr, value: pVal });
                upperBandData.push({ time: dStr, value: !isNaN(uVal) ? uVal : pVal * 1.02 });
                lowerBandData.push({ time: dStr, value: !isNaN(lVal) ? lVal : pVal * 0.98 });
            }
        });

        // 1. Add Shaded Confidence Interval Bounds (Upper & Lower 95% Error Bands)
        upperBandSeries = safeAddLineSeries(chartInstance, {
            color: 'rgba(245, 158, 11, 0.45)',
            lineWidth: 1,
            lineStyle: LightweightCharts.LineStyle ? LightweightCharts.LineStyle.Dashed : 2,
            title: '95% Confidence Upper Bound'
        });
        if (upperBandSeries) upperBandSeries.setData(upperBandData);

        lowerBandSeries = safeAddLineSeries(chartInstance, {
            color: 'rgba(245, 158, 11, 0.45)',
            lineWidth: 1,
            lineStyle: LightweightCharts.LineStyle ? LightweightCharts.LineStyle.Dashed : 2,
            title: '95% Confidence Lower Bound'
        });
        if (lowerBandSeries) lowerBandSeries.setData(lowerBandData);

        // 2. Add AI Forecast Main Trajectory Line
        aiForecastSeries = safeAddLineSeries(chartInstance, {
            color: '#F59E0B', // Vibrant Amber Gold
            lineWidth: 2.5,
            lineStyle: LightweightCharts.LineStyle ? LightweightCharts.LineStyle.Solid : 0,
            title: 'AI 7-Day Forecast Trajectory'
        });
        if (aiForecastSeries) aiForecastSeries.setData(forecastLineData);

        // 3. Add Buy/Sell Signal Markers on Main Candlestick Series
        const targetSeries = candlestickSeries || lineSeries;
        if (targetSeries && typeof targetSeries.setMarkers === 'function') {
            const sig = (fc.signal || 'HOLD').toUpperCase();
            let markerColor = '#F59E0B';
            let shape = 'circle';
            let textStr = `AI Target: ₹${fc.predicted_price ? fc.predicted_price.toFixed(2) : ''} (${sig})`;

            if (sig.includes('BUY')) {
                markerColor = '#10B981';
                shape = 'arrowUp';
                textStr = `🟢 BUY 🚀 (AI Target: ₹${fc.predicted_price ? fc.predicted_price.toFixed(2) : ''})`;
            } else if (sig.includes('SELL')) {
                markerColor = '#EF4444';
                shape = 'arrowDown';
                textStr = `🔴 SELL (AI Target: ₹${fc.predicted_price ? fc.predicted_price.toFixed(2) : ''})`;
            }

            targetSeries.setMarkers([
                {
                    time: lastActual.date,
                    position: sig.includes('BUY') ? 'belowBar' : 'aboveBar',
                    color: markerColor,
                    shape: shape,
                    text: textStr
                }
            ]);
        }
    }

    /**
     * Updates the chart in real-time with a live streaming price tick.
     * @param {Object} tick { time: 'YYYY-MM-DD', ltp: Number, open: Number, high: Number, low: Number, volume: Number }
     */
    function updateLiveTick(tick) {
        if (!chartInstance || !tick || !tick.time || isNaN(tick.ltp)) return;

        let timeVal = tick.time;
        if (typeof timeVal === 'string' && timeVal.includes(':')) {
            const d = new Date(timeVal.replace(' ', 'T'));
            if (!isNaN(d.getTime())) {
                timeVal = Math.floor(d.getTime() / 1000);
            }
        } else if (typeof timeVal === 'string' && !timeVal.includes('-')) {
            const parsedInt = parseInt(timeVal, 10);
            if (!isNaN(parsedInt)) timeVal = parsedInt;
        }

        const ltpVal = Number(tick.ltp);

        // Update Line Series if active
        if (lineSeries) {
            try {
                lineSeries.update({
                    time: timeVal,
                    value: ltpVal
                });
            } catch (e) {
                console.debug('TradingView lineSeries update error:', e);
            }
        }

        // Update Candlestick Series if active
        if (candlestickSeries) {
            try {
                const openVal = Number(tick.open || ltpVal);
                const highVal = Number(tick.high || Math.max(openVal, ltpVal));
                const lowVal = Number(tick.low || Math.min(openVal, ltpVal));
                candlestickSeries.update({
                    time: timeVal,
                    open: openVal,
                    high: highVal,
                    low: lowVal,
                    close: ltpVal
                });
            } catch (e) {
                console.debug('TradingView candlestickSeries update error:', e);
            }
        }

        // Update Volume Series if active
        if (volumeSeries && tick.volume !== undefined) {
            try {
                const openVal = Number(tick.open || ltpVal);
                volumeSeries.update({
                    time: timeVal,
                    value: Number(tick.volume),
                    color: ltpVal >= openVal ? 'rgba(52, 211, 153, 0.4)' : 'rgba(248, 113, 113, 0.4)'
                });
            } catch (e) {
                console.debug('TradingView volumeSeries update error:', e);
            }
        }
    }

    return {
        initChart: initChart,
        renderData: renderData,
        updateLiveTick: updateLiveTick
    };
})();
