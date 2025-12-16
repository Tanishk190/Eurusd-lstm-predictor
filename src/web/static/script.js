document.addEventListener('DOMContentLoaded', () => {
    const predictBtn = document.getElementById('predict-btn');
    const loadingDiv = document.getElementById('loading');
    const resultContainer = document.getElementById('result-container');
    const errorDiv = document.getElementById('error-message');

    const backtestBtn = document.getElementById('backtest-btn');
    const backtestLoading = document.getElementById('backtest-loading');
    const backtestResultContainer = document.getElementById('backtest-result-container');
    const backtestErrorDiv = document.getElementById('backtest-error-message');
    const backtestChartImg = document.getElementById('backtest-chart-img');

    let chartInstance = null;

    predictBtn.addEventListener('click', async () => {
        // Reset state
        errorDiv.classList.add('hidden');
        resultContainer.classList.add('hidden');
        loadingDiv.classList.remove('hidden');
        predictBtn.disabled = true;

        try {
            const response = await fetch('/api/predict');
            const result = await response.json();

            if (result.status === 'success') {
                displayResults(result.data);
            } else {
                showError(result.message || 'An error occurred');
            }
        } catch (error) {
            showError('Failed to connect to the server');
        } finally {
            loadingDiv.classList.add('hidden');
            predictBtn.disabled = false;
        }
    });

    backtestBtn.addEventListener('click', async () => {
        // Reset backtest state
        backtestErrorDiv.classList.add('hidden');
        backtestResultContainer.classList.add('hidden');
        backtestLoading.classList.remove('hidden');
        backtestBtn.disabled = true;

        try {
            const response = await fetch('/api/backtest');
            const result = await response.json();

            if (result.status === 'success') {
                displayBacktestResults(result.data);
            } else {
                showBacktestError(result.message || 'Backtest failed');
            }
        } catch (error) {
            showBacktestError('Failed to connect to the server');
        } finally {
            backtestLoading.classList.add('hidden');
            backtestBtn.disabled = false;
        }
    });

    function displayResults(data) {
        // Update text stats
        document.getElementById('current-price').textContent = data.current_close.toFixed(5);
        document.getElementById('current-date').textContent = data.current_date;
        document.getElementById('predicted-price').textContent = data.predicted_price.toFixed(5);
        document.getElementById('predicted-date').textContent = data.predicted_date;

        // Show container
        resultContainer.classList.remove('hidden');

        // Render Chart
        renderChart(data);
    }

    function displayBacktestResults(data) {
        document.getElementById('backtest-samples').textContent = data.n_test_samples;
        document.getElementById('backtest-mse').textContent = data.mse.toFixed(6);
        document.getElementById('backtest-mae').textContent = data.mae.toFixed(6);
        document.getElementById('backtest-rmse').textContent = data.rmse.toFixed(6);

        // Bust cache for the static image so user sees latest chart
        if (backtestChartImg) {
            const url = new URL(backtestChartImg.src, window.location.origin);
            url.searchParams.set('t', Date.now().toString());
            backtestChartImg.src = url.toString();
        }

        backtestResultContainer.classList.remove('hidden');
    }

    function renderChart(data) {
        const ctx = document.getElementById('predictionChart').getContext('2d');

        if (chartInstance) {
            chartInstance.destroy();
        }

        // Prepare data for Chart.js
        const labels = [...data.history_dates, data.predicted_date];
        const historicalData = data.history_prices;

        // Create a dataset that spans history + prediction
        // We need to connect the last historical point to the predicted point
        const predictedDataPoint = {
            x: data.predicted_date,
            y: data.predicted_price
        };

        // Combine for a continuous line? 
        // Actually, let's keep them as separate datasets for styling

        // Dataset 1: History
        const historyDataset = {
            label: 'Historical Price',
            data: historicalData,
            borderColor: '#94a3b8',
            backgroundColor: 'rgba(148, 163, 184, 0.1)',
            tension: 0.1,
            pointRadius: 3
        };

        // Dataset 2: Prediction (just the last point and the connection)
        // To connect, we need the last history point
        const lastHistoryPrice = historicalData[historicalData.length - 1];
        const predictionData = new Array(historicalData.length - 1).fill(null);
        predictionData.push(lastHistoryPrice); // Start from last known
        predictionData.push(data.predicted_price); // End at prediction

        const predictionDataset = {
            label: 'Prediction',
            data: predictionData,
            borderColor: '#3b82f6',
            borderDash: [5, 5],
            pointRadius: 5,
            pointBackgroundColor: '#3b82f6'
        };

        chartInstance = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [historyDataset, predictionDataset]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    intersect: false,
                    mode: 'index',
                },
                plugins: {
                    legend: {
                        labels: { color: '#94a3b8' }
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false,
                    }
                },
                scales: {
                    x: {
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        ticks: { color: '#94a3b8' }
                    },
                    y: {
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        ticks: { color: '#94a3b8' }
                    }
                }
            }
        });
    }

    function showError(msg) {
        errorDiv.textContent = msg;
        errorDiv.classList.remove('hidden');
    }

    function showBacktestError(msg) {
        backtestErrorDiv.textContent = msg;
        backtestErrorDiv.classList.remove('hidden');
    }
});
