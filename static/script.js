document.addEventListener('DOMContentLoaded', () => {
    const predictBtn = document.getElementById('predict-btn');
    const loadingDiv = document.getElementById('loading');
    const resultContainer = document.getElementById('result-container');
    const errorDiv = document.getElementById('error-message');
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

    function displayResults(data) {
        // Update text stats
        document.getElementById('current-price').textContent = data.current_close.toFixed(4);
        document.getElementById('current-date').textContent = data.current_date;
        document.getElementById('predicted-price').textContent = data.predicted_price.toFixed(4);
        document.getElementById('predicted-date').textContent = data.predicted_date;

        // Show container
        resultContainer.classList.remove('hidden');

        // Render Chart
        renderChart(data);
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
});
