from flask import Flask, render_template, jsonify
from src.models.predict import predict_next_day
from src.models.backtest import backtest_model

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/predict')
def get_prediction():
    try:
        data = predict_next_day(return_data=True)
        if data:
            return jsonify({'status': 'success', 'data': data})
        else:
            return jsonify({'status': 'error', 'message': 'Prediction failed'}), 500
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/backtest')
def get_backtest():
    """
    Run a historical backtest of the trained model and return metrics.

    The backtest also saves a chart image to the static directory
    (`backtest_results.png`), which the frontend can display.
    """
    try:
        result = backtest_model()
        if result:
            return jsonify({'status': 'success', 'data': result})
        else:
            return jsonify({'status': 'error', 'message': 'Backtest failed'}), 500
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
