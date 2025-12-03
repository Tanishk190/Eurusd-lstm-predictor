from flask import Flask, render_template, jsonify
from predict import predict_next_day

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

if __name__ == '__main__':
    app.run(debug=True)
