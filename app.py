from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))  

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
@app.route('/predict', methods=['POST'])
def predict():
    
    N = float(request.form['nitrogen'])
    P = float(request.form['phosphorous'])
    K = float(request.form['potassium'])
    temp = float(request.form['temperature'])
    humidity = float(request.form['humidity'])
    ph = float(request.form['ph'])
    rainfall = float(request.form['rainfall'])
    input_features = np.array([[N, P, K, temp, humidity, ph, rainfall]])
    prediction = model.predict(input_features)
    predicted_class = np.argmax(prediction)
    crops = ['rice', 'wheat', 'barley', 'maize', 'cotton', 'sugarcane', 'coffee', 'millets', 'soybean', 'sunflower', 'groundnut', 'rapeseed', 'jute', 'banana', 'mango', 'grape', 'watermelon', 'onion', 'cabbage', 'carrot', 'potato', 'tomato']
    predicted_crop = crops[predicted_class]
    
    return render_template('result.html', prediction=predicted_crop)


if __name__ == '__main__':
    app.run(debug=True)
