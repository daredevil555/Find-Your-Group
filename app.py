import numpy as np
from flask import Flask, request, render_template
import pickle5 as pickle

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = np.array(int_features).reshape(1,10)
    model = pickle.load(open('model.pkl', 'rb'))
    return render_template('index.html', prediction_text=final_features)
    output = model.predict(final_features)
    

if __name__ == "__main__":
    app.run(debug=True)
