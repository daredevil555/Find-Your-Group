import numpy as np
from flask import Flask, request, jsonify, render_template
import warnings
import pickle
warnings.filterwarnings("ignore")

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST','GET'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    print(final_features)
    output = model.predict(final_features)
    return render_template('index.html', prediction_text='Group: {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)
