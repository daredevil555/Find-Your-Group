import numpy as np
from flask import Flask, request, render_template
from sklearn.externals import joblib 

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
    final_features=np.array(int_features).reshape(1,10)
    knn_from_joblib = joblib.load('model.pkl')  
    y=knn_from_joblib.predict(final_features) 
    return render_template('index.html', prediction_text=y[0])         

if __name__ == "__main__":
    app.run(debug=True)
