import numpy as np
from flask import Flask,render_template,Response,request,jsonify,json
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    prediction=model.predict(final_features)

    output=round(prediction[0],2)

    new_price = round(10**prediction[0],2)


    return render_template('index.html', prediction_text = 'House price should be $ {}'.format(output))

if __name__ == "__main__":
    app.run(debug = True)
    


