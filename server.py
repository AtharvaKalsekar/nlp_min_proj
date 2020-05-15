import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

model = load_model("model.h5")
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
T = 3356
labels = ["crime","entertainment","politics","sport","tech"]

@app.route('/')
def home():
    return render_template('./index.html')

@app.route('/predict',methods=['GET'])
def predict():
    inp = request.args['data']
    print("data : ",inp)
    return render_template('index.html', prediction_text='ssup')

@app.route('/predict2',methods=['POST'])
def predict2():
    
    input_list = [x for x in request.form.values()]
    article = input_list[0]
    q = tokenizer.texts_to_sequences([article])
    q = pad_sequences(q, maxlen=T)
    #int_features = request.form.values()
    #final_features = [np.array(int_features)]
    #output = round(prediction[0], 2)
    output = model.predict(q)
    idx = np.argmax(output[0])
    category = labels[idx]
    #print("final : ", int_features)
    return render_template('index.html', prediction_text=category)

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)