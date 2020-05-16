import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

app = Flask(__name__)

model = load_model("model2.h5")
with open('tokenizer2.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
T = 1645
ps = PorterStemmer()
labels = ["business","entertainment","politics","sport","tech"]

def process(x):
    article = str(re.sub('[^a-zA-Z]',' ',x))
    article = article.lower()
    article = article.split()
    article = [ ps.stem(word) for word in article if word not in set(stopwords.words('english'))]
    article = " ".join(article)
    return article

@app.route('/')
def home():
    return render_template('./index.html',article="")

@app.route('/predict',methods=['GET'])
def predict():
    inp = request.args['data']
    print("data : ",inp)
    return render_template('index.html', prediction_text='ssup')

@app.route('/predict2',methods=['POST'])
def predict2():    
    input_list = [x for x in request.form.values()]
    article = input_list[0]
    dummy_article = article
    #print("x : ",article)
    article = process(article)
    q = tokenizer.texts_to_sequences([article])
    q = pad_sequences(q, maxlen=T)
    output = model.predict(q)
    idx = np.argmax(output[0])
    category = labels[idx]
    return render_template('index.html', prediction_text=category, article=dummy_article)


if __name__ == "__main__":
    app.run(debug=True)
