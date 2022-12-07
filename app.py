import numpy as np
from flask import Flask, request, jsonify, render_template,send_from_directory
import pandas as pd
import re
import re,string
from underthesea import sent_tokenize,word_tokenize
import pickle
import nltk
import dill
import joblib
app = Flask(__name__, static_folder='assets',static_url_path='')

saved_svc_model = joblib.load('svc-model.pkl')
saved_tfidf = pickle.load(open('tfidf_vector.pkl', 'rb'))

with open("vn-stopword.txt",encoding='utf-8') as file:
    stopwords = file.readlines()
    stopwords = [word.rstrip() for word in stopwords]

def vietnamese_text_preprocessing(sent):
    sent = re.sub(f'[{string.punctuation}\d\n]', '', sent)
    sent = word_tokenize(sent)
    sent = [w for w in sent if w not in stopwords or w != '']

    return " ".join(sent)
# server

@app.route('/assets/<path:path>')
def send_js(path):
    return send_from_directory('assets', path)
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    requestedValues = list(request.form.values())

    text = requestedValues[0]
    print(text)
    tfid_text = saved_tfidf.transform([vietnamese_text_preprocessing(text)])
    pred_result = saved_svc_model.predict(tfid_text)[0]
    pred_result

    prediction_result="Văn bản bình thường"
    if pred_result == 0:
        prediction_result = "Văn bản có vẻ tiêu cực"

    return render_template('index.html', pred_result=pred_result,
                           prediction_result=prediction_result)

if __name__ == '__main__':
    # Threaded option to enable multiple instances for multiple user access support
    app.run(threaded=True, port=5000)