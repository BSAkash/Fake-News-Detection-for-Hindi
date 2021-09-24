#import Flask
from flask import Flask, render_template, request, jsonify
from gensim.models import Word2Vec
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt
import csv
import joblib
from scipy.spatial import distance
from flask_cors import CORS, cross_origin
import nltk
from newspaper import Article
import numpy as np
import pickle as pk


app = Flask(__name__)
CORS(app, support_credentials=True)

app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/')
@cross_origin(supports_credentials=True)

def home():
    return render_template('home.html')

def documentvec(word2vec_model,summarywords):
    "This function Creates a document vector by taking the mean of word vectors of the words in the document"
    k=[]
    for i in range(len(summarywords)):
        if summarywords[i] in word2vec_model.wv:#model.wv.vocab gives the entire word vocabulary
            k.append(word2vec_model.wv[summarywords[i]])#of the generated model upon the given dataset
    return np.mean(k,axis=0)

@cross_origin(supports_credentials=True)
@app.route('/predict/', methods=['GET','POST'])
def predict():


    if request.method == "POST":
        request_data = request.get_json()

        feature_extraction_method = request_data['featureExtraction']
        machine_learning_method = request_data['machineLearning']
        data_entry_method = request_data['dataEntry']
        sample_data_entry_method = request_data['sampleDataEntry']

        text_data = None
        link_data = None

        if data_entry_method == 'sample':
            if sample_data_entry_method == 'sampleLink':
                data_entry_method == 'link'
                link_data = request_data['sampleDataTypeLink']
            else:
                data_entry_method == 'text'
                text_data = request_data['sampleDataTypeText']
        else:
            if 'text' in request_data:
                text_data = request_data['text']
            if 'link' in request_data:
                link_data = request_data['link']

        if data_entry_method == 'text':
            test_summary = text_data
            try:
                prediction = preprocessDataAndPredict(test_summary)
                response = jsonify(prediction)
                response.headers.add("Access-Control-Allow-Origin", "*")
                return response

            except ValueError:
                return "Please Enter valid values"

        else :
            url = link_data
            article = Article(url)
            article.download()
            article.parse()
            article.nlp()
            test_summary = article.summary
            try:
                prediction = preprocessDataAndPredict(test_summary, feature_extraction_method, machine_learning_method)
                response = jsonify(prediction)
                response.headers.add("Access-Control-Allow-Origin", "*")
                return response

            except ValueError:
                return "Please Enter valid values"
        pass
    pass

def preprocessDataAndPredict(test_summary, feature_extraction_method, machine_learning_method):

    category = {}
    if feature_extraction_method == "Word2Vec":

        model = Word2Vec.load("word2vec.model")
        with open('Final_Vectors.csv', newline='') as f:
            reader = csv.reader(f)
            True_vector = next(reader)  # gets the first line
            False_vector = next(reader)
        true_vector_floats = []
        false_vector_floats = []
        for item in True_vector:
            true_vector_floats.append(float(item))
        for item in False_vector:
            false_vector_floats.append(float(item))

        test_summary_words = test_summary.split(' ')
        test_corpus= []
        test_corpus.append(test_summary_words)

        model.build_vocab(test_corpus, update = True)
        model.train(test_corpus, total_examples=2, epochs = 1)
        test_vector = documentvec(model,test_summary_words)
        vector_pca_list = []
        vector_pca_list.append(test_vector)
        pca_reload = pk.load(open("pca.pkl",'rb'))
        result_new = pca_reload.transform(vector_pca_list)
        category[4] = result_new[0][0]
        category[5] = result_new[0][1]

        if machine_learning_method == "Cosine Similarity":

            true_class  = 1 - distance.cosine(test_vector,true_vector_floats)
            false_class  = 1 - distance.cosine(test_vector,false_vector_floats)
            if true_class > false_class:
                category[0] = 1
                category[1] = true_class
                category[2] = false_class
                category[3] = 0
            else:
                category[0] = 0
                category[1] = false_class
                category[2] = true_class
                category[3] =1

        elif machine_learning_method == "Logistic Regression":

            corpusLR = []
            corpusLR.append(test_vector)

            file = open("LR_W2V.pkl","rb")
            trained_model = joblib.load(file)
            category[0] = trained_model.predict(corpusLR)[0]



        elif machine_learning_method == "Random Forest":

            corpusRF = []
            corpusRF.append(test_vector)

            file = open("RFC_W2V.pkl","rb")
            trained_model = joblib.load(file)
            category[0] = trained_model.predict(corpusRF)[0]

        elif machine_learning_method == "Gradient Boosting":

            corpusGB = []
            corpusGB.append(test_vector)

            file = open("GBC_W2V.pkl","rb")
            trained_model = joblib.load(file)
            category[0] = trained_model.predict(corpusGB)[0]


        else:

            corpusDT = []
            corpusDT.append(test_vector)

            file = open("DT_W2V.pkl","rb")
            trained_model = joblib.load(file)
            category[0] = trained_model.predict(corpusDT)[0]

    else:


        file = open("vectorizer.pkl","rb")
        vectorizer = joblib.load(file)
        input=[test_summary]
        test_vector = vectorizer.transform(input)

        if machine_learning_method == "Logistic Regression":

            file = open("LR_TFIDF.pkl","rb")
            trained_model = joblib.load(file)
            category[0]= float(trained_model.predict(test_vector)[0])

        if machine_learning_method == "Random Forest":

            file = open("RFC_TFIDF.pkl","rb")
            trained_model = joblib.load(file)
            category[0] = float(trained_model.predict(test_vector)[0])

        if machine_learning_method == "Gradient Boosting":

            file = open("GBC_TFIDF.pkl","rb")
            trained_model = joblib.load(file)
            category[0] = float(trained_model.predict(test_vector)[0])

        else:

            file = open("DT_TFIDF.pkl","rb")
            trained_model = joblib.load(file)
            category[0] = float(trained_model.predict(test_vector)[0])

    return category

if __name__ == '__main__':
    app.run(host="localhost", port=5050, debug=True)