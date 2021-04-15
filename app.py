from flask import Flask,render_template,url_for,request,jsonify
import pandas as pd
from logging import FileHandler , WARNING
import pickle
import numpy as np
import keras
from sklearn.externals import joblib
import pickle
from keras.models import load_model
from sklearn.preprocessing import StandardScaler

sc=pickle.load(open('transform_ann.pkl','rb'))

print(keras.__version__)
clf = load_model('ann_deepblue_outliers1.h5')
app = Flask(__name__)
fh=FileHandler('errorlog.txt')
fh.setLevel(WARNING)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        nq = int(request.form['nq'])
        dr = int(request.form['dr'])
        ft = int(request.form['ft'])
        td = int(request.form['td'])
        mr3 = int(request.form['mr3'])
        data=[td,nq,mr3,dr,ft]
        for i in range(len(data)):
            data[i]=int(data[i])
        vect = np.array([td,nq,mr3,dr,ft]).reshape(1,-1)
        inp=sc.transform(vect)
        print(vect,inp)
        my_prediction = clf.predict(inp)
    return render_template('result.html',prediction = my_prediction[0][0])
@app.route('/predict_api/')
def predict_api():
    nq = int(request.args['nq'])
    dr = int(request.args['dr'])
    ft = int(request.args['ft'])
    td = int(request.args['td'])
    mr3 = int(request.args['mr3'])
    data=[td,nq,mr3,dr,ft]
    for i in range(len(data)):
        data[i]=int(data[i])
    vect = np.array([td,nq,mr3,dr,ft]).reshape(1,-1)
    inp=sc.transform(vect)
    print(vect,inp)
    my_prediction = clf.predict(inp)
    return jsonify({"prediction":str(my_prediction[0][0])})

#api.add_resource(home,'/')
#api.add_resource(predict_api,'/predict_api/<string:review>')

if __name__ == "__main__":
    app.run(debug=True)
