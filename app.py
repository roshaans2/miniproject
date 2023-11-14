from flask import Flask,render_template, url_for ,flash , redirect
import joblib
from flask import request
import numpy as np
import tensorflow
import cv2

import os
from flask import send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf

from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler

parkinsons_data = pd.read_csv("parkinsons.csv")

X = parkinsons_data.drop(columns=['name','status'],axis=1)
Y = parkinsons_data['status'] 

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=2)

scaler = StandardScaler()

scaler.fit(X_train)


app=Flask(__name__,template_folder='template')

app.config['SECRET_KEY'] = '5791628bb0b13ce0c676dfde280ba245'


dir_path = os.path.dirname(os.path.realpath(__file__))
# UPLOAD_FOLDER = dir_path + '/uploads'
# STATIC_FOLDER = dir_path + '/static'
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'

from tensorflow.keras.models import load_model
model = load_model('model111.h5')
model222=load_model("my_model.h5")
model3 = load_model("trained.h5")
model4 = load_model("corona.h5")


def api(full_path):
    data = image.load_img(full_path, target_size=(50, 50, 3))
    data = np.expand_dims(data, axis=0)
    data = data * 1.0 / 255

    #with graph.as_default():
    predicted = model.predict(data)
    return predicted



def api1(full_path):
    data = image.load_img(full_path, target_size=(64, 64, 3))
    data = np.expand_dims(data, axis=0)
    data = data * 1.0 / 255

    #with graph.as_default():
    predicted = model222.predict(data)
    return predicted



def api2(full_path):
    data = image.load_img(full_path, target_size=(300, 300, 3))
    data = np.expand_dims(data, axis=0)
    data = data * 1.0 / 255

    #with graph.as_default():
    predicted = model3.predict(data)
    return predicted


def api3(full_path):
    data = image.load_img(full_path, target_size=(300, 300, 3))
    data = np.expand_dims(data, axis=0)
    data = data * 1.0 / 255

    #with graph.as_default():
    predicted = model4.predict(data)
    return predicted
   






@app.route('/upload', methods=['POST','GET'])
def upload_file():

    if request.method == 'GET':
        return render_template('malariafm.html')
    else:
        try:
            file = request.files['image']
            full_name = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(full_name)
            print(full_name)
            indices = {0: 'PARASITIC', 1: 'Uninfected', 2: 'Invasive carcinomar', 3: 'Normal'}
            result = api(full_name)
            print(result)
            predicted_class = np.asscalar(np.argmax(result, axis=1))
            accuracy = round(result[0][predicted_class] * 100, 2)               
            label = indices[predicted_class]
            return render_template('malariaresult.html', image_file_name = file.filename, label = label, accuracy = accuracy)
        except:
            flash("Please select the image first !!", "danger")      
            return redirect(url_for("Malaria"))


@app.route('/upload11', methods=['POST','GET'])
def upload11_file():

    if request.method == 'GET':
        return render_template('pneumoniafm.html')
    else:
        try:
            file = request.files['image']
            full_name = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(full_name)
            indices = {0: 'Normal', 1: 'Pneumonia'}
            result = api1(full_name)
            if(result>=0.5):
                label= indices[1]
                accuracy= result
            else:
                label= indices[0]
                accuracy= 1-result
            return render_template('pneumoniaresult.html', image_file_name = file.filename, label = label, accuracy = accuracy)
        except:
            flash("Please select the image first !!", "danger")      
            return redirect(url_for("Pneumonia"))
        
        
        
@app.route('/upload3', methods=['POST','GET'])
def upload3_file():

    if request.method == 'GET':
        return render_template('braintumorfm.html')
    else:
        try:
            file = request.files['image']
            full_name = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(full_name)
            indices = {0: 'Normal', 1: 'Brain Tumor'}
            result = api2(full_name)
            if(result>=0.5):
                label= indices[1]
                accuracy= result
            else:
                label= indices[0]
                accuracy= 1-result
            return render_template('braintumorresult.html', image_file_name = file.filename, label = label,accuracy=accuracy)
        except:
            flash("Please select the image first !!", "danger")      
            return redirect(url_for("Brain"))
        
        
        
@app.route('/upload4', methods=['POST','GET'])
def upload4_file():

    if request.method == 'GET':
        return render_template('covidfm.html')
    else:
        try:
            file = request.files['image']
            full_name = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(full_name)
            indices = {0: 'Covid', 1: 'Normal'}
            result = api3(full_name)
            if(result>=0.5):
                label= indices[1]
                accuracy= result
            else:
                label= indices[0]
                accuracy= 1-result
            return render_template('covidresult.html', image_file_name = file.filename, label = label,accuracy=accuracy)
        except:
            flash("Please select the image first !!", "danger")      
            return redirect(url_for("Corona"))
        

        



        
@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)
        
        
        
@app.route("/Pneumonia")
def Pneumonia():
    return render_template("pneumoniawp.html")
@app.route("/pneumoniaform")
def pneumoniaform():
    return render_template("pneumoniafm.html")



@app.route("/")
def home():
    return render_template("home.html")

@app.route("/diabetes")
def diabetes():
    return render_template("diabeteswp.html")
@app.route('/diabetesform')  
def diabetesform():  
    return render_template("diabetesfm.html");






@app.route("/heart")
def heart():
    return render_template("heartwp.html")
@app.route("/heartform")
def heartform():
    return render_template("heartfm.html")



@app.route("/liver")
def liver():
    return render_template("liver.html")


@app.route("/kidney")
def kidney():
    return render_template("kidneywp.html")
@app.route("/kidneyform")
def kidneyform():
    return render_template("kidneyfm.html")


@app.route("/Malaria")
def Malaria():
    return render_template("malariawp.html")
@app.route("/malariaform")
def malariafm():
    return render_template("malariafm.html")


@app.route("/Brain")
def Brain():
    return render_template("braintumorwp.html")
@app.route("/brainform")
def brainform():
    return render_template("braintumorfm.html")


@app.route("/Corona")
def Corona():
    return render_template("covidwp.html")
@app.route("/coronaform")
def coronaform():
    return render_template("covidfm.html")

@app.route("/breastcancer")
def breastcancer():
    return render_template("breastcancerwp.html")
@app.route("/breastcancerform")
def breastcancerform():
    return render_template("breastcancerfm.html")

@app.route("/parkinsons")
def parkinsons():
    return render_template("parkinsonwp.html")
@app.route("/parkinsonform")
def parkinsonform():
    return render_template("parkinsonfm.html")


#Hospitals
@app.route("/braintumorhosp")
def braintumorhosp():
    return render_template("braintumorhp.html")

@app.route("/malariahosp")
def malariahosp():
    return render_template("malariahp.html")

@app.route("/diabeteshosp")
def diabeteshosp():
    return render_template("diabeteshp.html")

@app.route("/pneumoniahosp")
def pneumoniahosp():
    return render_template("pneumoniahp.html")

@app.route("/coronahosp")
def coronahosp():
    return render_template("covidhp.html")

@app.route("/kidneyhosp")
def kidneyhosp():
    return render_template("kidneyhp.html")

@app.route("/hearthosp")
def hearthosp():
    return render_template("hearthp.html")

@app.route("/breastcancerhosp")
def breastcancerhosp():
    return render_template("breastcancerhp.html")

@app.route("/parkinsonhosp")
def parkinsonhosp():
    return render_template("parkinsonhp.html")



@app.route("/diet")
def diet():
    return render_template("diet.html")

@app.route("/yoga")
def yoga():
    return render_template("yoga.html")

#diet

@app.route("/braintumordiet")
def braintumordiet():
    return render_template("braintumordp.html")

@app.route("/malariadiet")
def malariadiet():
    return render_template("malariadp.html")

@app.route("/diabetesdiet")
def diabetesdiet():
    return render_template("diabetesdp.html")

@app.route("/pneumoniadiet")
def pneumoniadiet():
    return render_template("pneumoniadp.html")

@app.route("/coviddiet")
def coviddiet():
    return render_template("coviddp.html")

@app.route("/kidneydiet")
def kidneydiet():
    return render_template("kidneydp.html")

@app.route("/heartdiet")
def heartdiet():
    return render_template("heartdp.html")


@app.route("/breastcancerdiet")
def breastcancerdiet():
    return render_template("breastcancerdp.html")

@app.route("/parkinsondiet")
def parkinsondiet():
    return render_template("parkinsondp.html")


@app.route("/braintumoryoga")
def braintumoryoga():
    return render_template("braintumorya.html")

@app.route("/malariayoga")
def malariayoga():
    return render_template("malariaya.html")

@app.route("/diabetesyoga")
def diabetesyoga():
    return render_template("diabetesya.html")

@app.route("/pneumoniayoga")
def pneumoniayoga():
    return render_template("pneumoniaya.html")

@app.route("/covidyoga")
def covidyoga():
    return render_template("covidya.html")

@app.route("/kidneyyoga")
def kidneyyoga():
    return render_template("kidneyya.html")

@app.route("/heartyoga")
def heartyoga():
    return render_template("heartya.html")

@app.route("/breastcanceryoga")
def breastcanceryoga():
    return render_template("breastcancerya.html")

@app.route("/parkinsonyoga")
def parkinsonyoga():
    return render_template("parkinsonya.html")
       
    
    
@app.route("/about")
def about():
    return render_template("aboutus.html")






def ValuePredictor(to_predict_list, size):
    to_predict = np.array(to_predict_list).reshape(1,size)
    if(size==8): #Diabetes
        loaded_model = joblib.load("diabetes_model")
        result = loaded_model.predict(to_predict)
    
    
    elif(size==13):
        loaded_model = joblib.load("heart_model1")
        result =loaded_model.predict(to_predict)
        
    elif(size==10):
        loaded_model = joblib.load("liver_model")
        result = loaded_model.predict(to_predict)
        
        
    elif(size==12):#Kidney
        loaded_model = joblib.load("kidney_model")
        result = loaded_model.predict(to_predict)
        
    elif(size==30):
        loaded_model = joblib.load("breast_cancer_model")
        result = loaded_model.predict(to_predict)
        if(result[0]==1):
            result[0]=0
        else:
            result[0]=1
        
    elif(size==22):
        loaded_model = joblib.load("parkinsons_model")
        to_predict = scaler.transform(to_predict)
        result = loaded_model.predict(to_predict)
        
      
        
    
    
        
        
         
         
    
    return result[0]











@app.route('/result',methods = ["POST"])
def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list=list(to_predict_list.values())
#         to_predict_list = list(map(float,to_predict_list))
        
        if(len(to_predict_list)==8):#Daiabtes
            result = ValuePredictor(to_predict_list,8)
       
        elif(len(to_predict_list)==13):#Heart
            result = ValuePredictor(to_predict_list,13)
            
        elif(len(to_predict_list)==10):#Liver
            result = ValuePredictor(to_predict_list,10)
            
        elif(len(to_predict_list)==12):
            result = ValuePredictor(to_predict_list,12)
            
        elif(len(to_predict_list)==30):
            result = ValuePredictor(to_predict_list,30)
            
        elif(len(to_predict_list)==22):
            result = ValuePredictor(to_predict_list,22)
            
        

   
    if(int(result)==1):
        prediction='Sorry ! You are Suffering'
    else:
        prediction='Congrats ! you are Healthy' 
    return(render_template("mlresult.html", prediction=prediction))

@app.route("/doctor")
def doctor():
    return render_template("index.html")

@app.route("/1")
def doc1():
    return render_template("1.html")

@app.route("/2")
def doc2():
    return render_template("2.htm")

@app.route("/3")
def doc3():
    return render_template("3.htm")

@app.route("/4")
def doc4():
    return render_template("4.htm")

@app.route("/5")
def doc5():
    return render_template("5.htm")

@app.route("/6")
def doc6():
    return render_template("6.htm")

@app.route("/7")
def doc7():
    return render_template("7.htm")

if __name__ == "__main__":
    app.run(debug=True)