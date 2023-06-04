from flask import Flask, request, render_template
import pandas as pd
import joblib
import pickle
from sklearn.preprocessing import LabelEncoder
app = Flask(__name__)

resultat='0'
mdl='0'

@app.route('/', methods=['GET', 'POST'])
def main():
    global result , mdl
    
    if request.method == "POST":

        
        ids = request.form.get("intrusiondata")
        model = request.form.get("model")
        
          
        with open("minmaxonehotLabelencoder.pkl", 'rb') as handle:
            scaler= pickle.load(handle)
            le = pickle.load(handle)

        def le_transform(df):
           df = pd.DataFrame(df)  # Convert NumPy array to pandas DataFrame
           for col in df.columns:
              if df[col].dtype == 'object':
                 label_encoder = LabelEncoder()
           df[col] = label_encoder.fit_transform(df[col])
           return df.values   

        data = ids.split(',')
        data = le.transform([ids])
        dense = data.todense()
        denselist = dense.tolist()
        df1 = pd.DataFrame(denselist)
        
        if model == "Decision tree" :
            clf = joblib.load("dt.pkl")
            mdl = "Decision tree"
        elif model == "Random Forest" :
            clf = joblib.load("RF.pkl")
            mdl = "Random Forest"
        elif model == "SVM" :
            clf = joblib.load("svm.pkl")
            mdl = "SVM"
        elif model == "Adaboost" :
            clf = joblib.load("ab.pkl")
            mdl = "Adaboost"
        
        
        
        prediction = clf.predict(df)      
      
        if prediction==1 :
            result='An intrusion has been detected '
        else :
            result='No intrusion has been detected l'
            
    else:
        result = " "
        
        
    return render_template("ui.html", 
                           output = result,
                           )

if __name__ == '__main__':
    app.run(debug = True)