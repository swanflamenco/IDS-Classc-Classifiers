from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import joblib
import pickle
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

resultat='0'
mdl='0'

@app.route('/', methods=['GET', 'POST'])
def main():
    global resultat, mdl

    if request.method == "POST":

        ids = request.form.get("intrusiondata")
        model = request.form.get("model")

        with open("minmaxonehot(2).pkl", 'rb') as handle:
            scaler, encoder, column_names = pickle.load(handle)

        data = ids.split(',')
        

        # Preprocess the input data
        df_test = pd.DataFrame([ids])
        df_test_encoded = encoder.transform(pd.get_dummies(df_test))

        # Append missing columns with 0
        missing_columns = set(encoder.get_feature_names()) - set(df_test_encoded.columns)
        for column in missing_columns:
            df_test_encoded[column] = 0

        # Scale the test data
        df_test_scaled = scaler.transform(df_test_encoded)
        if model == "Decision tree":
            clf = joblib.load("dt.pkl")
            mdl = "Decision tree"
        elif model == "Random Forest":
            clf = joblib.load("RF.pkl")
            mdl = "Random Forest"
        elif model == "SVM":
            clf = joblib.load("svm.pkl")
            mdl = "SVM"
        elif model == "Logistic Regression":
            clf = joblib.load("lr.pkl")
            mdl = "Logistic Regression"
        elif model == "MLP classifier":
            clf = joblib.load("mlp.pkl")
            mdl = "MLP classifier"

        prediction = clf.predict(df_test_encoded)

        if prediction == 1:
            resultat = 'Attaque'
        else:
            resultat = 'Normal'

    else:
        resultat = " "
        mdl = " "

    return render_template("ui.html",
                           output=resultat,
                           model=mdl)


if __name__ == '__main__':
    app.run(debug=True)