from flask import Flask, render_template,request,redirect,url_for
import os
import pandas as pd
import time
import pickle 
from dotenv import load_dotenv
import requests
import json

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score)
import numpy as np

from utils import average_score
'''pip install python-dotenv requests 
and pip install scikit-learn 
and venv\Scripts\activate 
and python -m venv venv 
and python app.py 
and pip install xgboost
and pip install shap'''


from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier, XGBRegressor

import joblib
import shap
import matplotlib.pyplot as plt
comparison_metrics=[]


app=Flask(__name__)
app.config['UPLOAD_FOLDER']="uploads"
@app.route('/')
def home():
    return render_template('home.html')
@app.route('/about')
def about():
    return render_template('about.html')
@app.route('/upload',methods=['GET','POST'])
def upload_file():
    if request.method=="POST":
        filename=None
        if 'dataset' in request.files and request.files['dataset'].filename !='':
            file=request.files['dataset']
            original_filename=file.filename
            filename=f"{int(time.time())}_{original_filename}"
            filepath=os.path.join(app.config["UPLOAD_FOLDER"],filename)
            file.save(filepath)
        else:
            filename=request.form.get('existing_files')
            filepath=os.path.join(app.config["UPLOAD_FOLDER"],filename)
            original_filename=filename
        if not filename:
            return render_template('upload.html', existing_files=os.listdir(app.config["UPLOAD_FOLDER"]), error="Please upload or select a file.")

        df=pd.read_csv(filepath)
        data_preview=df.head().to_html()

        with open('temp_df.pkl','wb') as f:
            pickle.dump(df,f)
        
        
        return render_template('select_target.html',
                               columns=df.columns,
                               filename=original_filename,
                               data=data_preview)
    existing_files = [f for f in os.listdir(app.config["UPLOAD_FOLDER"]) if f.endswith('.csv')]
    return render_template('upload.html',existing_files=existing_files)
@app.route('/detect',methods=['POST'])
def detect_problem():
    load_dotenv()
    OPENROUTER_API_KEY=os.getenv('OPENROUTER_API_KEY')

    target_col=request.form['target']
    problem_type=request.form.get("problem_type")
    prompt=request.form.get("llm_prompt")


    with open('target_col.pkl', 'wb') as f:
        pickle.dump(target_col, f)

    with open('temp_df.pkl','rb') as f:
        df=pickle.load(f)
    target_data=df[target_col]

    if not problem_type:
        datetime_cols=df.select_dtypes(include=['datetime64','datetime64[ns]','object']).columns
        time_series_detected=False
        for col in datetime_cols:
            try:
                parsed=pd.to_datetime(df[col],errors='raise')
                if parsed.ismonotonic_increasing:
                    time_series_detected=True
                    time_col=col
                    break
            except:
                continue
        if time_series_detected and pd.api.types.is_numeric_dtype(target_data):
            problem_type="Time Series Regression"
        elif pd.api.types.is_numeric_dtype(target_data):
            unique_values= target_data.nunique()/len(target_data)
            is_discrete=pd.api.types.is_integer_dtype(target_data)
            if (unique_values < 0.05 and target_data.nunique()<20) or is_discrete and target_data.nunique()<10:
                problem_type="Classification"
            else:
                problem_type="Regression"
        else:
            problem_type="Classification"   

    
    summary={
        "rows":df.shape[0],
        "columns":df.shape[1],
        "missing_values":int(df.isnull().sum().sum()),
        "numerical_columns":df.select_dtypes(include='number').columns.tolist(),
        "categorical_columns": df.select_dtypes(include='object').columns.tolist(),
        "likely_task": problem_type,
        "column_types": df.dtypes.astype(str).to_dict()
    }
    default_prompt = f"""We have a dataset with {summary['rows']} rows and {summary['columns']} columns.
It contains {summary['missing_values']} missing values.
Likely ML task: **{summary['likely_task']}**

Numerical columns: {summary['numerical_columns']}
Categorical columns: {summary['categorical_columns']}

What Machine learning model would you suggest? Provide a short explanation, pros, cons, and alternative models in english with proper formatting."""


    prompt=request.form.get("llm_prompt") or default_prompt

    token_count=int(len(prompt.split())*4.5)
    max_tokens=100000
    if token_count>max_tokens:
        prompt=prompt.split()[:max_tokens]
        prompt=" ".join(prompt)+'.....(truncated)'

    
    headers={
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    } 



    models_to_try=[
        "google/gemma-3-27b-it:free",
        "mistralai/mistral-small-3.2-24b-instruct:free",
        "meta-llama/llama-4-maverick:free"
    ]

    fallback_used=True
    llm_suggestion=""


    #few-shot prompting to give model some training 
    few_shot_messages = [
    {"role": "system", "content": "You are a helpful ML expert assistant."},
    {"role": "user", "content": "Dataset has 1000 rows, 10 columns, 2 missing values. Task: Classification. Numerical: ['age', 'income'], Categorical: ['gender']. Suggest ML model."},
    {"role": "assistant", "content": "Use **Random Forest**. It's robust to overfitting and handles categorical features well. Alternatives: Logistic Regression (simple), XGBoost (better performance, slower)."},
    
    {"role": "user", "content": "Dataset with 500 rows, 8 columns, 3 missing. Task: Regression. Numerical: ['years_exp', 'salary'], Categorical: ['education']. Suggest ML model."},
    {"role": "assistant", "content": "**Linear Regression** is a good baseline. If the relationship is non-linear, use **XGBoost Regressor** or **Random Forest Regressor**. Avoid models needing large data."}
    ]

    messages = few_shot_messages + [{"role": "user", "content": prompt}]

    for model_name in models_to_try:

        data={
            "model": model_name,
            "messages":messages
        }
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                data=json.dumps(data)
            )
            result = response.json()
            #print(result)  # <-- Add this line to see the full response in your terminal

            # Check for error in the API response
            if 'error' in result:
                llm_suggestion = f"LLM API Error: {result['error'].get('message', str(result['error']))}"
                continue
            elif 'choices' in result and result['choices']:
                llm_suggestion = result['choices'][0]['message']['content']
                fallback_used=False
                with open('llm_response.txt','w',encoding='utf-8') as f:
                    f.write(llm_suggestion)
                break
            else:
                llm_suggestion = "Unexpected API response format."
                continue
        except Exception as e:
            llm_suggestion = f"Error occurred while fetching LLM response: {str(e)}"
            continue
    if fallback_used:
        llm_suggestion="""
Fallback: Try using **Random Forest** for classification,  
or **Linear Regression** for linear regression.  
These models work well in general tabular ML problems.
"""
    return render_template('result.html',
                           file_name="Previously Uploaded",
                           data=df.head().to_html(),
                           problem_type=problem_type,
                           target_col=target_col,
                           llm_suggestion=llm_suggestion,
                           default_prompt=prompt,
                           token_count=token_count,
                           fallback_used=fallback_used)
@app.route('/train',methods=['POST','GET'])
def train_model():
    
   # print(request.form)
    with open('temp_df.pkl','rb') as f:
        df=pickle.load(f)
    with open('target_col.pkl','rb') as f:
        target_col=pickle.load(f)
    if not os.path.exists('llm_response.txt'):
        return render_template('train_result.html', model="N/A", score="N/A", metric="No LLM response available. Please run detection first and ensure the LLM API is working.")
    with open('llm_response.txt','r',encoding='utf-8') as f:
        llm_text=f.read()
    llm_text = llm_text.replace('*', '')
    from utils import extract_supported_models
    options=extract_supported_models(llm_text)


    if not options:
        options = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'Linear Regression']
    if request.method=='GET':
        return render_template('train.html',models=options)
    

    selected=request.form.getlist('selected_models')
    if not selected:
        return render_template('train.html',result=[],error="Please select at least one model")
        
    X=pd.get_dummies(df.drop(columns=[target_col]))
    y=df[target_col]
    if y.dtype=='object':
        y=y.astype('category').cat.codes
    
    results=[]
    for model_name in selected:

        if model_name == 'Logistic Regression':
            model=LogisticRegression()
        elif model_name == 'Decision Tree':
            model = DecisionTreeClassifier()
        elif model_name == 'Random Forest':
            model = RandomForestClassifier()
        elif model_name == 'Linear Regression':
            if not pd.api.types.is_numeric_dtype(y) or y.nunique() < 10:
                results.append({"model": model_name, "score": "Error", "metric": "Target not suitable for regression"})
                continue
            y = pd.to_numeric(y, errors='coerce')
            X = X[~y.isna()]
            y = y[~y.isna()]
            model = LinearRegression()
        elif model_name == 'Support Vector Machine':
            model= SVC(probability=True)
        elif model_name == 'K-Nearest Neighbors':   
            n_neighbors = min(5, len(X))  # Choose 5 or fewer based on data
            if n_neighbors < 1:
                results.append({"model": model_name, "score": "Error", "metric": "Not enough data for KNN"})
                continue
            model = KNeighborsClassifier(n_neighbors=n_neighbors)
        elif model_name == 'Naive Bayes':
            model= GaussianNB()
        elif model_name == 'XGBoost':
            from xgboost import XGBClassifier, XGBRegressor
            if y.dtype == 'object':
                if y.nunique() == 2:
                    y = y.map({label: idx for idx, label in enumerate(y.unique())})
                else:
                    y = y.astype('category').cat.codes
            if y.nunique() <= 2:
                model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
            else:
                model = XGBRegressor()
        else:
            continue
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=42)
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            metrics={}
            if model_name == 'Linear Regression':
                metrics["R²"] = round(r2_score(y_test, y_pred), 3)
                metrics["MAE"] = round(mean_absolute_error(y_test, y_pred), 3)
                metrics["RMSE"] = round(np.sqrt(mean_squared_error(y_test, y_pred)), 3)
            else:
                metrics["Accuracy"] = round(accuracy_score(y_test, y_pred), 3)
                metrics["Precision"] = round(precision_score(y_test, y_pred, average='weighted', zero_division=0), 3)
                metrics["Recall"] = round(recall_score(y_test, y_pred, average='weighted'), 3)
                metrics["F1-Score"] = round(f1_score(y_test, y_pred, average='weighted'), 3)
                try:
                    metrics["AUC"]=round(roc_auc_score(y_test, model.predict_proba(X_test),multi_class='ovr'), 3)
                except:
                    metrics["AUC"] = "N/A"
            from utils import save_confusion_matrix, save_roc_curve
            conf_path=save_confusion_matrix(y_test, y_pred, model_name.replace(" ",""))
            if hasattr(model, "predict_proba"):
                try:
                    roc_path=save_roc_curve(y_test, model.predict_proba(X_test), model_name.replace(" ",""))
                except:
                    roc_path = "N/A"
            else:
                roc_path = "N/A"
            results.append({
                "model": model_name,
                "metrics":metrics,
                "conf_img":conf_path,
                "roc_img":roc_path,
                "model_obj": model}) 
        except Exception as e:
            results.append({"model": model_name, "score": "Error", "metric": str(e)})
    print(results)
    
    for r in results:
        if 'metrics' not in r:
            continue 
        model_name=r['model']
        m=r['metrics']

        if 'Accuracy' in m:
            comparison_metrics.append({
                'model':model_name,
                'accuracy': m.get('Accuracy', 0),
                'precision': m.get('Precision', 0),
                'recall': m.get('Recall', 0),   
                'f1_score': m.get('F1-Score', 0),
                'auc': m.get('AUC', 0) if m.get('AUC')!= "N/A" else 0,
                'type': 'classification'
            })
        elif 'R²'in m:
            comparison_metrics.append({
                'model':model_name,
                'r2':m.get('R²',0),
                'mae':m.get('MAE',0),
                'rmse':m.get('RMSE',0),
                'type':'regression'
            })

    if not comparison_metrics:
        return render_template("train_result.html",
                            results=results,
                            comparison_metrics=[],
                            best_model="N/A",
                            best_score="N/A",
                            show_warning=True,
                            warning_message="All models failed to train. Please check your data and try again.")
    best_model = max(comparison_metrics, key=average_score)


    #for adding it to use existing models to get predictions for new data given by user
    final_model=None
    for r in results:
        if r['model']==best_model['model']:
            final_model=r.get('model_obj')
            break
    if final_model:
        joblib.dump(final_model,'best_model.pkl')



    #for x-AI (explainable ai: using shap to get which features contribute more or less info only on tree types models)

    show_warning=False
    if best_model.get('type')=='classification' and best_model.get('accuracy',1)<0.6:
        show_warning=True
    elif best_model.get('type')=='regression' and best_model.get('r2',1)<0.5:
        show_warning=True

    

    return render_template('train_result.html',
                           results=results,
                           comparison_metrics=comparison_metrics,
                           best_model=best_model['model'],
                           best_score=round(average_score(best_model),3),
                           show_warning=show_warning
    )


@app.route('/predict',methods=['GET','POST'])
def predict_new():
    if request.method=='POST':
        file=request.files['new_data']
        if file and file.filename.endswith('.csv'):
            new_df=pd.read_csv(file)
            model=joblib.load('best_model.pkl')

            with open('temp_df.pkl', 'rb') as f:
                df=pickle.load(f)
            with open('target_col.pkl', 'rb') as f: 
                target_col=pickle.load(f)
            X_ref=pd.get_dummies(df.drop(columns=[target_col]))
            new_X=pd.get_dummies(new_df)

            for col in X_ref.columns:
                if col not in new_X.columns:
                    new_X[col] = 0
            new_X = new_X[X_ref.columns]
            predictions=model.predict(new_X)
            new_df['Predictions'] = predictions
            output_file = f"predictions_{int(time.time())}.csv"
            new_df.to_csv(output_file, index=False)
        return send_file(output_file, as_attachment=True, download_name=output_file)
    return render_template('predict.html')
import csv 
from flask import send_file
@app.route('/download')
def download_results():
    csv_path='model_results.csv'
    with open(csv_path,'w',newline='') as csvfile:
        writer=csv.writer(csvfile)
        writer.writerow(['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC'])
        for r in comparison_metrics:
            writer.writerow([
                r['model'],
                r.get('accuracy', 'N/A'),
                r.get('precision', 'N/A'),
                r.get('recall', 'N/A'),
                r.get('f1_score', 'N/A'),
                r.get('auc', 'N/A')
            ])
    name = f"ModelComparison_{int(time.time())}.csv"
    return send_file(csv_path, as_attachment=True, download_name=name)

if __name__=='__main__':
    app.run(debug=True)
#python app.py