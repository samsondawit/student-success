import streamlit as st
import pandas as pd
from joblib import load
from PIL import Image
import sqlite3
import os
import warnings

current_script_dir = os.path.dirname(os.path.abspath(__file__))
logo_path = Image.open(os.path.join(current_script_dir, '..', 'logo.png'))
st.set_page_config(page_title="Vizualizations", page_icon=logo_path, layout="wide")

st.title("Student Success Predictor")
st.markdown("""
            ## About the Models
Each predictive model in our app has been trained on historical data, capturing complex patterns and relationships that can forecast academic outcomes. The models take into account academic, social, socio-economic, and demographic features. Here are the models you can choose from:
""")

st.write("**Model 1**: This is a `logistic regression model` trained to predict *graduation and drop out rate* with a `92% accuracy.` ")
# with st.expander("Detailed Description for Model 1"):
#     st.write("""
#                 This model was trained on the ["Predict Students' Dropout and Academic Success"](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success) dataset. It contains 4424 students and 36 features. The target is to predict whether a student will graduate or drop out from university. After feature engineering, 3 columns were dropped due to multicollinearity, and the data point was reduced to 3630 (3630x33). This is because all students with "enrolled" as the target were dropped since our primary objective is to predict graduation or dropout rates.
#                 The dataset contains socio-economic, demographic, macro, and enrollment data, as well as academic features.  14 machine learning models were trained on this dataset and hyperparameter tuning was carried out for optimal performance. Logistic regression performed best with a 92% score across all accuracy, precision, recall, and f1 score metrics. 
#                 For the logistic regression model, academic and socio-economic features were important in making a prediction, as seen by its feature importance graph in the visualization page. The model training notebook can be found [here](https://github.com/samsondawit/student-success/blob/main/Predicting%20Gradutation-Dropout%204.4k/Student%20Graduation.ipynb).
#                 """)
st.write("**Model 2**: This is a `support vector classifier model` trained to predict pass or fail rate in a Math class with a `91% accuracy`.")
# with st.expander("Detailed description for Model 2"):
#     st.write("""
#                 This model was trained on the ["Student Performance"](https://archive.ics.uci.edu/dataset/320/student+performance) dataset from a secondary education institute. The data attributes include student grades, demographic, social and school-related features and it was collected by using school reports and questionnaires. 
#                 Two datasets are provided regarding the performance in two distinct subjects: Mathematics (mat) and Portuguese language (por). This model was trained on the Math dataset to predict pass or fail in the math class. It has 32 features and 395 data points. 14 machine learning models were trained on this dataset and hyperparameter tuning was carried out for optimal performance.
#                 Support Vector Classifier (SVC) with a poly kernel performed best with 91% accuracy across all classification metrics. 
#                 For the SVC model, academic features such as grades of 1st and 2nd semesters, study time, and absences were important in making predictions. 
#                 Moreover, the model considered social features such as if the student goes out a lot and how much they travel. Other features like health and the mother's job were also important. The visualizations can be found in the visualizations tab after making a prediction with this model.
#                 The model training notebook can be found [here](https://github.com/samsondawit/student-success/blob/main/Student%20Pass%20Fail%20POR/Student%20Performance%20MAT.ipynb).
#                 """)
st.write("**Model 3**: This is a `support vector classifier model` trained to predict pass or fail rate in a Portugese language class with a `95% accuracy`.")
# with st.expander("Detailed description for Model 3"):
#     st.write("""
#                 This model was trained on the ["Student Performance"](https://archive.ics.uci.edu/dataset/320/student+performance) dataset from a secondary education institute. The data attributes include student grades, demographic, social and school-related features and it was collected by using school reports and questionnaires. 
#                 Two datasets are provided regarding the performance in two distinct subjects: Mathematics (mat) and Portuguese language (por). This model was trained on the Portugese language dataset to predict pass or fail in the class. It has 32 features and 649 data points. 14 machine learning models were trained on this dataset and hyperparameter tuning was carried out for optimal performance.
#                 Support Vector Classifier (SVC) with a poly kernel performed best with 95% accuracy across all classification metrics. 
#                 Academic features such as grades of 1st and 2nd semester, and absences were important in making predictions for the model. Interestingly, the model considers the health of the student, the quality of their family relation, whether or not they want to pursue higher education, and how much free time they have are also features important for the model.
#                 The visualizations can be found in the Visualizations tab after making a prediction with this model.
#                 The model training notebook can be found [here](https://github.com/samsondawit/student-success/blob/main/Student%20Pass%20Fail%20POR/Student%20Performance%20POR.ipynb).
#                 """)
st.write("**Model 4**: This is a `stacking classifier model` with adaboost, random forest, and logistic regression as base learners and logistic regression as final estimator, trained to predict academic performance based on GPA. It has an `83% accuracy`.")
# with st.expander("Detailed description for Model 4"):
#     st.write("""
#                The dataset was trained on the [Higher Education Students Performance Evaluation] (https://archive.ics.uci.edu/dataset/856/higher+education+students+performance+evaluation) dataset. 
#                The data was collected from the Faculty of Engineering and Faculty of Educational Sciences students. The purpose is to predict students' end-of-term performances. 
#                The survey had personal questions, family questions, and education habits. It contains demographic, socio-economic and academic features. 
#                The dataset has 145 datapoints and 31 features. 
               
#                This model was trained to predict if a student will pass or fail by the end of the year from all courses based on 32 features and 649 data points. 
#                14 machine learning models were trained on this dataset and hyperparameter tuning was carried out for optimal performance.
#                The best performing model was a Stacking Classifier with adaboost, random forest, and logistic regression as base learners, and logistic regression again as the final estimator. The model had an 83% accuracy across all classification metrics. 
#                The noticably lower accuracy is because of the small dataset. This further signifies the challenges that comes with predicting academic performance and as a consequence, personalized learning, due to lack of adequate datasets.
#                 It is not possible to directly get the feature importance of a stacking model, but academic, social, and socio-economic features were most important during their predictions. The visualizations can be found in the Visualizations tab after making a prediction with this model.
#                 The model training notebook can be found [here](https://github.com/samsondawit/student-success/blob/main/Student%20performance%20-%20Turkey/Student%20performance%20-%20Turkey%20dataset.ipynb).
#                 """)
    

st.markdown("""
### Model Summaries
Explore detailed descriptions of each model by expanding their sections. Each model has unique characteristics and performance metrics.
""")

models = [
    {
        "name": "Model 1: Logistic Regression",
        "accuracy": "92%",
        "description": """
- **Type**: Logistic Regression Model
- **Objective**: Predict graduation and dropout rates
- **Accuracy**: 92%
- **Dataset**: Trained on the ["Predict Students' Dropout and Academic Success"](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success) dataset, containing 4424 students and 36 features. The dataset contains socio-economic, demographic, macro, and enrollment data, as well as academic features.
- **Preprocessing**: After feature engineering, 3 columns were dropped due to multicollinearity, and all students with "enrolled" as the target were dropped since our primary objective is to predict graduation or dropout rates, reducing data points to 3630 (3630x33).
- **Insights**: Academic and socio-economic features were crucial for prediction. Logistic regression showed the best performance with a 92% score across accuracy, precision, recall, and F1 score metrics.
- **Visualization & Notebook**: Relevant graphs about the model and data is found in the visualization page. Model training notebook [here](https://github.com/samsondawit/student-success/blob/main/Predicting%20Gradutation-Dropout%204.4k/Student%20Graduation.ipynb).
        """,
    },
    {
        "name": "Model 2: Support Vector Classifier (Math)",
        "accuracy": "91%",
        "description": """
- **Type**: Support Vector Classifier Model
- **Objective**: Predict pass or fail rate in a Math class
- **Accuracy**: 91%
- **Dataset**: Based on the ["Student Performance"](https://archive.ics.uci.edu/dataset/320/student+performance) dataset from a secondary education institute, focusing on the Mathematics performance.
- **Features & Data Points**: Includes 32 features and 395 data points. The data attributes include student grades, demographic, social and school-related features.
- **Key Factors**: Academic performance (grades, study time), social behaviors (going out, travel), and other factors like health and parent's job were influential.
- **Visualization & Notebook**: Relevant graphs and figures for the model and dataset are found in the vizualizations page post-prediction. Training notebook [here](https://github.com/samsondawit/student-success/blob/main/Student%20Pass%20Fail%20POR/Student%20Performance%20MAT.ipynb).
        """,
    },
    {
        "name": "Model 3: Support Vector Classifier (Portuguese)",
        "accuracy": "95%",
        "description": """
- **Type**: Support Vector Classifier Model
- **Objective**: Predict pass or fail rate in a Portuguese language class
- **Accuracy**: 95%
- **Dataset**: Utilizes the ["Student Performance"](https://archive.ics.uci.edu/dataset/320/student+performance) dataset from a secondary education institution, with a focus on Portuguese language performance. The data attributes include student grades, demographic, social and school-related features. It has 32 features and 649 data points.
- **Insights**: The model considers academic records, health status, family relations, free time and aspirations for higher education as key features.
- **Visualization & Notebook**: Relevant visualizations available after predictions. Training notebook [here](https://github.com/samsondawit/student-success/blob/main/Student%20Pass%20Fail%20POR/Student%20Performance%20POR.ipynb).
        """,
    },
    {
        "name": "Model 4: Stacking Model",
        "accuracy": "83%",
        "description": """
- **Type**: Stacking Model with Adaboost, Random Forest, and Logistic Regression as base learners and Logistic Regression as the final estimator.
- **Objective**: Predict academic performance based on GPA.
- **Accuracy**: 83% (The lower accuracy due to size of data, emphasizing the need for bigger and more holistic datasets.)
- **Dataset**: The model is trained on the [Higher Education Students Performance Evaluation](https://archive.ics.uci.edu/dataset/856/higher+education+students+performance+evaluation) dataset, focusing on students from the Faculty of Engineering and Faculty of Educational Sciences.
- **Dataset Characteristics**: Contains 145 data points and 31 features, highlighting the challenge of predicting academic performance due to small dataset sizes.
- **Insights**:  It is not possible to directly get the feature importance of a stacking model, but academic, social, and socio-economic features were most important during the model's predictions.
- **Visualization & Notebook**: Relevant graphs and figures for the model and dataset are found in the vizualizations page post-prediction. Training notebook [here](https://github.com/samsondawit/student-success/blob/main/Student%20performance%20-%20Turkey/Student%20performance%20-%20Turkey%20dataset.ipynb).
        """,
    },
]

for model in models:
    with st.expander(f"{model['name']} (Accuracy: {model['accuracy']}) - Detailed Description"):
        st.markdown(model["description"])



st.markdown("---")
st.write("## Start making predictions: Choose a model and data input method")
st.write("Please select a prediction model and choose your preferred method to input data.")
st.sidebar.success("After making a prediction, you may see relevant graphs and figures in the Vizualization tab.")

current_script_dir = os.path.dirname(os.path.abspath(__file__))
model_paths = {
    'Model 1': os.path.join(current_script_dir, '..', '..', 'Predicting Graduation Dropout 4.4k', 'model', 'best_model.joblib'),
    'Model 2': os.path.join(current_script_dir, '..', '..', 'Student Pass Fail POR', 'models', 'best_model_por.joblib'), 
    'Model 3': os.path.join(current_script_dir, '..', '..', 'Student Pass Fail POR', 'models', 'best_model_mat.joblib'),
    'Model 4': os.path.join(current_script_dir, '..', '..', 'Student performance - Turkey', 'Model', 'best_model.joblib')
}

if 'last_selected_model' not in st.session_state:
    st.session_state['last_selected_model'] = None
if 'prediction_made' not in st.session_state:
    st.session_state['prediction_made'] = False
col1, col2 = st.columns(2)
with col1:
    selected_model_name = st.radio('Select a model:', list(model_paths.keys()))

with col2:
    if selected_model_name == 'Model 1':
        st.info("This model predicts if a student will graduate or drop out from university with 92% accuracy." )
        passing, failing = 'graduating', 'dropping out.'
    elif selected_model_name == 'Model 2':
        st.info("This model predicts if a student will pass their Math class with 91% accuracy." )
        passing, failing = 'passing', 'failing Math.'
    elif selected_model_name == 'Model 3':
        st.info("This model predicts if a student will pass their Portuguese class with 95% accuracy." )
        passing, failing = 'passing', 'failing Portuguese.'
    else:
        st.info("This model predicts if a student will pass or fail the current academic year with 83% accuracy." )
        passing, failing = 'passing', 'failing this academic year.'
st.session_state['selected_model_name'] = selected_model_name
st.session_state['model_paths'] = model_paths

#added not to show viz without choosing and bug fixing
if selected_model_name != st.session_state['last_selected_model']:
    st.session_state['prediction_made'] = False
    st.session_state['last_selected_model'] = selected_model_name


data_input_method = st.selectbox("Choose your data input method:", ["Choose from dataset", "Input manually"])

model = load(model_paths[selected_model_name])



studentdb = os.path.join(current_script_dir, '..', 'student_database.db')

# conn = sqlite3.connect(studentdb)
# table_names = ['Model1Data', 'Model2Data', 'Model3Data', 'Model4Data']


# csv_files = [
#     os.path.join(current_script_dir, '..', '..', 'Predicting Gradutation-Dropout 4.4k', 'data', 'x_test_4.4k.csv'),
#     os.path.join(current_script_dir, '..', '..', 'Student Pass Fail POR', 'data', 'x_test_MAT.csv'),
#     os.path.join(current_script_dir, '..', '..', 'Student Pass Fail POR', 'data', 'x_test_POR.csv'),
#     os.path.join(current_script_dir, '..', '..', 'Student performance - Turkey', 'Data', 'x_test_Turkey.csv')
# ]

# for csv_file, table_name in zip(csv_files, table_names):
#     df = pd.read_csv(csv_file)
#     df.to_sql(table_name, conn, if_exists='replace', index=False)

# conn.close()

model_to_table = {
    'Model 1': 'Model1Data',
    'Model 2': 'Model2Data',
    'Model 3': 'Model3Data',
    'Model 4': 'Model4Data',
}

table_names = list(model_to_table.values())

def fetch_data(table_name):
    try:
        with sqlite3.connect(studentdb) as conn:  
            query = f"SELECT * FROM {table_name}"
            df = pd.read_sql(query, conn)
        return df
    except sqlite3.Error as e:
        st.error(f"Database error: {e}")
        st.stop()

selected_table = model_to_table[selected_model_name]
    

scaler_path = os.path.join(current_script_dir, '..', '..', 'Predicting Graduation Dropout 4.4k', 'model', 'model_scaler.joblib')

if selected_model_name == 'Model 1':
    scaler = load(scaler_path)
else:
    scaler = None
    
if data_input_method == "Choose from dataset":
    df = fetch_data(selected_table)
    st.write("Dataset Preview:", df)

    options = st.multiselect('Select records for prediction (you can select multiple):', df.index, format_func=lambda x: f"Student {x}")

    if options:
        selected_records = df.loc[options]
        st.write("Selected Records for Prediction:", selected_records)

        if st.button('Predict'):
            if not options:
                st.error("Please select at least one record for prediction.")
            else:
                st.session_state['prediction_made'] = True
                if scaler is not None:
                    selected_records_for_scaling = selected_records.apply(pd.to_numeric, errors='ignore')
                    scaled_input = scaler.transform(selected_records_for_scaling)
                else:
                    scaled_input = selected_records
                    
                probabilities = model.predict_proba(scaled_input)
                

                formatted_predictions = ""
                for index, (prob_fail, prob_pass) in zip(options, probabilities):
                    pass_percent = prob_pass * 100
                    fail_percent = prob_fail * 100 
                    if pass_percent > fail_percent:
                        formatted_predictions += f'<div style="color:green; font-size:20px;">Student {index} has a {pass_percent:.2f}% chance of {passing} and a {fail_percent:.2f}% chance of {failing}</div>'
                    else:
                        formatted_predictions += f'<div style="color:red; font-size:20px;">Student {index} has a {pass_percent:.2f}% chance of {passing} and a {fail_percent:.2f}% chance of {failing}</div>'

                st.markdown(formatted_predictions, unsafe_allow_html=True)
                
if data_input_method == "Input manually":
    num_columns = 5  
    cols = st.columns(num_columns)
    features = fetch_data(selected_table).columns.tolist()

    col1, col2  = st.columns(2)
    with col1:
        if st.button('Fill Random'):
            df = fetch_data(selected_table)
            random_row = df.sample(n=1).iloc[0]
            for feature in features:
                st.session_state[feature] = str(random_row[feature])
                
    with col2:
        if st.button('Clear Inputs'):
            for feature in features:
                st.session_state[feature] = ""

    input_values = {}
    for i, feature in enumerate(features):
        with cols[i % num_columns]:
            input_values[feature] = st.text_input(feature, key=feature)
            
    with col1:
        predict_button = st.button('Predict')
        
    if predict_button:
        if any(value == "" for value in input_values.values()):
            st.error("Please fill in all fields before predicting.")
        else:
            st.session_state['prediction_made'] = True

            input_df = pd.DataFrame([input_values], columns=features)
            input_df = input_df.apply(pd.to_numeric, errors='ignore')  
            if scaler is not None:
                scaled_input = scaler.transform(input_df)
            else:
                scaled_input = input_df
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                probabilities = model.predict_proba(scaled_input)[0]  
                
                st.session_state['selected_model'] = selected_model_name
                st.session_state['selected_data'] = selected_records if data_input_method == "Choose from dataset" else input_df
                
            if probabilities[1] > probabilities[0]:
                formatted_predictions = f'<div style="color:green; font-size:25px;">This student has a {probabilities[1]*100:.2f}% chance of {passing} and a {probabilities[0]*100:.2f}% chance of {failing}</div>'
            else: 
                formatted_predictions = f'<div style="color:red; font-size:25px;">This student has a {probabilities[1]*100:.2f}% chance of {passing} and a {probabilities[0]*100:.2f}% chance of {failing}</div>'

            st.markdown(formatted_predictions, unsafe_allow_html=True)
