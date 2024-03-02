import streamlit as st
import pandas as pd
from joblib import load
import sqlite3
import os
import warnings

st.title("Student Success Predictor")
st.markdown("""
            ### About the Models
Each predictive model in our app has been trained on historical data, capturing complex patterns and relationships that can forecast academic outcomes. The models take into account academic, social, socio-economic, and demographic features. Here are the models you can choose from:

- **Model 1**: This model was trained on a dataset containing 4,424 students. It has been trained to predict *graduation and drop out rate*.
- **Model 2**: This model has been trained to predict pass or fail rate in a math class.
- **Model 3**: This model has been trained to predict pass or fail rate in a math class.
- **Model 4**: This model has been trained to predict academic success or failure based on GPA.
---""")
st.write("Please select a prediction model and choose your preferred method to input data.")
st.sidebar.success("After making a prediction, you may see relevant graphs and figures in the Vizualization tab.")

current_script_dir = os.path.dirname(os.path.abspath(__file__))
model_paths = {
    'Model 1': os.path.join(current_script_dir, '..', '..', 'Predicting Gradutation-Dropout 4.4k', 'model', 'best_model.joblib'),
    'Model 2': os.path.join(current_script_dir, '..', '..', 'Student Pass Fail POR', 'models', 'best_model_por.joblib'), 
    'Model 3': os.path.join(current_script_dir, '..', '..', 'Student Pass Fail POR', 'models', 'best_model_mat.joblib'),
    'Model 4': os.path.join(current_script_dir, '..', '..', 'Student performance - Turkey', 'Model', 'best_model.joblib')
}


selected_model_name = st.radio('Select a model:', list(model_paths.keys()))
st.write("You selected:", selected_model_name)
st.session_state['selected_model_name'] = selected_model_name
st.session_state['model_paths'] = model_paths

data_input_method = st.selectbox("Choose your data input method:", ["Choose from dataset", "Input manually"])

model = load(model_paths[selected_model_name])



studentdb = 'student_database.db'
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

scaler_path = os.path.join(current_script_dir, '..', '..', 'Predicting Gradutation-Dropout 4.4k', 'model', 'model_scaler.joblib')

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
                for index, (prob_pass, prob_fail) in zip(options, probabilities):
                    pass_percent = prob_pass * 100
                    fail_percent = prob_fail * 100
                    if pass_percent > fail_percent:
                        formatted_predictions += f'<div style="color:green; font-size:20px;">Student {index} has a {pass_percent:.2f}% chance of success and a {fail_percent:.2f}% chance of failure.</div>'
                    else:
                        formatted_predictions += f'<div style="color:red; font-size:20px;">Student {index} has a {pass_percent:.2f}% chance of success and a {fail_percent:.2f}% chance of failure.</div>'

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
                formatted_predictions = f'<div style="color:green; font-size:25px;">This student has a {probabilities[1]*100:.2f}% chance of success and a {probabilities[0]*100:.2f}% chance of failure.</div>'
            else: 
                formatted_predictions = f'<div style="color:red; font-size:25px;">This student has a {probabilities[1]*100:.2f}% chance of success and a {probabilities[0]*100:.2f}% chance of failure.</div>'

            st.markdown(formatted_predictions, unsafe_allow_html=True)
