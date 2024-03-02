import streamlit as st
st.set_page_config(layout="wide")
st.title('Academic Success Predictor')

st.markdown("""
## Welcome to the Academic Success Predictor!

This interactive tool is designed to help educators, students, and administrators predict academic outcomes based on a variety of factors. Utilizing advanced machine learning algorithms, this app can forecast the likelihood of student success, providing valuable insights that can inform educational strategies and interventions.

### How It Works
- **Select a Model**: Choose from 4 pre-trained models, each tailored to interpret different data sets and capture various indicators of student performance.
- **Input Data**: Input student data manually or select from an existing dataset to make predictions.
- **Receive Predictions**: The app will process the data through the selected predictive model and provide you with the probability of academic success or failure.
- **Explore Visualizations**: After making predictions, visit the Visualization tab in the sidebar to see detailed graphs and figures that explain the model's decision making. 

### Why Predict Academic Success?
Predicting academic success is a cruicial first step towards **personalized** education. By understanding potential outcomes, educators can:

- **Target Interventions**: Allocate resources and support to students who are predicted to require additional assistance.
- **Personalize Learning**: Tailor educational experiences to individual student needs, maximizing their chances for success.
- **Monitor Progress**: Keep track of the likelihood of success over time and adjust strategies as students develop and circumstances change.
- **Foster Academic Growth**: Use insights to create an environment where all students have the opportunity to succeed.

### Get Started
To begin, select a predictive model from the dropdown on the left. Next, choose your method for inputting data into the model. You can either manually enter student information or select from a preloaded dataset.

Once you've input the data, hit the "Predict" button and watch the model evaluate the likelihood of academic success. It's that simple!



""", unsafe_allow_html=True)

st.sidebar.success("Select a page to proceed.")

