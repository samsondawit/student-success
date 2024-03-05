import streamlit as st
from PIL import Image
from joblib import load
import os

current_script_dir = os.path.dirname(os.path.abspath(__file__))
logo_path = Image.open(os.path.join(current_script_dir, '..', 'logo.png'))
st.set_page_config(page_title="Vizualizations", page_icon=logo_path, layout="wide")

def displayimage(image_path):
    img = Image.open(image_path)
    st.image(img)
    

if 'prediction_made' in st.session_state and st.session_state['prediction_made']:
    selected_model_name = st.session_state['selected_model_name']
    model_paths = st.session_state['model_paths']
    model_path = model_paths[selected_model_name]
    model = load(model_path)
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    
    if selected_model_name == 'Model 1':
        st.title(f"Vizualizations for {selected_model_name} - Predicting Student Graduation or Drop Out")
        tab1, tab2 = st.tabs(["About the Model", "About the Data"])
        with tab1:
            st.write("### Feature Importance")
            st.markdown("""
            This chart illustrates the importance of each feature in the prediction made by the model.
            Features that are higher on the chart have a greater influence on the model's predictions.
            For this model, mostly academic features are important but there are a few socio-economic features as well.
            """)
            displayimage(os.path.join(current_script_dir, '..', '..', 'Predicting Graduation Dropout 4.4k', 'figures', 'feature importance.png'))
            st.write("\n")
            col1, col2 = st.columns(2)
        
            with col1:
                st.write("### Model Metrics Bar Chart")
                st.markdown("""
                The bar chart represents various metrics used to evaluate the model's performance.
                These metrics include accuracy, precision, recall, and F1 score.
                """)
                displayimage(os.path.join(current_script_dir, '..', '..', 'Predicting Graduation Dropout 4.4k', 'figures', 'best model metrics.png'))
            with col2:
                st.write("### Confusion Matrix")
                st.markdown("""
                The confusion matrix provides a visual representation of the model's performance,
                showing the number of correct and incorrect predictions broken down by each class.
                Top right is false negative predictions and bottom left is false positive predictions.
                """)
                displayimage(os.path.join(current_script_dir, '..', '..', 'Predicting Graduation Dropout 4.4k', 'figures', 'best model confusion matrix.png'))
        with tab2:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write("### Target Distribution")
                st.markdown("""
                This bar chart shows the distribution of the target variable within the dataset.
                """)
                displayimage(os.path.join(current_script_dir, '..', '..', 'Predicting Graduation Dropout 4.4k', 'figures', 'target distribution.png'))
            with col2:
                st.write("### Gender Distribution")
                st.markdown("""
                Below is the gender distribution of the dataset.
                """)
                displayimage(os.path.join(current_script_dir, '..', '..', 'Predicting Graduation Dropout 4.4k', 'figures', 'gender distribution.png'))
                
            with col3:
                st.write("### Maritial Status Distribution")
                st.markdown("""
                Below is the maritial status distribution of the dataset.
                """)
                displayimage(os.path.join(current_script_dir, '..', '..', 'Predicting Graduation Dropout 4.4k', 'figures', 'Maritial Status.png'))
            with st.expander("### See overall correlation matrix"):
                # st.write("### Overall Correlation Matrix")
                st.markdown("""
                The correlation matrix illustrates the relationship between all features.
                A high positive or negative value indicates a strong relationship.
                """)
                displayimage(os.path.join(current_script_dir, '..', '..', 'Predicting Graduation Dropout 4.4k', 'figures', 'correlation heatmap.png'))

            st.write("### Correlation Matrix by Feature Type")
            st.markdown("""
            These correlation matrices show relationships between academic, socio-economic, and demographic features.
            """)
            col1, col2 = st.columns(2)
            
            displayimage(os.path.join(current_script_dir, '..', '..', 'Predicting Graduation Dropout 4.4k', 'figures', 'academic data - correlation matrix.png'))
            with col1:
                displayimage(os.path.join(current_script_dir, '..', '..', 'Predicting Graduation Dropout 4.4k', 'figures', 'socio-economic features -correlation heatmap.png'))
            with col2:
                displayimage(os.path.join(current_script_dir, '..', '..', 'Predicting Graduation Dropout 4.4k', 'figures', 'demographic features - correlation matrix.png'))

            st.write("### Correlation Comparison")
            st.markdown("""
            These correlation graphs show the features that have high correlation with the target. The first graph shows the top 10 highest features that have the highest correlation with target, and the next two show the top socio-economic and academic features that have the highest correlation with the target. 
            """)
            col1, col2, col3 = st.columns(3)
            with col1:
                displayimage(os.path.join(current_script_dir, '..', '..', 'Predicting Graduation Dropout 4.4k', 'figures', 'top ten highest corr graph.png'))
            with col2:
                displayimage(os.path.join(current_script_dir, '..', '..', 'Predicting Graduation Dropout 4.4k', 'figures', 'top 10 highest socio economic features corr graph.png'))
            with col3:
                displayimage(os.path.join(current_script_dir, '..', '..', 'Predicting Graduation Dropout 4.4k', 'figures', 'top 10 highest academic features corr graph.png'))
            st.write("### Comparative Bar Chart of All Models")
            st.markdown("""
            This bar chart compares the performance metrics across all 14 models trained on this dataset. Hyperparameter tuning was carried out using GridSearchCV for optimal model performance. Logistic regression showed the highest accuracy with 92% accuracy. 
            
            """)
            displayimage(os.path.join(current_script_dir, '..', '..', 'Predicting Graduation Dropout 4.4k', 'figures', 'Model comparision for chosen dataset.PNG'))

        
    elif selected_model_name in ['Model 2']:  
        st.title(f"Vizualizations for {selected_model_name} - Predict Pass or Fail in Math Class")

        tab1, tab2 = st.tabs(["About the Model", "About the Data"])
        with tab1:
            st.write("### Feature Importance")
            st.markdown("""
            This chart illustrates the importance of each feature in the prediction made by the model.
            Features that are higher on the chart have a greater influence on the model's predictions. 
            As seen for Model 2, academic data have greater influence on the model's predictions, but there are also socio economic and social features like mother's job, health of the student, study time, free time, and how much the student goes out.  
            """)
            displayimage(os.path.join(current_script_dir, '..', '..', 'Student Pass Fail POR', 'figure MAT', 'feature importance for SVC (best model).png'))
            st.write("\n")
            col1, col2 = st.columns(2)
        
            with col1:
                st.write("### Model Metrics")
                st.markdown("""
                The bar chart represents various metrics used to evaluate the model's performance.
                These metrics include accuracy, precision, recall, and F1 score.
                """)
                displayimage(os.path.join(current_script_dir, '..', '..', 'Student Pass Fail POR', 'figure MAT', 'best model performance metrics.png'))
            with col2:
                st.write("### Confusion Matrix")
                st.markdown("""
                The confusion matrix provides a visual representation of the model's performance,
                showing the number of correct and incorrect predictions broken down by each class.
                Top right is false negative predictions and bottom left is false positive predictions.
                """)
                displayimage(os.path.join(current_script_dir, '..', '..', 'Student Pass Fail POR', 'figure MAT', 'best model confusion matrix.png'))
        with tab2:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write("### Target Distribution")
                st.markdown("""
                This bar chart shows the distribution of the target variable within the dataset.
                """)
                displayimage(os.path.join(current_script_dir, '..', '..', 'Student Pass Fail POR', 'figure MAT', 'target distribution.png'))
            with col2:
                st.write("### Gender Distribution")
                st.markdown("""
                Below is the gender distribution of the dataset.
                """)
                displayimage(os.path.join(current_script_dir, '..', '..', 'Student Pass Fail POR', 'figure MAT', 'gender distribution.png'))
                
            with col3:
                st.write("### Relationship Status Distribution")
                st.markdown("""
                Below is the relationship status distribution of the dataset.
                """)

                displayimage(os.path.join(current_script_dir, '..', '..', 'Student Pass Fail POR', 'figure MAT', 'romantic relationship and academic success MAT.png'))
            with st.expander("### See overall correlation matrix"):
                # st.write("### Overall Correlation Matrix")
                st.markdown("""
                The correlation matrix illustrates the relationship between all features.
                A high positive or negative value indicates a strong relationship.
                """)
                displayimage(os.path.join(current_script_dir, '..', '..', 'Student Pass Fail POR', 'figure MAT', 'heatmap MAT.png'))

            st.write("### Correlation Matrix by Feature Type")
            st.markdown("""
            These correlation matrices show relationships between academic, social, and school related features.
            """)

            col1, col2 = st.columns(2)
            
            with col1:
                displayimage(os.path.join(current_script_dir, '..', '..', 'Student Pass Fail POR', 'figure MAT', 'academic features -  correlation matrix.png'))
            with col2:
                displayimage(os.path.join(current_script_dir, '..', '..', 'Student Pass Fail POR', 'figure MAT', 'social features - correlation map.png'))
            
            displayimage(os.path.join(current_script_dir, '..', '..', 'Student Pass Fail POR', 'figure MAT', 'school related features - correlation map.png'))

            st.write("### Correlation Comparison")
            st.markdown("""
            These correlation graphs show the features that have high correlation with the target. The first graph shows top 10 features that have the highest correlation with the target. 
            The next two show the highest social and socio-economic features that have the highest correlation with target.
            """)
            col1, col2, col3 = st.columns(3)
            with col1:
                displayimage(os.path.join(current_script_dir, '..', '..', 'Student Pass Fail POR', 'figure MAT', 'top 10 features highest corr with target.png'))
            with col2:
                displayimage(os.path.join(current_script_dir, '..', '..', 'Student Pass Fail POR', 'figure MAT', 'social features with highest corr with target.png'))
            with col3:
                displayimage(os.path.join(current_script_dir, '..', '..', 'Student Pass Fail POR', 'figure MAT', 'school related features with highest correlation with target.png'))
            st.write("### Comparative Bar Chart of All Models")
            st.markdown("""
            This bar chart compares the performance metrics across all 14 trained models trained on this dataset.
            Hyperparameter tuning was carried out using GridSearchCV for optimal model performance. Support Vector Classifier performed best with 91% accuracy. 
            
            """)
            displayimage(os.path.join(current_script_dir, '..', '..', 'Student Pass Fail POR', 'figure MAT', 'model comparison.PNG'))

        
    elif selected_model_name == 'Model 3':  
        st.title(f"Vizualizations for {selected_model_name} - Predict Pass or Fail in Portuguese language Class")

        tab1, tab2 = st.tabs(["About the Model", "About the Data"])
        with tab1:
            st.write("### Feature Importance")
            st.markdown("""
            This chart illustrates the importance of each feature in the prediction made by the model.
            Features that are higher on the chart have a greater influence on the model's predictions. As seen for Model 3, academic data have greater influence on the model's predictions, but there are also socio economic and personal features like family relations, plans for higher education, mother's and father's job. 
            """)
            displayimage(os.path.join(current_script_dir, '..', '..', 'Student Pass Fail POR', 'figure POR', 'feature importance for best model.png'))
            st.write("\n")
            col1, col2 = st.columns(2)
        
            with col1:
                st.write("### Model Metrics Bar Chart")
                st.markdown("""
                The bar chart represents various metrics used to evaluate the model's performance.
                These metrics include accuracy, precision, recall, and F1 score.
                """)
                displayimage(os.path.join(current_script_dir, '..', '..', 'Student Pass Fail POR', 'figure POR', 'best model metrics.png'))
            with col2:
                st.write("### Confusion Matrix")
                st.markdown("""
                The confusion matrix provides a visual representation of the model's performance,
                showing the number of correct and incorrect predictions broken down by each class. Top right is false negative predictions and bottom left is false positive predictions. 
                """)
                displayimage(os.path.join(current_script_dir, '..', '..', 'Student Pass Fail POR', 'figure POR', 'confusion matrix for bet model.png'))
        with tab2:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write("### Target Distribution Bar Chart")
                st.markdown("""
                This bar chart shows the distribution of the target variable within the dataset.
                """)
                displayimage(os.path.join(current_script_dir, '..', '..', 'Student Pass Fail POR', 'figure POR', 'target distribution.png'))
            with col2:
                st.write("### Gender Distribution")
                st.markdown("""
                Below is the gender distribution of the dataset.
                """)
                displayimage(os.path.join(current_script_dir, '..', '..', 'Student Pass Fail POR', 'figure POR', 'gender distribution.png'))
                
            with col3:
                st.write("### Relationship Status Distribution")
                st.markdown("""
                Below is the relationship status distribution of the dataset.
                """)

                displayimage(os.path.join(current_script_dir, '..', '..', 'Student Pass Fail POR', 'figure POR', 'romantic relationship.png'))
            with st.expander("### See overall correlation matrix"):
                # st.write("### Overall Correlation Matrix")
                st.markdown("""
                The correlation matrix illustrates the relationship between all features.
                A high positive or negative value indicates a strong relationship.
                """)
                displayimage(os.path.join(current_script_dir, '..', '..', 'Student Pass Fail POR', 'figure POR', 'heatmap POR.png'))

            st.write("### Correlation Matrix by Feature Type")
            st.markdown("""
            These correlation matrices show relationships between academic, school related, and social features.
            """)
            displayimage(os.path.join(current_script_dir, '..', '..', 'Student Pass Fail POR', 'figure POR', 'academic features.png'))
            col1, col2 = st.columns(2)
            
            with col1:
                displayimage(os.path.join(current_script_dir, '..', '..', 'Student Pass Fail POR', 'figure POR', 'school related features.png'))
            with col2:
                displayimage(os.path.join(current_script_dir, '..', '..', 'Student Pass Fail POR', 'figure POR', 'social features.png'))
            

            st.write("### Correlation Comparison")
            st.markdown("""
            These correlation graphs show the features that have high correlation with the target. The first graph show the top 10 features with the highest correlation with the target, 
            and the following 3 graphs show the highest school related, academic, and socio-economic features. 
            """)
            col1, col2 = st.columns(2)
            with col1:
                displayimage(os.path.join(current_script_dir, '..', '..', 'Student Pass Fail POR', 'figure POR', 'top ten features graph.png'))
            with col2:
                displayimage(os.path.join(current_script_dir, '..', '..', 'Student Pass Fail POR', 'figure POR', 'top school realted features with highest corr.png'))
            col1, col2 = st.columns(2)    
            with col1:
                displayimage(os.path.join(current_script_dir, '..', '..', 'Student Pass Fail POR', 'figure POR', 'top academic features graph.png'))
            with col2:
                displayimage(os.path.join(current_script_dir, '..', '..', 'Student Pass Fail POR', 'figure POR', 'top socioeconomic features with highest correlation.png'))
            
            st.write("### Comparative Bar Chart of All Models")
            st.markdown("""
            This bar chart compares the performance metrics across all 14 trained models trained on this dataset. Hyperparameter tuning was carried out using GridSearchCV for optimal model performance. Support Vector Classifier performed best with 95% accuracy. 
            """)
            displayimage(os.path.join(current_script_dir, '..', '..', 'Student Pass Fail POR', 'figure POR', 'model comparison.PNG'))

    elif selected_model_name == 'Model 4':  
        st.title(f"Vizualizations for {selected_model_name} - Predict Pass or Fail for the Current Academic Year")

        tab1, tab2 = st.tabs(["About the Model", "About the Data"])
        with tab1:
            st.write("### Feature Importance")
            st.markdown("""
            This is a hypertuned stacked model. Since it's not possible to get the feature importances of a stacked model, the feature importances of the base learners are shown. The base learners were adaboost, logistic regression, and random forest. Here are their respective feature importances.
            """)
            col1, col2, col3 = st.columns(3)
            with col1: 
                displayimage(os.path.join(current_script_dir, '..', '..', 'Student performance - Turkey', 'Figures', 'adaboost feature importance.png'))
            with col2: 
                displayimage(os.path.join(current_script_dir, '..', '..', 'Student performance - Turkey', 'Figures', 'rf feature importance.png'))
            with col3:
                displayimage(os.path.join(current_script_dir, '..', '..', 'Student performance - Turkey', 'Figures', 'lr feature importance.png'))
                
            st.write("\n")
            col1, col2 = st.columns(2)
        
            with col1:
                st.write("### Model Metrics Bar Chart")
                st.markdown("""
                The bar chart represents various metrics used to evaluate the model's performance.
                These metrics include accuracy, precision, recall, and F1 score.
                """)
                displayimage(os.path.join(current_script_dir, '..', '..', 'Student performance - Turkey', 'Figures', 'best model metrics.png'))
            with col2:
                st.write("### Confusion Matrix")
                st.markdown("""
                The confusion matrix provides a visual representation of the model's performance,
                showing the number of correct and incorrect predictions broken down by each class.
                Top right is false negative predictions and bottom left is false positive predictions.
                """)
                displayimage(os.path.join(current_script_dir, '..', '..', 'Student performance - Turkey', 'Figures', 'confusion matrix for stacked model.png'))
        with tab2:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write("### Target Distribution")
                st.markdown("""
                This bar chart shows the distribution of the target variable within the dataset.
                """)
                displayimage(os.path.join(current_script_dir, '..', '..', 'Student performance - Turkey', 'Figures', 'target distribution.png'))
            with col2:
                st.write("### Gender Distribution")
                st.markdown("""
                Below is the gender distribution of the dataset.
                """)
            
                displayimage(os.path.join(current_script_dir, '..', '..', 'Student performance - Turkey', 'Figures', 'Gender distribution.png'))
                
            with col3:
                st.write("### Relationship Status Distribution")
                st.markdown("""
                Below is the relationship status distribution of the dataset.
                """)
                
                displayimage(os.path.join(current_script_dir, '..', '..', 'Student performance - Turkey', 'Figures', 'Partner distribution.png'))
            with st.expander("### See overall correlation matrix"):
                # st.write("### Overall Correlation Matrix")
                st.markdown("""
                The correlation matrix illustrates the relationship between all features.
                A high positive or negative value indicates a strong relationship.
                """)
                displayimage(os.path.join(current_script_dir, '..', '..', 'Student performance - Turkey', 'Figures', 'Correlation Matrix.png'))

            st.write("### Correlation Matrix by Feature Type")
            st.markdown("""
            These correlation matrices show relationships between academic and socio-economic features.
            """)
        
            col1, col2 = st.columns(2)
        
            with col1:
                displayimage(os.path.join(current_script_dir, '..', '..', 'Student performance - Turkey', 'Figures', 'socioeconomic features.png'))
            with col2:
                displayimage(os.path.join(current_script_dir, '..', '..', 'Student performance - Turkey', 'Figures', 'academic features corr matrix.png'))
            

            st.write("### Correlation Comparison")
            st.markdown("""
            These correlation graphs show the features that have high correlation with the target.
            The first graph shows top 10 features that have the highest correlation with target. The next two graphs show the top academic and socio economic features that have the highest correlation with target.
            """)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                displayimage(os.path.join(current_script_dir, '..', '..', 'Student performance - Turkey', 'Figures', 'Top 10 features with highest corr with target.png'))
            with col2:
                displayimage(os.path.join(current_script_dir, '..', '..', 'Student performance - Turkey', 'Figures', 'top 10 academic featers corr graph .png'))
            with col3:
                displayimage(os.path.join(current_script_dir, '..', '..', 'Student performance - Turkey', 'Figures', 'top 10 socio economic features with corr to target.png'))
            
            st.write("### Comparative Bar Chart of All Models")
            st.markdown("""
            This bar chart compares the performance metrics across all 14 trained models trained on this dataset. Hyperparameter tuning was carried out using GridSearchCV for optimal model performance.  
            Hypertuned stacked classifer performed best with 95% accuracy. 
            
            """)
            displayimage(os.path.join(current_script_dir, '..', '..', 'Student performance - Turkey', 'Figures', 'model comparison.PNG'))
    
else:
    st.info("Please make a prediction first.")
