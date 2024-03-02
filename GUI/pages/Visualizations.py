import streamlit as st
from PIL import Image
from joblib import load
import os


st.set_page_config(layout="wide")
def displayimage(image_path):
    img = Image.open(image_path)
    st.image(img)
    

if 'prediction_made' in st.session_state and st.session_state['prediction_made']:
    selected_model_name = st.session_state['selected_model_name']
    model_paths = st.session_state['model_paths']
    model_path = model_paths[selected_model_name]
    model = load(model_path)
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    st.title(f"Vizualizations for {selected_model_name}")
    if selected_model_name == 'Model 1':
        tab1, tab2 = st.tabs(["About the Model", "About the Data"])
        with tab1:
            st.write("### Feature Importance")
            st.markdown("""
            This chart illustrates the importance of each feature in the prediction made by the model.
            Features that are higher on the chart have a greater influence on the model's predictions.
            """)
            displayimage(os.path.join(current_script_dir, '..', '..', 'Predicting Gradutation-Dropout 4.4k', 'figures', 'feature importance.png'))
            st.write("\n")
            col1, col2 = st.columns(2)
        
            with col1:
                st.write("### Model Metrics Bar Chart")
                st.markdown("""
                The bar chart represents various metrics used to evaluate the model's performance.
                These metrics include accuracy, precision, recall, and F1 score among others.
                """)
                displayimage(os.path.join(current_script_dir, '..', '..', 'Predicting Gradutation-Dropout 4.4k', 'figures', 'best model metrics.png'))
            with col2:
                st.write("### Confusion Matrix")
                st.markdown("""
                The confusion matrix provides a visual representation of the model's performance,
                showing the number of correct and incorrect predictions broken down by each class.
                """)
                displayimage(os.path.join(current_script_dir, '..', '..', 'Predicting Gradutation-Dropout 4.4k', 'figures', 'best model confusion matrix.png'))
        with tab2:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write("### Target Distribution")
                st.markdown("""
                This bar chart shows the distribution of the target variable within the dataset,
                which helps in understanding the balance or imbalance of classes.
                """)
                displayimage(os.path.join(current_script_dir, '..', '..', 'Predicting Gradutation-Dropout 4.4k', 'figures', 'target distribution.png'))
            with col2:
                st.write("### Gender Distribution")
                st.markdown("""
                Below is the gender distribution of the dataset.
                """)
                displayimage(os.path.join(current_script_dir, '..', '..', 'Predicting Gradutation-Dropout 4.4k', 'figures', 'gender distribution.png'))
                
            with col3:
                st.write("### Maritial Status Distribution")
                st.markdown("""
                Below is the maritial status distribution of the dataset.
                """)
                displayimage(os.path.join(current_script_dir, '..', '..', 'Predicting Gradutation-Dropout 4.4k', 'figures', 'Maritial Status.png'))
            with st.expander("### See overall correlation matrix"):
                # st.write("### Overall Correlation Matrix")
                st.markdown("""
                The correlation matrix illustrates the relationship between all features.
                A high positive or negative value indicates a strong relationship.
                """)
                displayimage(os.path.join(current_script_dir, '..', '..', 'Predicting Gradutation-Dropout 4.4k', 'figures', 'correlation heatmap.png'))

            st.write("### Correlation Matrix by Feature Type")
            st.markdown("""
            These correlation matrices show relationships between academic, socio-economic, and demographic features.
            """)
            col1, col2 = st.columns(2)
            
            displayimage(os.path.join(current_script_dir, '..', '..', 'Predicting Gradutation-Dropout 4.4k', 'figures', 'academic data - correlation matrix.png'))
            with col1:
                displayimage(os.path.join(current_script_dir, '..', '..', 'Predicting Gradutation-Dropout 4.4k', 'figures', 'socio-economic features -correlation heatmap.png'))
            with col2:
                displayimage(os.path.join(current_script_dir, '..', '..', 'Predicting Gradutation-Dropout 4.4k', 'figures', 'demographic features - correlation matrix.png'))

            st.write("### Correlation Comparison")
            st.markdown("""
            These correlation graphs show the features that have high correlation with the target.
            """)
            col1, col2, col3 = st.columns(3)
            with col1:
                displayimage(os.path.join(current_script_dir, '..', '..', 'Predicting Gradutation-Dropout 4.4k', 'figures', 'top ten highest corr graph.png'))
            with col2:
                displayimage(os.path.join(current_script_dir, '..', '..', 'Predicting Gradutation-Dropout 4.4k', 'figures', 'top 10 highest socio economic features corr graph.png'))
            with col3:
                displayimage(os.path.join(current_script_dir, '..', '..', 'Predicting Gradutation-Dropout 4.4k', 'figures', 'top 10 highest academic features corr graph.png'))
            st.write("### Comparative Bar Chart of All Models")
            st.markdown("""
            This bar chart compares the performance metrics across all 14 trained models trained on this dataset.
            """)
            displayimage(os.path.join(current_script_dir, '..', '..', 'Predicting Gradutation-Dropout 4.4k', 'figures', 'model comparision for chosen dataset.png'))

        
    elif selected_model_name in ['Model 2']:  
        tab1, tab2 = st.tabs(["About the Model", "About the Data"])
        with tab1:
            st.write("### Feature Importance")
            st.markdown("""
            This chart illustrates the importance of each feature in the prediction made by the model.
            Features that are higher on the chart have a greater influence on the model's predictions.
            """)
            displayimage(os.path.join(current_script_dir, '..', '..', 'Student Pass Fail POR', 'figure MAT', 'feature importance for SVC (best model).png'))
            st.write("\n")
            col1, col2 = st.columns(2)
        
            with col1:
                st.write("### Model Metrics")
                st.markdown("""
                The bar chart represents various metrics used to evaluate the model's performance.
                These metrics include accuracy, precision, recall, and F1 score among others.
                """)
                displayimage(os.path.join(current_script_dir, '..', '..', 'Student Pass Fail POR', 'figure MAT', 'best model performance metrics.png'))
            with col2:
                st.write("### Confusion Matrix")
                st.markdown("""
                The confusion matrix provides a visual representation of the model's performance,
                showing the number of correct and incorrect predictions broken down by each class.
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
            These correlation matrices show relationships between academic, socio-economic, and demographic features.
            """)

            col1, col2 = st.columns(2)
            
            with col1:
                displayimage(os.path.join(current_script_dir, '..', '..', 'Student Pass Fail POR', 'figure MAT', 'academic features -  correlation matrix.png'))
            with col2:
                displayimage(os.path.join(current_script_dir, '..', '..', 'Student Pass Fail POR', 'figure MAT', 'social features - correlation map.png'))
            
            displayimage(os.path.join(current_script_dir, '..', '..', 'Student Pass Fail POR', 'figure MAT', 'school related features - correlation map.png'))

            st.write("### Correlation Comparison")
            st.markdown("""
            These correlation graphs show the features that have high correlation with the target.
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
            """)
            displayimage(os.path.join(current_script_dir, '..', '..', 'Student Pass Fail POR', 'figure MAT', 'model comparison.PNG'))

        
    elif selected_model_name == 'Model 3':  
        tab1, tab2 = st.tabs(["About the Model", "About the Data"])
        with tab1:
            st.write("### Feature Importance")
            st.markdown("""
            This chart illustrates the importance of each feature in the prediction made by the model.
            Features that are higher on the chart have a greater influence on the model's predictions.
            """)
            displayimage(os.path.join(current_script_dir, '..', '..', 'Student Pass Fail POR', 'figure POR', 'feature importance for best model.png'))
            st.write("\n")
            col1, col2 = st.columns(2)
        
            with col1:
                st.write("### Model Metrics Bar Chart")
                st.markdown("""
                The bar chart represents various metrics used to evaluate the model's performance.
                These metrics include accuracy, precision, recall, and F1 score among others.
                """)
                displayimage(os.path.join(current_script_dir, '..', '..', 'Student Pass Fail POR', 'figure POR', 'best model metrics.png'))
            with col2:
                st.write("### Confusion Matrix")
                st.markdown("""
                The confusion matrix provides a visual representation of the model's performance,
                showing the number of correct and incorrect predictions broken down by each class.
                """)
                displayimage(os.path.join(current_script_dir, '..', '..', 'Student Pass Fail POR', 'figure POR', 'confusion matrix for bet model.png'))
        with tab2:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write("### Target Distribution Bar Chart")
                st.markdown("""
                This bar chart shows the distribution of the target variable within the dataset,
                which helps in understanding the balance or imbalance of classes.
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
            These correlation matrices show relationships between academic, socio-economic, and demographic features.
            """)
            displayimage(os.path.join(current_script_dir, '..', '..', 'Student Pass Fail POR', 'figure POR', 'academic features.png'))
            col1, col2 = st.columns(2)
            
            with col1:
                displayimage(os.path.join(current_script_dir, '..', '..', 'Student Pass Fail POR', 'figure POR', 'school related features.png'))
            with col2:
                displayimage(os.path.join(current_script_dir, '..', '..', 'Student Pass Fail POR', 'figure POR', 'social features.png'))
            

            st.write("### Correlation Comparison")
            st.markdown("""
            These correlation graphs show the features that have high correlation with the target.
            """)
            displayimage(os.path.join(current_script_dir, '..', '..', 'Student Pass Fail POR', 'figure POR', 'top ten features graph.png'))
            col1, col2, col3 = st.columns(3)
            with col1:
                displayimage(os.path.join(current_script_dir, '..', '..', 'Student Pass Fail POR', 'figure POR', 'top school realted features with highest corr.png'))
            with col2:
                displayimage(os.path.join(current_script_dir, '..', '..', 'Student Pass Fail POR', 'figure POR', 'top academic features graph.png'))
            with col3:
                displayimage(os.path.join(current_script_dir, '..', '..', 'Student Pass Fail POR', 'figure POR', 'top school realted features with highest corr.png'))
            
            st.write("### Comparative Bar Chart of All Models")
            st.markdown("""
            This bar chart compares the performance metrics across all 14 trained models trained on this dataset.
            """)
            displayimage(os.path.join(current_script_dir, '..', '..', 'Student Pass Fail POR', 'figure POR', 'model comparison.PNG'))

    elif selected_model_name == 'Model 4':  
        tab1, tab2 = st.tabs(["About the Model", "About the Data"])
        with tab1:
            st.write("### Feature Importance")
            st.markdown("""
            This is a hypertuned stacked model. The base learners were adaboost, logistic regression, and random forest. Here are their respective feature importances.
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
                """)
                displayimage(os.path.join(current_script_dir, '..', '..', 'Student performance - Turkey', 'Figures', 'confusion matrix for stacked model.png'))
        with tab2:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write("### Target Distribution")
                st.markdown("""
                This bar chart shows the distribution of the target variable within the dataset,
                which helps in understanding the balance or imbalance of classes.
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
            This bar chart compares the performance metrics across all 14 trained models trained on this dataset.
            """)
            displayimage(os.path.join(current_script_dir, '..', '..', 'Student Pass Fail POR', 'figure POR', 'model comparison.PNG'))
    
else:
    st.info("Please make a prediction first.")
