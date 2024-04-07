# Predicting Academic Success: A Foundational Step Towards Personalized Learning

This project demonstrates the potential of applying machine learning (ML) to the educational sector, given appropriate data. By training 14 models on 4 distinct datasets, we provide this project as a proof of concept for personalized learning through the prediction of academic success.

## Overview

Utilizing four comprehensive datasets from the UCI Machine Learning repository, this study explores the prediction of students' academic success. These datasets cover a wide range of factors, including academic, social, socio-economic, demographic, and enrollment data, offering a holistic approach to understanding student performance.

### Datasets and Training

- **Data Split**: 80% training, 20% testing
- **ML Algorithms Explored**:
  - Logistic Regression
  - Naïve Bayes
  - Decision Tree
  - Extreme Gradient Boosting (XGBoost)
  - Light Gradient Boosting (LightGBM)
  - AdaBoost
  - Random Forest
  - Stochastic Gradient Descent Classifier (SGD)
  - Support Vector Classifier (SVC)
  - Neural Network
  - 2 Stacking Classifiers (with various base learners)
  - 2 Voting Classifiers (one soft, one hard)
- **Hyperparameter Tuning**: Utilized GridSearchCV for optimization

### Key Findings

- **Portuguese Higher Education Dataset**: Best performance by Logistic Regression (92% across all metrics) for graduation/dropout prediction.
- **Portuguese High School Datasets**: Superior performance by SVC for pass/failure prediction in Mathematics (91%) and Portuguese language (95%).
- **Turkish University Dataset**: A stacking classifier with an ensemble of Random Forest, AdaBoost, and Logistic Regression as base learners, and Logistic Regression as a final learner, showed best performance (83%).

## System Architecture

Our solution includes an interactive user interface built with Streamlit, enabling users to make predictions on student outcomes. Below is an architectural overview:

![how the UI works](https://github.com/samsondawit/student-success/assets/117774323/49c9f113-4d57-40e0-a541-b796675a954a)


### SQL Database Integration

An SQL database comprising 4 tables for the models ensures automatic loading of relevant data upon user selection.

### User Interaction

Users can perform inferences on individual or multiple students, choosing either to select students from the database or manually input feature values (requires dataset familiarity).

## Repository Structure

The repository is organized into several key folders:

- **Predicting Graduation Dropout 4.4k**: Contains data, notebooks, saved models (.joblib), and figures for dropout prediction.
- **Student Pass Fail POR**: Hosts model training for Mathematics and Portuguese subjects, including data, notebooks, models, and figures.
- **Student Performance - Turkey**: Includes materials for predicting academic performance in Turkey.
- **GUI Folder**: Contains the Streamlit-based GUI and database. To run locally:
  1. Install dependencies from `requirements.txt`.
  2. Execute `streamlit run Home.py`.

## Accessing the Interface

The Streamlit application is accessible at [https://academic-success.streamlit.app/](https://academic-success.streamlit.app/).
