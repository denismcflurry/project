Disclaimer: this project is only for education and not for medical diagnosis.
Disclaimer: this project is only for education and not for medical diagnosis.
Disclaimer: this project is only for education and not for medical diagnosis.
Disclaimer: this project is only for education and not for medical diagnosis.

# HeartRisk AI

Final project for the Building AI course.

## Summary

HeartRisk AI is a small educational machine learning project. It uses a heart disease dataset to predict if a fictional patient has signs connected with heart disease. This is not a real medical tool.

## Background

Heart disease is a common and serious problem. I chose this topic because I am interested in AI, data science, and medical machine learning.

The project explores one question:

* Can simple AI models find patterns in patient data and predict heart disease risk?

## How is it used?

The user runs `main.py`.

The program:

* loads `data/heart.csv`
* splits the data into training and test sets
* trains Logistic Regression and Random Forest
* compares their accuracy and error rate
* tests fictional patients
* generates random fictional patients and checks if both models agree or disagree

Random patients are only for demonstration. They are not real people.

## Data sources and AI methods

The project uses a heart disease dataset from Kaggle.

The target is binary:

* `0` = lower risk
* `1` = higher risk / heart disease signs

Libraries used:

* pandas
* NumPy
* scikit-learn

AI methods used:

* Logistic Regression
* Random Forest
* train-test split
* accuracy score
* error rate
* confusion matrix
* classification report

## Challenges

This project does not diagnose heart disease. It is only a student demo.

Limitations:

* small dataset
* fictional random patients
* possible model mistakes
* no real medical validation
* not safe for real health decisions

## What next?

The project could be improved by adding:

* graphs
* feature importance
* more models
* Streamlit web app
* better explanations
* larger medical dataset

## Acknowledgments

This project was created for the Building AI course.

Sources and tools:

* Kaggle Heart Disease dataset
* Python
* pandas
* NumPy
* scikit-learn
