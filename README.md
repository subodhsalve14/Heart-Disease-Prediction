# Heart Disease Prediction using Machine Learning

## 📖 Overview
This project aims to predict the likelihood of a patient having heart disease based on various medical attributes. The primary goal is to build a machine learning model that can provide accurate predictions, which could serve as a helpful tool for medical professionals in diagnosing cardiovascular conditions.

## 🚀 Live Demo
You can interact with the live model here:
http://heartdiseasepredictionusingml.streamlit.app/

## ✨ Features
Data Preprocessing & Cleaning: Handles missing values and prepares the data for model training.

Exploratory Data Analysis (EDA): In-depth analysis and visualization of the dataset to uncover insights and correlations.

Model Training: Implements various classification algorithms to find the best-performing model.

Model Evaluation: Assesses model performance using metrics like Accuracy, Precision, Recall, and F1-Score.

Web Interface: A simple user interface built with Streamlit to input patient data and get a prediction.

## 📊 Dataset
The project utilizes the UCI Heart Disease Dataset. This dataset contains 76 attributes, but only 14 are commonly used for experiments.

Source: UCI Machine Learning Repository: Heart Disease Data Set

Key Attributes:

age: Age of the patient

sex: (1 = male; 0 = female)

cp: Chest pain type

trestbps: Resting blood pressure

chol: Serum cholestoral in mg/dl

fbs: Fasting blood sugar > 120 mg/dl

restecg: Resting electrocardiographic results

thalach: Maximum heart rate achieved

exang: Exercise induced angina

oldpeak: ST depression induced by exercise relative to rest

slope: The slope of the peak exercise ST segment

ca: Number of major vessels (0-3) colored by flourosopy

thal: 3 = normal; 6 = fixed defect; 7 = reversable defect

target: 1 or 0 (presence of heart disease)

## ⚙️ Installation
To get a local copy up and running, follow these simple steps.

Prerequisites
Python 3.8 or higher

pip package manager

Steps
Clone the repository:

git clone https://github.com/your-username/heart-disease-prediction.git

Navigate to the project directory:

cd heart-disease-prediction

Create a virtual environment (recommended):

python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

Install the required dependencies:

pip install -r requirements.txt

## 🚀 Usage
Run the Jupyter Notebook:
For a detailed walkthrough of the data analysis, model training, and evaluation process, open the Heart_Disease_Prediction.ipynb notebook.

jupyter notebook Heart_Disease_Prediction.ipynb

Run the web application:
To run the Streamlit app locally:

streamlit run app.py

Open your browser and navigate to the local URL provided by Streamlit.

## 🤖 Model Training & Evaluation
Several machine learning models were trained and evaluated to find the one with the best performance.

Models Used:

Logistic Regression

K-Nearest Neighbors (KNN)

Support Vector Machine (SVM)

Decision Tree Classifier

Random Forest Classifier

Evaluation Metrics:
The models were evaluated based on the following metrics, with the confusion matrix providing a detailed breakdown of performance.

Accuracy: The proportion of correctly classified instances.

Precision: The ability of the classifier not to label as positive a sample that is negative.

Recall (Sensitivity): The ability of the classifier to find all the positive samples.

F1-Score: A weighted average of Precision and Recall.

Results:
(You can add a table here summarizing the performance of each model)

Model

Accuracy

Precision

Recall

F1-Score

Logistic Regression

85%

0.86

0.85

0.85

Random Forest

88%

0.89

0.88

0.88

...

...

...

...

...

Conclusion: The Random Forest Classifier was selected as the final model due to its superior performance across all metrics.

## 🛠️ Technologies Used
Python: The core programming language.

Pandas: For data manipulation and analysis.

NumPy: For numerical operations.

Matplotlib & Seaborn: For data visualization.

Scikit-learn: For building and evaluating machine learning models.

Jupyter Notebook: For interactive development and documentation.

Streamlit: For creating the web interface.


## 📄 License
This project is distributed under the MIT License. See LICENSE for more information.

