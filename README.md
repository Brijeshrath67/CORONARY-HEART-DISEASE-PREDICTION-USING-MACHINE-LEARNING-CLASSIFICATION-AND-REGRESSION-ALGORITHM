# CORONARY-HEART-DISEASE-PREDICTION-USING-MACHINE-LEARNING-CLASSIFICATION-AND-REGRESSION-ALGORITHM

# 🫀 Coronary Heart Disease Prediction Using Machine Learning  
### Classification & Regression Models with Exploratory Data Analysis

<p align="center">
  <img src="https://img.shields.io/badge/ML-Classification%20|%20Regression-blue" />
  <img src="https://img.shields.io/badge/Python-3.10+-yellow" />
  <img src="https://img.shields.io/badge/Jupyter-Notebook-orange" />
  <img src="https://img.shields.io/badge/Status-Completed-success" />
</p>

---

## 📌 **Project Overview**
Coronary Heart Disease (CHD) is one of the leading causes of mortality worldwide. Early prediction of heart disease can significantly improve treatment outcomes and save lives.

In this project, we develop **machine-learning models** to predict the likelihood of CHD using patient medical history and biometric features.  
Both **classification** and **regression** techniques are implemented, along with detailed **Exploratory Data Analysis (EDA)**.

This notebook demonstrates a full ML workflow:

✅ Data cleaning  
✅ Exploratory data analysis  
✅ Encoding & feature engineering  
✅ Model training  
✅ Performance comparison  
✅ Visualization  

---

## 🧠 **Objectives**
- Build a system to **analyze heart-related medical data**
- Perform **EDA** to understand data patterns
- Train **classification models** to predict disease presence (Yes/No)
- Train **regression models** to predict CHD severity scores
- Evaluate models using accuracy, precision, recall, RMSE, etc.
- Present insights that can help in early diagnosis

---

## 📂 **Project Structure**
```
📁 Heart-Disease-Prediction
│── data/
│    └── heart.csv (or dataset used)
│── notebook/
│    └── CHD Prediction.ipynb
│── README.md
│── .gitignore
│── requirements.txt
```

---

## 📊 **Dataset Description**

The dataset includes key medical features commonly linked to heart disease, such as:

| Feature | Description |
|--------|-------------|
| Age | Patient age |
| Gender | Male/Female |
| Cholesterol | Cholesterol level (mg/dL) |
| Blood Pressure | Resting blood pressure |
| Heart Rate | Maximum heart rate achieved |
| Fasting Blood Sugar | >120 mg/dL (1 = True) |
| Chest Pain Type | 4 categorical types |
| Exercise Angina | Yes/No |
| Oldpeak | ST depression |
| Slope | Slope of ST segment |
| Target | 1 = CHD present, 0 = No |

---

## 🔍 **Exploratory Data Analysis (EDA)**  
Your notebook performed the following EDA steps:

### ✅ 1. Basic Data Overview  
- Checked shape, datatypes  
- Handled missing values  
- Summary statistics  

### ✅ 2. Univariate Analysis  
- Distribution plots for numeric features  
- Count plots for categorical features  

### ✅ 3. Bivariate Analysis  
- Correlation heatmap  
- Pairwise relationships  
- CHD distribution across age, cholesterol, BP  

### ✅ 4. Outlier Detection  
- Boxplots for cholesterol, oldpeak, blood pressure  

### ✅ 5. Feature Engineering  
- Categorical encoding  
- Feature scaling  
- Train–test split  

---

## 🤖 **Machine Learning Models Used**

### ✅ **Classification Models**
| Model | Purpose |
|-------|---------|
| Logistic Regression | Baseline linear classifier |
| Naïve Bayes | Assumes feature independence |
| Random Forest | Handles non-linear patterns |
| Decision Tree | Simple explainable model |

### ✅ **Regression Models**
| Model | Purpose |
|-------|---------|
| Linear Regression | Baseline regressor |
| Random Forest Regressor | Predicts CHD severity score |
| Decision Tree Regressor | Simple and interpretable |

---

## ✅ **Model Evaluation Metrics**

### **Classification Metrics**
- Accuracy  
- Precision  
- Recall  
- F1-Score  
- Confusion Matrix  

### **Regression Metrics**
- Mean Absolute Error (MAE)  
- Mean Squared Error (MSE)  
- Root Mean Squared Error (RMSE)  
- R²-Score  

---

## 📈 **Results Summary**

### ✅ **Best Classification Model**
**Random Forest Classifier**  
- High accuracy  
- Best precision & recall  
- Handles non-linear interactions well  

### ✅ **Best Regression Model**
**Random Forest Regressor**  
- Lowest RMSE  
- Best performance on unseen data  

---

## 🛠️ **Technologies Used**
- Python  
- NumPy  
- Pandas  
- Matplotlib  
- Seaborn  
- Scikit-learn  
- Jupyter Notebook  

---

## ⚙️ **How to Run This Project**

### ✅ 1. Clone the repo  
```bash
git clone https://github.com/yourusername/heart-disease-prediction.git
cd heart-disease-prediction
```

### ✅ 2. Install dependencies  
```bash
pip install -r requirements.txt
```

### ✅ 3. Open the notebook  
```bash
jupyter notebook
```

---

## 📘 **License**
✅ **MIT License (simple and open)** 

---

## 🚀 **Future Improvements**
- Add deep learning models (ANN)  
- Deploy using Streamlit  
- Build a responsive web app   
- Add real-time ECG-based features  

---

## 🙌 **Acknowledgments**
This project is created for academic learning, portfolio building, and practical understanding of machine learning in healthcare.

---

👨‍💻 Author


Brijesh Rath


📧 Email: rathbrijesh2006@gmail.com


💼 GitHub: (https://github.com/Brijeshrath67)

--

## ⭐ **If you like this project, give it a star on GitHub!**
