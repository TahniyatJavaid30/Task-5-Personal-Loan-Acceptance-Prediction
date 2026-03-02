# Task-5-Personal-Loan-Acceptance-Prediction
Predict which customers are likely to accept a personal loan offer.

# 💼 Personal Loan Acceptance Prediction – Bank Marketing Analysis

## 📌 Objective

The objective of this project is to:

* Predict which customers are likely to **accept a personal loan offer**.
* Perform **Exploratory Data Analysis (EDA)** on customer demographics.
* Train a **classification model** (Logistic Regression / Decision Tree).
* Extract **business insights** to support marketing decisions.

This is a **Binary Classification** problem.

---

## 📊 Dataset

Dataset used: **Bank Marketing Dataset (UCI Machine Learning Repository)**

The dataset contains customer information collected during direct marketing campaigns.

Key Features:

* Age
* Job
* Marital Status
* Education
* Balance
* Housing Loan
* Contact Type
* Campaign
* Previous Outcome
* Target Variable: `y`

Target Variable:

* `y`

  * yes → Customer accepted loan
  * no → Customer rejected loan

---

## 🛠️ Technologies Used

* Python 🐍
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

---

## 📂 Project Structure

```id="l29dzq"
personal-loan-acceptance/
│
├── bank_marketing_model.ipynb
├── bank.csv
├── requirements.txt
└── README.md
```

---

# 🔎 Project Workflow

---

## 1️⃣ Data Loading

```python
import pandas as pd

df = pd.read_csv("bank.csv", sep=";")
df.head()
```

---

## 2️⃣ Basic Data Exploration

```python
df.shape
df.info()
df["y"].value_counts()
```

✔ Checked dataset size
✔ Verified data types
✔ Observed class imbalance

---

# 📊 Exploratory Data Analysis (EDA)

---

## 📍 Age Distribution

![Image](https://www.researchgate.net/publication/379076630/figure/fig2/AS%3A11431281230285587%401710867951824/The-bar-chart-of-Customer-Age-Distribution-It-can-be-seen-that-the-age-distribution-of.ppm)

![Image](https://media2.dev.to/dynamic/image/width%3D800%2Cheight%3D%2Cfit%3Dscale-down%2Cgravity%3Dauto%2Cformat%3Dauto/https%3A%2F%2Fdev-to-uploads.s3.amazonaws.com%2Fuploads%2Farticles%2F53ehh3bxrrzvflw469el.png)

![Image](https://www.researchgate.net/publication/346210645/figure/fig1/AS%3A963462343708697%401606718731992/Age-distribution-Comparison-among-age-frequency-histograms-for-ISP-bank-account-holders.png)

![Image](https://www.researchgate.net/publication/372630374/figure/fig1/AS%3A11431281190417111%401695323480397/Age-histogram-As-can-be-seen-in-Figure-1-only-a-small-proportion-of-the-banks-clients.png)

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.histplot(df["age"], kde=True)
plt.title("Age Distribution")
plt.show()
```

📌 Insight: Most customers fall between 30–50 years.

---

## 📍 Job vs Loan Acceptance

![Image](https://www.researchgate.net/publication/331586507/figure/fig1/AS%3A733968806707200%401552003208253/A-bar-chart-representing-opinion-on-whether-loans-are-provided-by-micro-finance.png)

![Image](https://miro.medium.com/v2/resize%3Afit%3A1400/1%2AFcRWtZPZSJfuqLAGtmbaeA.png)

![Image](https://www.researchgate.net/publication/381510071/figure/fig2/AS%3A11431281386415577%401745069165097/a-Bar-plot-for-categorical-features-versus-target-b-box-plot-for-numerical-features_Q320.jpg)

![Image](https://d13ot9o61jdzpp.cloudfront.net/images/two_series_actual_vs_target_chart.png)

```python
sns.countplot(x="job", hue="y", data=df)
plt.xticks(rotation=45)
plt.title("Job vs Loan Acceptance")
plt.show()
```

📌 Insight: Certain professions show higher acceptance rates.

---

## 📍 Marital Status vs Loan Acceptance

![Image](https://www.researchgate.net/publication/371575364/figure/fig5/AS%3A11431281221683026%401706839399561/Loan-approval-on-the-basis-of-gender-and-marital-status.png)

![Image](https://user-images.githubusercontent.com/32011622/60384234-f8b79c80-9a98-11e9-9bb4-7043893b5991.png)

![Image](https://miro.medium.com/v2/resize%3Afit%3A956/1%2APck7kEelP16RRJAXEeAktw.png)

![Image](https://miro.medium.com/v2/resize%3Afit%3A852/1%2AYttlmifwsfiUyeAFIbXUHg.png)

```python
sns.countplot(x="marital", hue="y", data=df)
plt.title("Marital Status vs Loan Acceptance")
plt.show()
```

📌 Insight: Married and single customers respond differently to campaigns.

---

# 🔄 Data Preprocessing

* Converted target variable (`yes/no`) to binary (1/0)
* Applied One-Hot Encoding to categorical features
* Split data into training and testing sets

```python
df["y"] = df["y"].map({"yes":1, "no":0})
df = pd.get_dummies(df, drop_first=True)
```

---

# 🤖 Model Training

We trained a **Logistic Regression** model.

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

X = df.drop("y", axis=1)
y = df["y"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
```

---

# 📈 Model Evaluation

## ✅ Accuracy

```python
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
```

---

## 📉 Confusion Matrix

```python
cm = confusion_matrix(y_test, y_pred)
print(cm)
```

Interpretation:

* True Positives → Correctly predicted loan acceptance
* True Negatives → Correctly predicted rejection
* False Positives → Incorrect acceptance prediction
* False Negatives → Missed potential customers

---

# 📊 Business Insights

Based on model results and feature patterns:

* Customers aged 30–50 show higher acceptance probability.
* Certain job categories are more responsive to marketing campaigns.
* Marital status influences financial decision-making.
* Previous successful contact increases conversion rate.

📌 Marketing teams can:

* Target high-probability customer segments.
* Optimize campaign budget allocation.
* Improve customer segmentation strategies.

---

# 🧠 Skills Demonstrated

✔ Data Exploration & Visualization
✔ Handling Categorical Data
✔ Binary Classification Modeling
✔ Logistic Regression
✔ Confusion Matrix Interpretation
✔ Business Insight Extraction

---

# 🚀 Future Improvements

* Try Decision Tree / Random Forest
* Handle class imbalance (SMOTE)
* Hyperparameter tuning
* Build marketing dashboard using Streamlit
* Deploy model as API

---

# 🏁 Conclusion

This project demonstrates a complete **end-to-end marketing analytics workflow**, including:

* Data exploration
* Feature engineering
* Classification modeling
* Business insight extraction

It provides real-world experience in **customer targeting and predictive marketing**, commonly used in banking and financial services industries.
