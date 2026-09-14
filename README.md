<div align="center">

# 🚦 KSI — Toronto Traffic Fatality Prediction

### Machine-learning web application for analyzing Toronto traffic collisions and estimating the likelihood of fatal outcomes

Built as a full-stack data science project combining **machine learning, data preprocessing, backend services, and an interactive user interface**.

<br>

![Python](https://img.shields.io/badge/Python-ML%20%26%20Backend-3776AB?style=for-the-badge&logo=python&logoColor=white)
![JavaScript](https://img.shields.io/badge/JavaScript-Frontend-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Collision%20Risk-7B61FF?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Portfolio%20Project-success?style=for-the-badge)

<br>

[Overview](#-overview) •
[Problem](#-the-problem) •
[Features](#-features) •
[How It Works](#-how-it-works) •
[Tech Stack](#-tech-stack) •
[Installation](#-installation) •
[Challenges](#-technical-challenges) •
[Future Improvements](#-future-improvements)

</div>

---

## 🌟 Overview

**KSI — Toronto Fatality Prediction** is a machine-learning project designed to analyze traffic-collision data and estimate whether a collision is likely to result in a fatal outcome.

The project combines a predictive model with a web-based interface so users can interact with the system instead of working directly with notebooks or raw scripts.

It demonstrates an end-to-end workflow involving:

- data cleaning and preprocessing
- feature preparation
- machine-learning model development
- backend integration
- frontend development
- model deployment
- real-world data analysis

---

## 🎯 The Problem

Traffic collisions are influenced by many factors, including road conditions, environment, vehicle involvement, location, and collision characteristics.

The goal of this project is to explore whether historical collision data can be used to identify patterns associated with **fatal outcomes**.

Rather than only asking:

> “What happened in previous collisions?”

this project explores:

> “Can historical collision patterns help estimate the severity of a new collision scenario?”

This makes the project both a **machine-learning classification problem** and a practical example of data-driven decision support.

---

## 💡 Why I Built This Project

I wanted to create a machine-learning project that solved a real-world classification problem using public transportation data.

The project allowed me to practice the complete ML development lifecycle:

- understanding a real dataset
- cleaning inconsistent data
- preparing features for machine learning
- training and testing predictive models
- handling class imbalance and model evaluation
- connecting predictions to a backend
- building a frontend users can interact with
- deploying a complete application

The goal was to go beyond a notebook and build a usable **end-to-end machine-learning application**.

---

## ✨ Features

| Feature | Description |
|---|---|
| 📊 **Collision Data Analysis** | Uses historical Toronto traffic collision data |
| 🤖 **Machine-Learning Prediction** | Estimates the likelihood of a fatal collision outcome |
| 🧹 **Data Preprocessing** | Cleans and prepares raw collision data for modeling |
| 🧠 **Predictive Modeling** | Applies classification techniques to collision features |
| 🔌 **Backend Integration** | Connects the trained ML workflow to the application |
| 🖥️ **Interactive UI** | Allows users to enter collision-related information |
| 📈 **Prediction Results** | Presents model output in an understandable format |
| ☁️ **Deployable Application** | Structured for web deployment |

---

## 🔄 How It Works

```text
┌──────────────────────────┐
│ Historical KSI Dataset   │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ Data Cleaning            │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ Feature Engineering      │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ Model Training           │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ Model Evaluation         │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ Backend Prediction API   │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ Web Interface            │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ Fatality Risk Prediction │
└──────────────────────────┘
```

### Application Flow

1. Collision-related information is provided through the user interface.
2. The frontend sends the input to the backend.
3. The backend applies the same preprocessing logic used during model training.
4. The trained machine-learning model generates a prediction.
5. The prediction is returned to the frontend.
6. The result is displayed to the user.

---

## 🧰 Tech Stack

### Machine Learning / Data

- **Python**
- data preprocessing
- feature engineering
- supervised machine learning
- classification workflow
- model evaluation

### Frontend

- **JavaScript**
- **HTML**
- **CSS**

### Backend

- **Python**
- API-based prediction workflow

### Deployment

- Web deployment structure included in the project

> Add the exact libraries used in the project here, such as `pandas`, `NumPy`, `scikit-learn`, `Flask`, `FastAPI`, `React`, or others.

---

## 📁 Project Structure

```text
KSI--toronto-fatality-prediction/
│
├── Model/
│   └── data processing, model training, and ML logic
│
├── UI/
│   └── frontend and backend application
│
├── requirements.txt
├── .gitattributes
└── README.md
```

A future cleanup could standardize the repository structure to:

```text
KSI--toronto-fatality-prediction/
│
├── model/
├── backend/
├── frontend/
├── assets/
│   └── screenshots/
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## 🗂️ Dataset

This project uses Toronto traffic-collision data related to **KSI — Killed or Seriously Injured collisions**.

Because large datasets may exceed GitHub file-size limits, the dataset is not stored directly in the repository.

After downloading the dataset, place the CSV file in the location expected by the model code.

### Recommended README improvement

Instead of exposing a long raw URL, format the dataset link like this:

```markdown
[Download the dataset](YOUR_DATASET_LINK)
```

You can also add:

- dataset source
- number of rows
- number of features
- target variable
- date range
- city/open-data source

That makes the project much easier for recruiters and developers to understand.

---

## 🧠 Machine-Learning Workflow

### 1. Data Collection

The project begins with historical Toronto collision records.

### 2. Data Cleaning

Raw datasets often contain:

- missing values
- inconsistent categories
- irrelevant columns
- mixed data types
- duplicated information

These issues must be handled before model training.

### 3. Feature Preparation

Relevant collision attributes are transformed into a format suitable for machine learning.

This can include:

- categorical encoding
- numerical conversion
- missing-value handling
- feature selection
- label preparation

### 4. Model Training

A classification model learns patterns that distinguish fatal from non-fatal collision outcomes.

### 5. Evaluation

The trained model should be evaluated on data it has not seen during training.

Useful metrics include:

- accuracy
- precision
- recall
- F1-score
- ROC-AUC
- confusion matrix

### 6. Deployment

The final model is connected to the application so predictions can be generated from user input.

---

## 📊 Model Performance

Add your real model results here.

Example format:

| Metric | Result |
|---|---:|
| Accuracy | Add result |
| Precision | Add result |
| Recall | Add result |
| F1-score | Add result |
| ROC-AUC | Add result |

> Do not publish estimated or invented performance numbers. Use the exact values produced by your trained model.

For a safety-related classification task, **recall, precision, and confusion-matrix results** are often more informative than accuracy alone.

---

## 🧠 Technical Challenges

### 1. Real-World Data Quality

Public datasets often contain missing values, inconsistent formats, and categorical fields that cannot be used directly by a machine-learning model.

Preparing the dataset reliably is a major part of the project.

### 2. Class Imbalance

Fatal collisions may represent a smaller portion of the overall dataset.

This can make accuracy misleading because a model may appear accurate while performing poorly on the class that matters most.

### 3. Feature Encoding

Machine-learning models require numerical representations, so categorical collision information must be converted carefully without losing useful information.

### 4. Training vs. Prediction Consistency

The backend must apply the **same preprocessing steps** to new user input that were applied during model training.

Any mismatch can cause incorrect predictions or application errors.

### 5. Full-Stack Integration

The project also required connecting the trained ML model to a backend and then connecting that backend to the web interface.

This turned the project from a data-science experiment into a full application.

### 6. Deployment

Deploying a machine-learning model introduces additional issues such as:

- dependency management
- model-file paths
- environment configuration
- backend URLs
- frontend/backend communication

---

## 📚 What I Learned

This project strengthened my experience with:

- Python
- JavaScript
- data cleaning
- exploratory data analysis
- feature engineering
- supervised machine learning
- classification models
- model evaluation
- API integration
- frontend/backend communication
- debugging deployed applications
- dependency management
- deploying an ML-powered web application

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/Murhej/KSI--toronto-fatality-prediction.git
cd KSI--toronto-fatality-prediction
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

**Windows**

```bash
venv\Scripts\activate
```

**macOS / Linux**

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the dataset

Download the required collision dataset and place it in the location expected by the model code.

### 5. Run the application

Run the backend using the entry point defined inside the `UI/` directory, then start the frontend if it runs separately.

> Replace this section with the exact commands used by your project so another developer can run it without guessing.

---

## 📸 Screenshots

Add screenshots of the deployed application here.

Recommended screenshots:

1. **Home / Input Screen**
2. **Collision Information Form**
3. **Prediction Result**
4. **Responsive / Mobile View**

Example:

```markdown
![Application Home](assets/screenshots/home.png)
![Prediction Result](assets/screenshots/prediction.png)
```

Screenshots are one of the fastest ways to make a GitHub portfolio project feel complete and professional.

---

## 🔮 Future Improvements

Potential improvements include:

- compare additional ML algorithms
- optimize hyperparameters
- improve class-imbalance handling
- add probability/confidence output
- add explainable-AI features
- show which variables influenced a prediction
- improve data visualizations
- add geographic collision mapping
- automate preprocessing
- create automated tests
- improve mobile responsiveness
- add CI/CD deployment
- expose a documented prediction API

---

## ⚠️ Limitations

This model is based on historical data and statistical patterns.

A prediction should **not** be interpreted as a guaranteed outcome or as a substitute for professional transportation, engineering, emergency-response, or public-safety analysis.

Model performance may be affected by:

- incomplete data
- changing traffic patterns
- class imbalance
- data collection methods
- unseen collision scenarios
- model assumptions

---

## 🛡️ Disclaimer

> **This project is intended for educational, research, and portfolio purposes.**

The prediction system should not be used as the sole basis for emergency response, road-safety policy, legal decisions, or other high-stakes decisions.

---

## 👨‍💻 Author

### Murhej

Machine Learning / AI / Software Development Portfolio Project

GitHub: [@Murhej](https://github.com/Murhej)

---

<div align="center">

### ⭐ If you found this project interesting, consider starring the repository.

Built as an end-to-end exploration of **Machine Learning + Data + Full-Stack Development**.

</div>
