# Reddit Comment Classification - Machine Learning Experiments

## 📋 Project Overview

This repository contains a comprehensive machine learning pipeline for classifying Reddit comments into sentiment categories. The project systematically explores text preprocessing, feature engineering, model selection, and hyperparameter tuning to build an effective sentiment classifier.

## 📊 Dataset Information

**File**: `reddit_preprocessing.csv`

- **Size**: 36,662 comments after preprocessing
- **Features**:
  - `clean_comment`: Preprocessed text content
  - `category`: Sentiment labels mapped to [-1, 0, 1]
- **Classes**: Negative (-1), Neutral (0), Positive (1)

## 🧪 Experiment Series

### 📊 1. Preprocessing & EDA
**Objective**: Data cleaning, exploration, and preparation for modeling.

**Key Activities**:
- Text cleaning (lowercasing, punctuation removal, etc.)
- Exploratory data analysis
- Class distribution analysis
- Data visualization

### 🎯 2. Baseline Model
**Objective**: Establish baseline performance metrics.

**Models Tested**:
- Simple classifiers with basic features
- Performance benchmarks for comparison

### 🔬 3. BoW vs TF-IDF Comparison
**Objective**: Compare text vectorization techniques.

**Tested Configurations**:
- **Vectorizers**: Bag-of-Words vs TF-IDF
- **N-gram Ranges**: (1,1), (1,2), (1,3)
- **Max Features**: 5,000
- **Model**: Random Forest

**Key Finding**: TF-IDF with trigrams performed best.

### 📈 4. Feature Size Optimization
**Objective**: Determine optimal feature count for TF-IDF.

**Tested**: 1,000 to 10,000 features (1K increments)

**Result**: 10,000 features provided optimal performance.

### ⚖️ 5. Handling Class Imbalance
**Objective**: Address dataset imbalance.

**Techniques**:
1. Class weighting
2. SMOTE
3. ADASYN
4. Random Under-sampling
5. SMOTEENN

**Best Method**: Class weighting.

### 🚀 6. XGBoost with Hyperparameter Tuning
**Objective**: Optimize XGBoost performance.

**Tuned Parameters**:
- n_estimators: 50-300
- learning_rate: 1e-4 to 1e-1
- max_depth: 3-10

**Best Accuracy**: 77.23%

### ⚡ 7. LightGBM Detailed HPT
**Objective**: Comprehensive LightGBM optimization.

**Features**:
- Detailed hyperparameter search
- Early stopping implementation
- Feature importance analysis

### 🎲 8. Stacking Ensemble
**Objective**: Combine best models using stacking.

**Approach**:
- Meta-learner training
- Multiple base models
- Performance improvement through ensemble

## 📊 Results Summary

| Experiment | Description | Best Model/Technique | Key Metric |
|------------|-------------|---------------------|------------|
| 1 | Preprocessing & EDA | Data preparation | N/A |
| 2 | Baseline | Simple classifiers | Baseline accuracy |
| 3 | Vectorization | TF-IDF with (1,3) n-grams | ~74% accuracy |
| 4 | Feature Size | 10,000 features | Optimal performance |
| 5 | Imbalance Handling | Class Weighting | Best F1 scores |
| 6 | XGBoost Tuning | Optimized XGBoost | 77.23% accuracy |
| 7 | LightGBM Tuning | Optimized LightGBM | Comparable to XGBoost |
| 8 | Stacking | Ensemble model | Best overall performance |

## 🛠️ Technical Stack

### Core Libraries
- **MLOps**: MLflow for experiment tracking
- **Data Processing**: Pandas, NumPy
- **Text Processing**: Scikit-learn, NLTK
- **Machine Learning**: Scikit-learn, XGBoost, LightGBM
- **Hyperparameter Tuning**: Optuna
- **Visualization**: Matplotlib, Seaborn

### MLflow Integration
- **Tracking Server**: EC2 instance
- **Artifact Storage**: Amazon S3
- **Experiments**: 8 separate experiments tracked
- **Artifacts**: Models, metrics, parameters, confusion matrices

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.10+
- AWS credentials (for MLflow S3 storage)
- 8GB+ RAM recommended
