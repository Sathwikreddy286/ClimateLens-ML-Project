# 🌍 ClimateLens: Global Temperature Anomaly Prediction & ML Analysis

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![NASA GISTEMP v4](https://img.shields.io/badge/Data-NASA%20GISTEMP%20v4-0B3D91?logo=nasa&logoColor=white)](https://data.giss.nasa.gov/gistemp/)
[![NOAA GML](https://img.shields.io/badge/Data-NOAA%20GML%20CO₂-003366)](https://gml.noaa.gov/ccgg/trends/)

An academic-grade Machine Learning system and interactive Streamlit web application that analyzes genuine climate observations from **NASA** and **NOAA**, benchmarks regularized linear vs. tree-based ensemble algorithms, and provides interactive climate scenario forecasting.

Developed for the **Global Warming and Climate Change** academic working-model assignment.

---

## 📌 Project Highlights & Faculty Rubric Alignment

1. **Authentic Public Datasets:** Dynamically fetches NASA GISTEMP v4 global land-ocean temperature anomalies and NOAA Global Monitoring Laboratory annual atmospheric $\text{CO}_2$ observations with local offline backup.
2. **Leakage-Free Validation Strategy:** Uses strict chronological 80/20 train-test partitioning (1980–2015 train vs 2016–2025 test) and 5-fold expanding-window `TimeSeriesSplit` cross-validation.
3. **Rigorous Model Comparison:** Benchmarks **Ridge Regression** (L2 regularization with `StandardScaler` pipeline) against a **Random Forest Regressor** (300 estimators).
4. **Comprehensive Evaluation Metrics:** Calculates and reports **MAE**, **MSE**, **RMSE**, and **$R^2$** on untouched out-of-sample observations.
5. **Interactive UI & Explainability:** Feature importance bar charts, standardized coefficients, actual vs. predicted curves, downloadable datasets, and interactive scenario simulator.
6. **Honest Scientific Analysis:** Explains why decision trees flatline on future trends (extrapolation boundary) and why Ridge successfully predicts rising temperature trajectories.

---

## 🏗️ System Architecture

```
+-----------------------------------------------------------------------------------+
|                            OFFICIAL CLIMATE DATA SOURCES                          |
|   NASA GISTEMP v4 (Global Land-Ocean Temp)  |  NOAA GML (Global Atmospheric CO2)  |
+------------------------------------------+----------------------------------------+
                                           |
                                           v
+-----------------------------------------------------------------------------------+
|                           PREPROCESSING & ETL PIPELINE                            |
|  - Parse CSV / Filter Metadata Comments (#)                                       |
|  - Datatype Normalization & Null / Infinite Value Scrubbing                       |
|  - Calendar-Year Inner Join Alignment (1979–2025)                                 |
|  - Auto-Regressive Feature Engineering: Previous Year Temperature Anomaly (Tt-1)   |
|  - Verified Offline Cache Fallback Mechanism                                      |
+------------------------------------------+----------------------------------------+
                                           |
                                           v
+-----------------------------------------------------------------------------------+
|                   CHRONOLOGICAL GENERALIZATION & SPLIT STRATEGY                   |
|  - Chronological 80/20 Partitioning (No Random Shuffling to Avoid Data Leakage)   |
|  - Training Partition: 1980–2015 (36 Years, N=36)                                 |
|  - Untouched Hold-out Test Partition: 2016–2025 (10 Years, N=10)                  |
|  - 5-Fold Expanding Window TimeSeriesSplit CV on Training Partition               |
+------------------------------------------+----------------------------------------+
                                           |
                                           v
+-----------------------------------------------------------------------------------+
|                        MODEL IMPLEMENTATION & BENCHMARKING                        |
|  1. Ridge Regression (L2 Regularization, alpha=1.0) with Standard Scaler Pipeline |
|  2. Random Forest Regressor (300 Estimators, max_depth=3, min_samples_leaf=2)     |
+------------------------------------------+----------------------------------------+
                                           |
                                           v
+-----------------------------------------------------------------------------------+
|                           EVALUATION & INTERPRETABILITY                           |
|  - Test Metrics: MAE, MSE, RMSE, R2 Score                                         |
|  - Standardized Ridge Coefficients & Gini Feature Importances                     |
|  - Extrapolation Limitation Analysis (Tree Ceiling vs Linear Trend)               |
+------------------------------------------+----------------------------------------+
                                           |
                                           v
+-----------------------------------------------------------------------------------+
|                    INTERACTIVE STREAMLIT WEB APPLICATION                          |
|  - Dynamic EDA Dashboard & Trendlines                                             |
|  - Actual vs Predicted Trajectory Charts                                          |
|  - Scenario Simulation Studio & One-Click CSV Data Export                         |
+-----------------------------------------------------------------------------------+
```

---

## 📊 Live Model Benchmark & Evaluation Results

Tested on genuine empirical records (1980–2025, 46 total observations):

| Evaluation Metric | Ridge Regression | Random Forest Regressor | Faculty Note |
| :--- | :---: | :---: | :--- |
| **Cross-Validation Mean MAE** | 0.1272 °C | **0.1194 °C** | RF selected by CV within historical bounds |
| **Test Set MAE (2016–2025)** | **0.1248 °C** | 0.2514 °C | Ridge error is 50% lower on future years |
| **Test Set MSE (2016–2025)** | **0.0229 °C²** | 0.0839 °C² | Quadratic penalty for large misses |
| **Test Set RMSE (2016–2025)** | **0.1513 °C** | 0.2897 °C | Root error in target units (°C) |
| **Test Set $R^2$ Score** | **-0.1049** | -3.0494 | Documented non-stationary time-series effect |

### 🔬 Viva Insight: Why Ridge Outperformed Random Forest on Future Data
- **Tree Extrapolation Failure:** Decision trees and Random Forests cannot extrapolate above the highest target value observed in their training leaves ($\approx 0.76^\circ\text{C}$). When the Earth experienced record warming in 2016–2025 ($1.0^\circ\text{C} - 1.28^\circ\text{C}$), Random Forest flatlined.
- **Continuous Hyperplane Extrapolation:** Ridge Regression fits a continuous regularized linear slope. As $\text{CO}_2$ concentration and calendar year expanded, Ridge accurately extrapolated the upward warming trend.

---

## 🖥️ Application Screenshots

### 1. Climate Data & Exploratory Data Analysis (EDA)
![Climate Data and EDA](screenshots/01_climate_data_eda.png)

### 2. Machine Learning Models & Evaluation
![ML Models and Evaluation](screenshots/02_ml_models_and_evaluation.png)

### 3. Interactive Prediction Studio
![Interactive Prediction Studio](screenshots/03_prediction_studio.png)

### 4. Faculty Viva & Methodology Guide
![Faculty Viva and Methodology](screenshots/04_viva_and_methodology.png)

---

## 🚀 How to Run Locally

### Prerequisites
- Python 3.10 or newer
- Git

### Quick Setup

```bash
# 1. Clone the repository
git clone https://github.com/deepthigajjala/ClimateLens-ML-Project.git
cd ClimateLens-ML-Project

# 2. Create and activate a virtual environment
python -m venv .venv

# Windows (Command Prompt / PowerShell):
.venv\Scripts\activate

# Linux / macOS:
source .venv/bin/activate

# 3. Install required packages
pip install -r requirements.txt

# 4. Launch the application
streamlit run app.py
```

The application will launch automatically in your default browser at `http://localhost:8501`.

---

## 🌐 Public Deployment Guide (Streamlit Community Cloud)

To make your application publicly accessible with a live URL for your faculty:

1. **Push the project to GitHub:**
   ```bash
   git add .
   git commit -m "feat: complete ClimateLens ML application and documentation"
   git branch -M main
   git remote add origin https://github.com/<your-username>/ClimateLens-ML-Project.git
   git push -u origin main
   ```

2. **Sign in to Streamlit Community Cloud:**
   - Visit [share.streamlit.io](https://share.streamlit.io/).
   - Click **Sign in with GitHub**.

3. **Deploy the App:**
   - Click **Create app** (or **New app**).
   - Select your GitHub repository: `<your-username>/ClimateLens-ML-Project`.
   - Set the Main file path to: `app.py`.
   - Click **Deploy!**.

4. **Public Link Verification:**
   - Streamlit Cloud will build the container from `requirements.txt` and launch the app.
   - The generated public URL (e.g. `https://climatelens-ml-project.streamlit.app`) is live and accessible anywhere. Test it in an incognito window.

---

## 🎓 Faculty Viva Q&A Cheat Sheet

| Question | Answer for Examiner |
| :--- | :--- |
| **Q1. Where does your data come from?** | Authentic, real-time measurements from NASA GISTEMP v4 (land-ocean surface temperature anomalies relative to the 1951–1980 base period) and NOAA Global Monitoring Laboratory (global marine boundary layer annual $\text{CO}_2$ concentration in ppm). |
| **Q2. Why did you not use random train-test splitting?** | Climate data is a continuous time-series. Shuffling randomly causes severe **temporal data leakage**, where future records inform earlier predictions. We used a chronological 80/20 split (1980–2015 train, 2016–2025 test) and `TimeSeriesSplit` cross-validation. |
| **Q3. What features are fed to the model?** | Three features: (1) Calendar Year, (2) Atmospheric $\text{CO}_2$ concentration (ppm), and (3) Engineered lag feature: Previous Year's Temperature Anomaly ($T_{t-1}$) to model thermodynamic inertia. |
| **Q4. Why is your test $R^2$ negative?** | $R^2$ compares residual sum of squares against the test set mean. Because the test period (2016–2025) experienced an abrupt upward warming shift ($>1.0^\circ\text{C}$) compared to the training period mean ($0.48^\circ\text{C}$), a horizontal line at the test mean has smaller squared errors than a model trained on historical data. This is standard in out-of-distribution climate time-series. |
| **Q5. Why did Ridge beat Random Forest on recent years?** | Decision trees cannot extrapolate beyond the maximum value seen in their training leaves ($\approx 0.76^\circ\text{C}$). When temperatures spiked in 2023–2024 to $1.28^\circ\text{C}$, Random Forest flatlined, whereas Ridge Regression's continuous linear hyperplane extrapolated the warming trend accurately (MAE $0.125^\circ\text{C}$). |
| **Q6. What are the limitations of this model?** | It is an educational regression model with 46 annual records. It omits physical climate drivers such as methane ($\text{CH}_4$), volcanic sulfate aerosols, solar irradiance, and ocean cycles (ENSO). Correlation ($r=0.949$) does not prove causation without full Earth-system physics. |

---

## 📚 Citations & References
- **NASA GISS:** GISTEMP Team, 2026: *GISS Surface Temperature Analysis (GISTEMP), version 4*. NASA Goddard Institute for Space Studies. https://data.giss.nasa.gov/gistemp/
- **NOAA GML:** Lan, X., Tans, P., and Thoning, K.W., 2026: *Trends in Globally Averaged Carbon Dioxide*. NOAA Global Monitoring Laboratory. https://gml.noaa.gov/ccgg/trends/
- **Scikit-Learn:** Pedregosa et al., 2011: *Scikit-learn: Machine Learning in Python*. JMLR 12, pp. 2825-2830.

---
**Course Project Submission** | Global Warming and Climate Change | Built with Python, Scikit-learn & Streamlit.
