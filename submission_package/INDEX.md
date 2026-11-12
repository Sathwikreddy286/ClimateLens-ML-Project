# ClimateLens — Final College Assignment Submission Package

**Course:** Global Warming and Climate Change  
**Project Title:** Global Temperature Anomaly Prediction and Climate Change Analysis Using Machine Learning  
**Student Name:** [Your Name]  
**Register Number:** [Your Register Number]  
**Department:** Department of Computer Science & Engineering  
**Academic Semester:** 2026 Academic Year  

---

## 📁 Package Contents

| File Name | Description | Format |
| :--- | :--- | :---: |
| **Technical_Report.md** | Complete 3 A4-page individual technical report covering problem definition, architecture diagram, methodology, empirical results, critical limitation analysis, and references. | Markdown Document |
| **01_climate_data_eda.png** | Full dashboard screenshot of Tab 1: Exploratory Data Analysis, KPI cards, 5-year rolling temperature anomaly chart, NOAA CO₂ curve, and correlation scatter plot. | High-Res PNG Image |
| **02_ml_models_and_evaluation.png** | Full dashboard screenshot of Tab 2: Machine Learning Models & Evaluation, comparison table, actual vs. predicted test curves, and feature importance bar charts. | High-Res PNG Image |
| **03_prediction_studio.png** | Full dashboard screenshot of Tab 3: Interactive Prediction Studio, climate scenario presets, user input form, side-by-side Ridge vs RF estimates, and extrapolation alert. | High-Res PNG Image |
| **04_viva_and_methodology.png** | Full dashboard screenshot of Tab 4: Faculty Viva & Methodology guide directly addressing all 7 faculty assignment rubrics. | High-Res PNG Image |
| **cleaned_climate_dataset.csv** | Cleaned, time-ordered, and lag-engineered dataset merged from NASA GISTEMP v4 and NOAA GML (46 observations, 1980–2025). | CSV Spreadsheet |
| **model_comparison_metrics.csv** | Summary metrics table comparing Ridge Regression and Random Forest on Cross-Validation MAE, Test MAE, Test MSE, Test RMSE, and Test R². | CSV Spreadsheet |
| **model_test_predictions.csv** | Out-of-sample test set (2016–2025) predictions comparing actual temperature anomalies against Ridge and Random Forest forecasts. | CSV Spreadsheet |

---

## 🔬 Key Scientific & Machine Learning Findings
1. **Model Selection Protocol:** Random Forest achieved a lower Mean Cross-Validation MAE (0.1194°C vs 0.1272°C for Ridge) on the historical training set (1980–2015) because it effectively fit historical non-linear fluctuations within bounded temperatures (0.14°C to 0.87°C). It was legitimately selected based on training evidence.
2. **Extrapolation Failure of Decision Trees:** Decision trees cannot extrapolate beyond the maximum value in their training leaves (~0.76°C). When 2016–2025 experienced record out-of-sample warming (up to 1.28°C in 2024), Random Forest flatlined, doubling test error (MAE 0.2514°C). Ridge Regression, by fitting a continuous regularized linear slope, accurately extrapolated the rising temperature trajectory (MAE 0.1248°C).
3. **Accurate Interpretation of Negative R²:** R² evaluates squared error against the test set's own constant mean (1.014°C). Because the test partition underwent an abrupt upward regime shift compared to the training mean (0.480°C), a horizontal line at the test mean achieves smaller squared error than the models. A negative R² confirms that these simple statistical models cannot accurately account for accelerating out-of-distribution climate variance without physical Earth-system dynamics.
