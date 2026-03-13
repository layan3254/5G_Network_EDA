# 🌐 5G Network Intelligence: Predictive Analysis & Stability Forecasting

## 📑 Project Overview
This project provides a comprehensive data science framework for analyzing 5G network telemetry and predicting connection stability. The primary focus is on uncovering the complex relationships between environmental factors and network performance through advanced Exploratory Data Analysis (EDA) and the creation of a custom "Stress Score" indicator.

The Link:
https://5gnetworkeda-htkugj9cf5afrxhlhqntrt.streamlit.app/

### 🔍 1. Exploratory Data Analysis (EDA) - The Core Analysis
The analysis moved beyond raw metrics to identify the real drivers of 5G instability. Key findings included:

- The Jitter Factor: Identification of a critical negative correlation (-0.85) between Jitter and Signal Quality, proving that Jitter is the primary "silent killer" of 5G performance.

- Temporal Volatility: Visualization of hourly trends revealed that 10:00 AM is the peak period for network drops globally, while 5:00 PM remains the most stable.

- Infrastructure Resilience: A deep dive into peak stress points showed that some locations (e.g., Kolkata) maintain stable connections even at record-high Stress Scores of 14,782, highlighting local infrastructure efficiency.

- Performance Gaps: Mapping download speeds across global carriers identified significant disparities in 5G SA (Standalone) vs. NSA (Non-Standalone) performance.

### 🛠 2. Data Preprocessing & Feature Engineering
To bridge the gap between raw data and actionable insights, the following technical steps were executed:

- Feature Innovation: Development of the "Stress Score" (Congestion × Temperature × |Signal|) and the "Signal Quality Index" to quantify network pressure more accurately than individual raw metrics.

- Outlier Management: Application of Winsorization to cap extreme noise in KPIs (Latency, Signal Strength), ensuring data integrity without losing trend information.

- Categorical Hierarchy: Manual encoding of ordinal features (Congestion Levels) to preserve the logical progression of network load (Low < Medium < High).

### 🤖 3. Modeling: Validating the Analytical Insights
As a final validation of the EDA and feature engineering phase, a machine learning pipeline was implemented:

- The Verdict: A Random Forest Classifier was trained to test if the engineered "Stress Score" and "Signal Quality" could accurately forecast drops.

- Results: The model successfully outperformed linear baselines, confirming that 5G stability is a non-linear challenge driven by the combination of factors identified during the analysis phase.

- Predictive Power: The Feature Importance analysis confirmed that the custom Stress Score was indeed the most influential predictor, validating the entire analytical approach.

### 🚀 4. Interactive Dashboard (Streamlit)
The insights were deployed into an interactive Streamlit Dashboard that allows for:

- Real-time filtering of network performance by City, Carrier, and Device.

- An AI-powered simulator to test "What-if" scenarios and predict connection risk based on current environmental inputs.

<img width="865" height="573" alt="image" src="https://github.com/user-attachments/assets/1bb4b94c-7bae-42d4-98fc-a342ccd17562" />

<img width="1643" height="1135" alt="image" src="https://github.com/user-attachments/assets/62ec34d7-2f09-4504-bb86-274a7c7d7d13" />

<img width="2203" height="1286" alt="image" src="https://github.com/user-attachments/assets/4cc91eb0-fd15-45d8-b3de-4947b13581c5" />

<img width="1792" height="957" alt="image" src="https://github.com/user-attachments/assets/abd813c9-e75c-4017-8970-6dab0811acc5" />

<img width="1783" height="953" alt="image" src="https://github.com/user-attachments/assets/a53df993-3681-4d91-9fe1-44f2437d1420" />

<img width="1774" height="1135" alt="image" src="https://github.com/user-attachments/assets/beee202d-47ac-4b96-b9cb-6c534da76106" />






