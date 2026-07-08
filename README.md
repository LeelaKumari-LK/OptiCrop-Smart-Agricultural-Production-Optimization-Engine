📌 Overview

OptiCrop is an intelligent crop recommendation system that uses machine learning to help farmers identify the most suitable crop to grow based on their current soil and environmental conditions. By analyzing key agricultural parameters — Nitrogen (N), Phosphorous (P), Potassium (K), temperature, humidity, pH, and rainfall — OptiCrop provides data-driven recommendations to maximize crop yield and improve resource efficiency.

The system is deployed as a Flask web application, making it accessible directly from any browser without requiring any technical knowledge from the user.


🎯 Objectives


Recommend the most suitable crop based on soil and climate parameters
Analyze relationships between environmental factors and crop suitability
Provide a simple, user-friendly interface for farmers
Support agricultural researchers and policymakers with data-driven insights
Demonstrate an end-to-end machine learning pipeline from data to deployment



🌐 Web Application Pages

PageURLDescription🏠 Home/Project introduction and navigationℹ️ About/aboutDetailed project description and objectives🌱 FindYourCrop/findyourcropInput form for crop prediction🔮 Predict/predictBackend prediction endpoint (POST)


🛠️ Technologies Used

CategoryTechnologyLanguagePython 3.10Web FrameworkFlask 3.1.3Data AnalysisPandas 2.3.3, NumPy 2.2.6VisualizationMatplotlib 3.10.9, Seaborn 0.13.2Machine LearningScikit-learn 1.7.2Model SerializationPickleFrontendHTML5, CSS3IDEPyCharmEnvironmentAnaconda (conda env: opticrop)


📊 Dataset


Source: Kaggle — Smart Agricultural Production Optimizing Engine
File: Crop_recommendation.csv
Records: 2200 samples
Crops: 22 different crop types
Features: 7 numeric input features + 1 target label


FeatureDescriptionTypeNNitrogen content in soilintPPhosphorous content in soilintKPotassium content in soilinttemperatureTemperature in °CfloathumidityRelative humidity in %floatphpH value of soilfloatrainfallRainfall in mmfloatlabelRecommended crop (target)string


🔬 Project Workflow

📥 Data Collection
       ↓
📊 Exploratory Data Analysis (Univariate, Bivariate, Multivariate)
       ↓
🧹 Data Preprocessing (Null check, Outlier handling, Seasonal analysis)
       ↓
✂️ Train/Test Split (80% train, 20% test)
       ↓
🔵 K-Means Clustering (Elbow Method → 4 clusters)
       ↓
🤖 Logistic Regression Model Training
       ↓
📈 Model Evaluation (96% Accuracy)
       ↓
💾 Model Serialization (model.pkl, scaler.pkl)
       ↓
🌐 Flask Web Application Deployment


📈 Model Performance

MetricScoreOverall Accuracy :96%  
Macro Avg Precision: 0.97  
Macro Avg Recall:0.97
Macro Avg F1-Score:0.97
Test Samples:440

Crops with Perfect F1-Score (1.00)

apple banana chickpea coconut coffee grapes mango mungbean muskmelon orange pomegranate watermelon


📁 Project Structure

OptiCrop/
│
├── app.py                      
├── eda.py                        
├── Crop_recommendation.csv         
├── model.pkl                       
├── scaler.pkl                      
│
├── templates/
│   ├── home.html                   
│   ├── about.html                  
│   └── findyourcrop.html           
│
├── static/
│   └── images/
│       ├── home_bg.jpg           
│       ├── about_bg.jpg          
│       └── crop_bg.jpg            
│
└── docs/
    ├── Document1_Code_Layout_Readability_Reusability.docx
    ├── Document2_Coding_and_Solutioning.docx
    └── Document3_Executable_Files.docx


🌱 Sample Prediction

InputValue
Nitrogen90
Phosphorous42
Potassium43
Temperature20.8°C
Humidity82.0%
pH6.5
Rainfall202.9mm
Predicted Crop🌾 Rice


🌍 Seasonal Crop Insights

SeasonConditionExample Crops☀️ SummerTemperature > 30°C, Humidity > 50%Pigeonpeas, Mango, Papaya, Grapes❄️ WinterTemperature < 20°C, Humidity > 30%Maize, Lentil, Pomegranate, Orange🌧️ RainyRainfall > 200mm, Humidity > 50%Rice, Coconut, Papaya


👩‍💻 Author

Leela


Lakshmi


madhu salini


Dataset: Kaggle — Chitrakumari25
Libraries: NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn, Flask
Tools: Anaconda, PyCharm, Python 3.10
