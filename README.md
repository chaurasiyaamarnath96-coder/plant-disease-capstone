🌱 Plant Disease Detection and Recommendation System

An end-to-end Deep Learning application for automated tomato plant disease detection using computer vision, transfer learning, and web deployment. The system identifies 10 different tomato leaf diseases from images, provides actionable recommendations, logs predictions to a database, and offers an interactive Streamlit web interface.

📌 Project Overview

Agricultural diseases significantly impact crop productivity and food security. Early detection and diagnosis can help farmers take timely action and reduce losses.

This project leverages Deep Learning, Computer Vision, and Transfer Learning to automatically classify tomato leaf diseases from images and provide corresponding treatment recommendations.

The complete system includes:

Disease Classification using EfficientNet-B0
Automated Recommendation Engine
SQLite Database Logging
Interactive Streamlit Web Application
Prediction Analytics Dashboard
Image Upload and Real-Time Inference
🚀 Features
✅ Disease Detection

Detects 10 tomato leaf conditions:

Tomato Healthy
Bacterial Spot
Early Blight
Late Blight
Leaf Mold
Septoria Leaf Spot
Spider Mites
Target Spot
Tomato Mosaic Virus
Tomato Yellow Leaf Curl Virus
✅ Disease Recommendation Engine

Provides disease-specific:

Cause
Severity
Prevention Guidelines
Treatment Recommendations
✅ Deep Learning Model
Transfer Learning using EfficientNet-B0
Trained on PlantVillage Dataset
Multi-class Image Classification
PyTorch Implementation
✅ Database Integration

Stores prediction history automatically using SQLite:

Timestamp
Disease Name
Confidence Score
Recommendation
✅ Streamlit Web Application

Users can:

Upload leaf images
View predictions instantly
Get confidence scores
Access treatment recommendations
✅ Analytics Dashboard

Visualizes:

Disease frequency
Total predictions
Confidence distribution
Prediction history
🗂️ Project Structure
Plain Text
plant-disease-capstone/
 
├── data/
│ ├── raw/
│ └── processed/
│
├── notebooks/
│ ├── 01_EDA.ipynb
│ ├── 02_Training.ipynb
│ └── 03_Evaluation.ipynb
│
├── src/
│ ├── __init__.py
│ ├── model.py
│ ├── predict.py
│ ├── transforms.py
│ ├── disease_info.py
│ ├── database.py
│ ├── autoencoder.py
│ └── anomaly.py
│
├── models/
│ ├── best_model.pth
│ └── autoencoder.pth
│
├── database/
│ └── plant_disease.db
│
├── streamlit_app/
│ ├── app.py
│ └── pages/
│ └── dashboard.py
│
├── requirements.txt
├── README.md
└── .gitignore
🧠 Model Architecture
EfficientNet-B0

The disease detection model is based on:

Plain Text
Input Image
↓
Resize (224×224)
↓
EfficientNet-B0
↓
Fully Connected Layer
↓
10 Disease Classes
Why EfficientNet-B0?
Lightweight and efficient
Excellent transfer learning performance
Lower computational cost
High classification accuracy
📊 Dataset
PlantVillage Dataset

The model was trained on the PlantVillage Tomato Dataset containing tomato leaf images across different disease categories.

Classes:

Plain Text
Tomato___Bacterial_spot
 
Tomato___Early_blight
 
Tomato___Late_blight
 
Tomato___Leaf_Mold
 
Tomato___Septoria_leaf_spot
 
Tomato___Spider_mites Two-spotted_spider_mite
 
Tomato___Target_Spot
 
Tomato___Tomato_Yellow_Leaf_Curl_Virus
 
Tomato___Tomato_mosaic_virus
 
Tomato___healthy
📈 Model Performance
Validation Results
Metric	ScoreValidation Accuracy	99.50%
Macro F1 Score	0.99
Weighted F1 Score	1.00
Precision	99%+
Recall	99%+
Classification Report Highlights
Plain Text
Tomato Healthy
Precision: 1.00
Recall: 1.00
F1 Score: 1.00
 
Target Spot
Precision: 1.00
Recall: 1.00
F1 Score: 1.00
 
Septoria Leaf Spot
Precision: 1.00
Recall: 1.00
F1 Score: 1.00
 
Tomato Mosaic Virus
Precision: 0.95
Recall: 1.00
F1 Score: 0.97
⚙️ Technology Stack
Machine Learning
PyTorch
Torchvision
NumPy
Scikit-Learn
Data Analysis
Pandas
Matplotlib
Seaborn
Web Application
Streamlit
Database
SQLite
Image Processing
Pillow (PIL)
OpenCV
💾 Database Design
Table: predictions
SQL
CREATE TABLE predictions (
 
id INTEGER PRIMARY KEY AUTOINCREMENT,
 
timestamp TEXT,
 
disease TEXT,
 
confidence REAL,
 
recommendation TEXT
);
 
🔍 Prediction Workflow
Plain Text
User Uploads Image
↓
Image Preprocessing
↓
EfficientNet-B0
↓
Disease Prediction
↓
Confidence Score
↓
Recommendation Engine
↓
SQLite Logging
↓
Result Visualization
📸 Application Screens
Home Page

Features:

Image Upload
Disease Detection
Confidence Metrics
Recommendation System
Dashboard

Features:

Prediction History
Disease Distribution
Total Predictions
Confidence Analysis
🏃 Installation
Clone Repository
Shell
git clone https://github.com/your-username/plant-disease-capstone.git
 
cd plant-disease-capstone
Create Virtual Environment
Shell
python -m venv venv
Activate Environment

Windows:

Shell
venv\Scripts\activate
Install Dependencies
Shell
pip install -r requirements.txt
▶️ Run Streamlit App
Shell
streamlit run streamlit_app/app.py

Open:
https://plant-disease-capstone-skmiuq9ubhvc5wagsktyhr.streamlit.app/
📋 Sample Usage
Launch application.
Upload a tomato leaf image.
Click Predict Disease.
View:
Disease Name
Confidence Score
Recommendation
Prediction is automatically stored in SQLite database.
🎯 Future Enhancements
Leaf vs Non-Leaf Detection Model
Unknown Disease Identification
AutoEncoder-Based Anomaly Detection
Mobile Application Deployment
Cloud Deployment (AWS/Azure)
Real-Time Field Detection
Multi-Crop Disease Classification
Farmer Alert System
📚 Key Learnings

This project demonstrates:

Deep Learning
Computer Vision
Transfer Learning
Model Evaluation
Database Integration
MLOps Fundamentals
Streamlit Deployment
End-to-End AI Application Development
👨‍💻 Author

Amar Nath Chaurasiya

Aspiring Data Scientist | Machine Learning Enthusiast | Automation & Industrial Systems Background

Skills Demonstrated
Python
PyTorch
SQL
SQLite
Streamlit
Computer Vision
Data Analysis
Deep Learning
Model Deployment
⭐ Project Impact

This project transforms a high-performing deep learning model into a complete AI-powered application capable of assisting users with tomato disease detection, diagnosis support, recommendation generation, and prediction tracking through an interactive web interface.

Achieved 99.5% Validation Accuracy using EfficientNet-B0 Transfer Learning. 🚀🌱