# NeoFraud - AI-Powered Fraud Detection System

## 🌟 Overview

NeoFraud is a state-of-the-art fraud detection system that leverages multiple machine learning models to identify fraudulent transactions with exceptional accuracy. This Streamlit-powered web application provides an intuitive interface for data scientists and fraud analysts to:

- Upload transaction datasets
- Train multiple ML models simultaneously
- Compare model performance visually
- Gain insights through interactive visualizations

## 🚀 Key Features

- **Multi-Model Comparison**: Evaluate Naive Bayes, KNN, AdaBoost, and Decision Tree models side-by-side
- **Interactive Dashboards**: Beautiful, futuristic visualizations of model performance
- **Real-time Metrics**: Instant training and testing metrics for quick evaluation
- **Modern UI**: Sleek, dark-themed interface with gradient accents
- **Responsive Design**: Works seamlessly across desktop and mobile devices

## 🛠 Technologies Used

- **Python** (Primary programming language)
- **Streamlit** (Web application framework)
- **Scikit-learn** (Machine learning models)
- **Pandas/Numpy** (Data manipulation)
- **Plotly/Matplotlib** (Visualizations)
- **Imbalanced-learn** (Handling class imbalance)

## 📊 Supported Models

1. **Gaussian Naive Bayes**
   - Fast probabilistic classifier
   - Excellent baseline model

2. **K-Nearest Neighbors (KNN)**
   - Instance-based learning
   - Spatial reasoning capabilities

3. **AdaBoost Classifier**
   - Ensemble method with adaptive boosting
   - Improved performance through weak learners

4. **Decision Tree Classifier**
   - Interpretable tree-based model
   - Clear decision paths


## ⚙️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/neofraud.git
   cd neofraud
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the application:
   ```bash
   streamlit run app.py
   ```

## 📈 Performance Metrics

NeoFraud tracks and visualizes all key classification metrics:

- Accuracy
- Precision (for fraud class)
- Recall (for fraud class)
- F1-Score (for fraud class)
- Confusion matrices
- ROC curves (where applicable)

## 📁 Data Requirements

For optimal performance, your dataset should:

1. Be in CSV format
2. Contain a 'Class' column (0 for genuine, 1 for fraud)
3. Have all other columns as numeric features

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.

## ✉️ Contact

Project Maintainer - [Hana Elanshassy](mailto:hana.elanshassy2@gmail.com) and [ noureen Osama] 

Project Link: [https://github.com/HanaElanshassy/CreditCardFraudDetectionMachineLearning](https://github.com/HanaElanshassy/CreditCardFraudDetectionMachineLearning)

---

