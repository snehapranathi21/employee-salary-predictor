# 💼 Employee Salary Prediction App (Streamlit & Machine Learning)

An **advanced Employee Salary Prediction web application** built using **Streamlit** and **Machine Learning**.  
The app predicts employee salaries based on **age, gender, education level, job title, and years of experience**, and also provides **interactive visual analytics and model evaluation metrics**

## 🚀 Features

- 🎯 Predict employee salary using a trained **Random Forest Regressor**
- 📊 Interactive filters (Age, Experience, Gender)
- 📈 Salary distribution visualization
- 📊 Average salary by job title chart
- 🧠 Machine learning model evaluation (R² score & MSE)
- 🖥️ User-friendly Streamlit interface
- 📥 Prediction result ready for download 


## 🛠️ Technologies Used

- **Python 3**
- **Streamlit** – Web application framework
- **Pandas & NumPy** – Data handling
- **Matplotlib & Seaborn** – Data visualization
- **Scikit-learn** – Machine learning model
  - RandomForestRegressor
  - Train-test split
  - Evaluation metrics
- **LabelEncoder** – Encoding categorical variables


## 📂 Project Structure
employee-salary-prediction/
│
├── app.py
├── Salary_Data.csv
├── requirements.txt
└── README.md

## 🧠 Machine Learning Model

The Employee Salary Prediction system uses a **Random Forest Regressor**, which is an ensemble learning method based on multiple decision trees. This model is chosen for its ability to handle non-linear relationships and mixed data types effectively.

### Model Details:
- **Algorithm:** Random Forest Regressor
- **Number of Estimators:** 200 decision trees
- **Train-Test Split:** 80% training, 20% testing
- **Random State:** 42

### Data Preprocessing:
- Removed missing and duplicate values
- Encoded categorical features (Gender, Education Level, Job Title) using **Label Encoding**
- Selected relevant features such as Age, Experience, Education Level, and Job Role

### Model Evaluation Metrics:
- **Training R² Score** – Measures how well the model fits training data
- **Test R² Score** – Evaluates model generalization on unseen data
- **Mean Squared Error (MSE)** – Measures average squared prediction error
- **R² Score** – Indicates overall model performance



## 🔮 Future Enhancements

The following improvements can be implemented to enhance the system further:

- Integrate additional machine learning models (XGBoost, Gradient Boosting, Linear Regression)
- Save and load the trained model using **Pickle or Joblib**
- Add salary prediction export option (CSV / Excel)
- Implement salary trend analysis over time
- Deploy the application on **Streamlit Cloud, AWS, or Azure**
- Include role-based authentication for HR users
- Improve UI with advanced dashboards and filters



## 📜 Conclusion

The Employee Salary Prediction application demonstrates a complete **end-to-end machine learning workflow**, including data preprocessing, model training, evaluation, and deployment using **Streamlit**.  
By leveraging a Random Forest Regressor, the system provides accurate and reliable salary predictions based on employee attributes.

This project is well-suited for **final-year academic projects**, **machine learning portfolios**, and **real-world HR analytics use cases**, showcasing practical implementation of data science concepts in a user-friendly web application.




## 📬 Feel Free to Contact
If you have any questions, suggestions, or would like to collaborate, feel free to reach out:

- **Name:** Sneha pranathi Guddinti
- **Email:** snehapranathi21@gmail.com  
- **LinkedIn:** https://www.linkedin.com/in/sneha-pranathi-guddinti-3176a029b/  
- **GitHub:** https://github.com/snehapranathi21 

I am always open to feedback, learning opportunities, and collaboration.

