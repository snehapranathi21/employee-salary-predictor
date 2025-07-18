# app.py (Advanced Streamlit App for Employee Salary Prediction)
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Config
st.set_page_config(page_title="Salary Predictor", layout="wide")
st.title("💼 Employee Salary Prediction App")

# Load dataset
df = pd.read_csv("Salary_Data.csv")
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

# Encode categorical variables
le_gender = LabelEncoder().fit(df["Gender"])
le_edu = LabelEncoder().fit(df["Education Level"])
le_job = LabelEncoder().fit(df["Job Title"])
df["Gender"] = le_gender.transform(df["Gender"])
df["Education Level"] = le_edu.transform(df["Education Level"])
df["Job Title"] = le_job.transform(df["Job Title"])

# Features and Target
X = df.drop("Salary", axis=1)
y = df["Salary"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Evaluation
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Sidebar
st.sidebar.header("📊 Data Criteria")
age_range = st.sidebar.slider("Age Range", int(df["Age"].min()), int(df["Age"].max()), (25, 40))
exp_range = st.sidebar.slider("Experience Range", 0, 40, (0, 10))
genders = st.sidebar.multiselect("Gender", le_gender.classes_, default=list(le_gender.classes_))

# Filtered Data
filtered = df.copy()
filtered["Gender"] = le_gender.inverse_transform(filtered["Gender"])
filtered = filtered[
    (filtered["Age"] >= age_range[0]) & (filtered["Age"] <= age_range[1]) &
    (filtered["Years of Experience"] >= exp_range[0]) &
    (filtered["Years of Experience"] <= exp_range[1]) &
    (filtered["Gender"].isin(genders))
]

# Visualizations
col1, col2 = st.columns(2)
with col1:
    st.subheader("Salary Distribution")
    fig1, ax1 = plt.subplots()
    sns.histplot(filtered["Salary"], kde=True, ax=ax1, color="blue")
    st.pyplot(fig1)

with col2:
    st.subheader("Average Salary by Job Title")
    job_avg = filtered.groupby("Job Title")["Salary"].mean().sort_values(ascending=False)
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    job_avg.plot(kind='bar', color='green', ax=ax2)
    ax2.set_ylabel("Average Salary")
    st.pyplot(fig2)

# Prediction Form
st.subheader("🎯 Predict Employee Salary")
with st.form("prediction_form"):
    age = st.slider("Age", 18, 65, 30)
    gender = st.selectbox("Gender", le_gender.classes_)
    edu = st.selectbox("Education Level", le_edu.classes_)
    job = st.selectbox("Job Title", le_job.classes_)
    exp = st.slider("Years of Experience", 0, 40, 5)
    submitted = st.form_submit_button("Predict Salary")

    if submitted:
        input_df = pd.DataFrame([[age,
                                  le_gender.transform([gender])[0],
                                  le_edu.transform([edu])[0],
                                  le_job.transform([job])[0],
                                  exp]],
                                columns=X.columns)
        salary = model.predict(input_df)[0]
        st.success(f"Predicted Salary: ₹{salary:,.2f}")

        # Download Prediction
        download_df = pd.DataFrame.from_dict({
            "Age": [age],
            "Gender": [gender],
            "Education Level": [edu],
            "Job Title": [job],
            "Experience": [exp],
            "Predicted Salary": [salary]
        })


# Model evaluation section
st.subheader("📈 Model Evaluation")
st.write(f"**Training Score (R²):** {train_score:.2f}")
st.write(f"**Test Score (R²):** {test_score:.2f}")
st.write(f"**Mean Squared Error:** {mse:.2f}")
st.write(f"**R² Score on Test Set:** {r2:.2f}")

