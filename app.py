import streamlit as st
import pandas as pd
import numpy as np
from tinydb import TinyDB, Query
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

st.set_page_config(page_title="Stress Level Prediction", layout="centered")

st.markdown("""
<style>
/* MAIN APP BACKGROUND */
.stApp {
    background: linear-gradient(rgba(98,0,150,0.85), rgba(98,0,150,0.85)),
    url("https://images.unsplash.com/photo-1516822003754-cca485356ecb");
    background-size: cover;
    background-position: center;
}

/* TEXT COLORS */
label, h1, h2, h3, p {
    color: white !important;
}

/* INPUT FIELDS */
input, textarea {
    background-color: rgba(255,255,255,0.95) !important;
    color: black !important;
}

/* BUTTONS */
button {
    background-color: #7d3cff !important;
    color: white !important;
}

/* SIDEBAR BACKGROUND */
section[data-testid="stSidebar"] {
    background-color: #4b0082 !important;
}

/* SIDEBAR TITLE */
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] label {
    color: #ffffff !important;
    font-weight: 600;
}

/* SIDEBAR SELECTBOX */
section[data-testid="stSidebar"] .stSelectbox div {
    background-color: rgba(255,255,255,0.15) !important;
    color: white !important;
}

/* SELECTBOX TEXT */
section[data-testid="stSidebar"] .stSelectbox span {
    color: white !important;
}

/* DROPDOWN OPTIONS */
div[role="listbox"] {
    background-color: #5e2b97 !important;
    color: white !important;
}

/* MENU HOVER */
div[role="option"]:hover {
    background-color: #7d3cff !important;
}
</style>
""", unsafe_allow_html=True)


db = TinyDB("users.json")
User = Query()

st.title("Stress Level Prediction in Sleep Patterns")
st.write("This system predicts stress levels using sleep-related health parameters and compares multiple machine learning algorithms.")

data = pd.read_csv(r"C:\Users\DELL\PycharmProjects\HumanStress\.venv\New_HumanStress_SleepingHabits_2026\Dataset\SaYoPillow.csv")
data.columns = [
    "snoring_rate",
    "respiration_rate",
    "body_temperature",
    "limb_movement",
    "blood_oxygen",
    "eye_movement",
    "sleeping_hours",
    "heart_rate",
    "stress_level"
]

X = data.drop("stress_level", axis=1)
y = data["stress_level"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

nb_model = GaussianNB().fit(X_train, y_train)
lr_model = LogisticRegression(max_iter=500).fit(X_train, y_train)
svm_model = SVC().fit(X_train, y_train)
rf_model = RandomForestClassifier().fit(X_train, y_train)
knn_model = KNeighborsClassifier().fit(X_train, y_train)
dt_model = DecisionTreeClassifier().fit(X_train, y_train)
mlp_model = MLPClassifier(max_iter=300).fit(X_train, y_train)

stress_level_labels = {
    0: "Low/Normal",
    1: "Medium Low",
    2: "Medium",
    3: "Medium High",
    4: "High"
}

if "auth" not in st.session_state:
    menu = st.sidebar.selectbox("Menu", ["Register", "Login"])
else:
    menu = st.sidebar.selectbox("Menu", ["Predict", "Logout"])

if menu == "Register":
    st.subheader("Register")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    full_name = st.text_input("Full Name")
    email = st.text_input("Email")
    age = st.number_input("Age", min_value=1, max_value=120)

    if st.button("Register"):
        if not all([username.strip(), password.strip(), full_name.strip(), email.strip(), age > 0]):
            st.error("All fields are required")
        elif db.search(User.username == username.strip()):
            st.error("Username already exists")
        else:
            db.insert({
                "username": username.strip(),
                "password": password.strip(),
                "full_name": full_name.strip(),
                "email": email.strip(),
                "age": age
            })
            st.success("Registration successful")

elif menu == "Login":
    st.subheader("Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if not username.strip() or not password.strip():
            st.error("Username and Password cannot be empty")
        elif db.search((User.username == username.strip()) & (User.password == password.strip())):
            st.session_state["auth"] = True
            st.success("Login successful")
            st.rerun()
        else:
            st.error("Invalid credentials")

elif menu == "Predict":
    if "auth" not in st.session_state:
        st.warning("Please login first")
    else:
        st.subheader("Enter Sleep & Health Parameters")

        c1, c2 = st.columns(2)
        with c1:
            snoring_rate = st.number_input("Snoring Rate", value=40.0)
            respiration_rate = st.number_input("Respiration Rate", value=15.0)
            body_temperature = st.number_input("Body Temperature", value=36.5)
            limb_movement = st.number_input("Limb Movement", value=2.0)
        with c2:
            blood_oxygen = st.number_input("Blood Oxygen", value=92.0)
            eye_movement = st.number_input("Eye Movement", value=70.0)
            sleeping_hours = st.number_input("Sleeping Hours", value=7.0)
            heart_rate = st.number_input("Heart Rate", value=90.0)

        if st.button("Predict Stress Level"):
            new_data = pd.DataFrame([[
                snoring_rate,
                respiration_rate,
                body_temperature,
                limb_movement,
                blood_oxygen,
                eye_movement,
                sleeping_hours,
                heart_rate
            ]], columns=X.columns)

            pred = nb_model.predict(new_data)[0]
            label = stress_level_labels[pred]
            st.success(f"Predicted Stress Level: {pred} ({label})")

        st.subheader("Algorithm Comparison")

        acc_scores = {
            "Logistic Regression": accuracy_score(y_test, lr_model.predict(X_test)),
            "Naive Bayes": accuracy_score(y_test, nb_model.predict(X_test)),
            "SVM": accuracy_score(y_test, svm_model.predict(X_test)),
            "Random Forest": accuracy_score(y_test, rf_model.predict(X_test)),
            "KNN": accuracy_score(y_test, knn_model.predict(X_test)),
            "Decision Tree": accuracy_score(y_test, dt_model.predict(X_test)),
            "MLP": accuracy_score(y_test, mlp_model.predict(X_test))
        }

        fig, ax = plt.subplots(figsize=(10, 5))

        colors = plt.cm.tab10(range(len(acc_scores)))
        bars = ax.bar(acc_scores.keys(), acc_scores.values(), color=colors)

        ax.set_ylim(0, 1.1)  # EXTRA HEADROOM
        ax.set_ylabel("Accuracy")
        ax.set_title("Algorithm Accuracy Comparison", pad=20)  # TITLE SPACING
        ax.tick_params(axis="x", rotation=30)

        # VALUE LABELS (LOWERED A BIT)
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.02,  # LOWER OFFSET
                f"{h:.2f}",
                ha="center",
                va="bottom",
                fontsize=9
            )

        plt.tight_layout(rect=[0, 0, 1, 0.92])  # PUSH GRAPH DOWN
        st.pyplot(fig)


elif menu == "Logout":
    st.session_state.clear()
    st.success("Logged out successfully")
    st.rerun()
