import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestCentroid
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_wine

st.set_page_config(page_title="Nearest Centroid Classifier", layout="wide")
st.title("🍷 Nearest Centroid Classifier with Streamlit (Wine Dataset)")

# Load Wine dataset
def load_default_data():
    wine = load_wine(as_frame=True)
    df = wine.frame
    return df

# Sidebar: Upload file or use sample data
st.sidebar.header("📂 Upload or Use Sample Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.success("✅ Custom dataset loaded.")
else:
    data = load_default_data()
    st.info("ℹ️ Using built-in Wine dataset.")

# Dataset preview
st.subheader("🔎 Dataset Preview")
st.dataframe(data.head())

# Sidebar: Feature and target selection
st.sidebar.header("🛠 Feature & Target Selection")

columns = data.columns.tolist()
target_col_index = len(columns) - 1 if columns else 0  # Default to last column
target_col = st.sidebar.selectbox("Select Target Column", columns, index=target_col_index)
feature_cols = st.sidebar.multiselect(
    "Select Feature Columns", [col for col in columns if col != target_col], default=columns[:2]
)

# Check valid selections
if not feature_cols or not target_col:
    st.warning("⚠️ Please select features and a target column to continue.")
    st.stop()

# Prepare features and target
X = data[feature_cols]
y = data[target_col]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = NearestCentroid()
model.fit(X_train, y_train)

# Predictions and accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.subheader("📊 Model Performance")
st.write(f"**Accuracy:** `{accuracy:.2f}`")

# Input form for prediction
st.subheader("🔮 Make a Prediction")
with st.form("prediction_form"):
    input_values = []
    for feature in feature_cols:
        val = st.number_input(f"Enter {feature}", value=float(X[feature].mean()))
        input_values.append(val)
    submitted = st.form_submit_button("Predict")
    if submitted:
        try:
            input_array = np.array(input_values).reshape(1, -1)
            prediction = model.predict(input_array)[0]
            st.success(f"✅ Predicted class: **{prediction}**")
        except Exception as e:
            st.error(f"Error during prediction: {e}")

# Optional: Scatter plot for 2D visualization
if len(feature_cols) == 2:
    fig, ax = plt.subplots()
    for label in np.unique(y):
        idx = y == label
        ax.scatter(X.loc[idx, feature_cols[0]], X.loc[idx, feature_cols[1]], label=str(label))
    ax.set_xlabel(feature_cols[0])
    ax.set_ylabel(feature_cols[1])
    ax.legend(title="Class")
    st.pyplot(fig)
else:
    st.info("ℹ️ Select exactly 2 features to visualize a 2D scatter plot.")
