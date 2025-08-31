import pandas as pd
import joblib
import gradio as gr
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
df = pd.read_csv(r"C:\Users\umesh\OneDrive\Desktop\ML\DRUG\drug200.csv")

# Encode categorical features
df["Sex"] = df["Sex"].map({"M": 1, "F": 0})
df["BP"] = df["BP"].map({"LOW": 0, "NORMAL": 1, "HIGH": 2})
df["Cholesterol"] = df["Cholesterol"].map({"NORMAL": 0, "HIGH": 1})

# Encode target
le = LabelEncoder()
y = le.fit_transform(df["Drug"])
X = df.drop(columns=["Drug"])

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train[["Age", "Na_to_K"]] = scaler.fit_transform(X_train[["Age", "Na_to_K"]])
X_test[["Age", "Na_to_K"]] = scaler.transform(X_test[["Age", "Na_to_K"]])

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")

# Save objects
joblib.dump(model, "logistic_drug_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(le, "label_encoder.pkl")

# Reload
model = joblib.load("logistic_drug_model.pkl")
scaler = joblib.load("scaler.pkl")
le = joblib.load("label_encoder.pkl")

sex_map = {"Male": 1, "Female": 0}
bp_map = {"LOW": 0, "NORMAL": 1, "HIGH": 2}
chol_map = {"NORMAL": 0, "HIGH": 1}

def predict_drug(age, sex, bp, chol, na_to_k):
    patient = {
        "Age": age,
        "Sex": sex_map[sex],
        "BP": bp_map[bp],
        "Cholesterol": chol_map[chol],
        "Na_to_K": na_to_k
    }
    df_input = pd.DataFrame([patient])
    df_input[["Age", "Na_to_K"]] = scaler.transform(df_input[["Age", "Na_to_K"]])

    pred = model.predict(df_input)
    return f"Predicted Drug: {le.inverse_transform(pred)[0]}"

# Confusion Matrix
target_names = le.classes_
plt.figure(figsize=(6,5))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="g", cbar=False,
            xticklabels=target_names, yticklabels=target_names, cmap="Blues")
plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))

# Gradio App
iface = gr.Interface(
    fn=predict_drug,
    inputs=[
        gr.Number(label="Age", precision=0),
        gr.Radio(["Male", "Female"], label="Sex"),
        gr.Radio(["LOW", "NORMAL", "HIGH"], label="Blood Pressure"),
        gr.Radio(["NORMAL", "HIGH"], label="Cholesterol"),
        gr.Number(label="Na_to_K Ratio")
    ],
    outputs="text",
    title="💊 Drug Prediction",
    description="Enter patient details to predict the most suitable drug."
)

iface.launch(share=True)  # use share=True in Colab
