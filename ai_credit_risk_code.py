import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, mean_squared_error, mean_absolute_error,
    confusion_matrix
)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from captum.attr import IntegratedGradients, NoiseTunnel
import shap
from sklearn.metrics import precision_score, recall_score, f1_score

# --- Load and preprocess dataset ---
file_path = "/Users/wujiayi/Desktop/FTD/AML/credit_risk_dataset.csv"
df = pd.read_csv(file_path)
if 'loan_grade' in df.columns:
    df = df.drop(columns=['loan_grade'])
    
df = df[df['person_age'] <= 100]
df = df[df['person_emp_length'] <= df['person_age']]

# Fill missing numerical values with median
df['person_emp_length'].fillna(df['person_emp_length'].median(), inplace=True)
df['loan_int_rate'].fillna(df['loan_int_rate'].median(), inplace=True)

# One-Hot Encoding for categorical variables
df = pd.get_dummies(
    df,
    columns=[
        'person_home_ownership',
        'loan_intent',
        'cb_person_default_on_file'
    ],
    drop_first=True
)

# Define dependent (y) and independent variables (X)
y = df['loan_status']
X = df.drop(columns=['loan_status'])

# --- Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# --- Log-transform features for all models ---
# Apply log1p to handle zeros and skew
X_train = np.log1p(X_train)
X_test  = np.log1p(X_test)

# Ensure correct dtypes
X_train = X_train.astype('float64')
X_test  = X_test.astype('float64')

# --- Classical ML models and performance + confusion matrices ---
results = []
models = {}
predictions = {}
probas = {}

# Logistic Regression
logistic = LogisticRegression(solver='liblinear')
logistic.fit(X_train, y_train)
models['Logistic Regression'] = logistic
predictions['Logistic Regression'] = logistic.predict(X_test)
probas['Logistic Regression'] = logistic.predict_proba(X_test)[:, 1]
threshold = 0.2
predictions['Logistic Regression'] = (probas['Logistic Regression'] >= threshold).astype(int)
recall_lg = recall_score(y_test, predictions['Logistic Regression'])
precision_lg = precision_score(y_test, predictions['Logistic Regression'])
f1_score_lg = f1_score(y_test, predictions['Logistic Regression'])
print(f"LG Recall: {recall_lg:.3f}")
print(f"LG Precision: {precision_lg:.3f}")
print(f"LG F1 Score: {f1_score_lg:.3f}")

# SVM
svm = SVC(probability=True, kernel='rbf', C=1)
svm.fit(X_train, y_train)
models['SVM'] = svm
predictions['SVM'] = svm.predict(X_test)
probas['SVM'] = svm.predict_proba(X_test)[:, 1]
threshold = 0.2
predictions['SVM'] = (probas['SVM'] >= threshold).astype(int)
recall_svm = recall_score(y_test, predictions['SVM'])
precision_svm = precision_score(y_test, predictions['SVM'])
f1_score_svm = f1_score(y_test, predictions['SVM'])
print(f"SVM Recall: {recall_svm:.3f}")
print(f"SVM Precision: {precision_svm:.3f}")
print(f"SVM F1 Score: {f1_score_svm:.3f}")

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
models['Random Forest'] = rf
predictions['Random Forest'] = rf.predict(X_test)
probas['Random Forest'] = rf.predict_proba(X_test)[:, 1]
threshold = 0.2
predictions['Random Forest'] = (probas['Random Forest'] >= threshold).astype(int)
recall_rf = recall_score(y_test, predictions['Random Forest'])
precision_rf = precision_score(y_test, predictions['Random Forest'])
f1_score_rf = f1_score(y_test, predictions['Random Forest'])
print(f"RF Recall: {recall_rf:.3f}")
print(f"RF Precision: {precision_rf:.3f}")
print(f"RF F1 Score: {f1_score_rf:.3f}")

# XGBoost
xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
xgb.fit(X_train, y_train)
models['XGBoost'] = xgb
predictions['XGBoost'] = xgb.predict(X_test)
probas['XGBoost'] = xgb.predict_proba(X_test)[:, 1]
threshold = 0.2
predictions['XGBoost'] = (probas['XGBoost'] >= threshold).astype(int)
recall_xgb = recall_score(y_test, predictions['XGBoost'])
precision_xgb = precision_score(y_test, predictions['XGBoost'])
f1_score_xgb = f1_score(y_test, predictions['XGBoost'])
print(f"XGB Recall: {recall_xgb:.3f}")
print(f"XGB Precision: {precision_xgb:.3f}")
print(f"XGB F1 Score: {f1_score_xgb:.3f}")

# Compute metrics and plot confusion matrices
for name in models:
    y_pred = predictions[name]
    y_prob = probas[name]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    results.append({
        'Model': name,
        'Accuracy': acc,
        'AUC-ROC': auc,
        'MSE': mse,
        'MAE': mae
    })
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    #plt.savefig(f"/Users/wujiayi/Desktop/FTD/AML/graphs/{name}_confusion_matrix.png")
    plt.show()

# Display classical results
results_df = pd.DataFrame(results).set_index('Model')
print("Classical model performance:\n", results_df)

# --- SHAP Global Explanation for XGBoost ---
#explainer_shap = shap.Explainer(xgb, X_train)
#shap_values = explainer_shap(X_test)

feature_rename_dict = {
    'loan_percent_income': 'Percentage of loan to income',
    'loan_int_rate': 'Loan interest rate',
    'person_age': 'Age',
    'person_income': 'Income',
    'loan_intent_VENTURE': 'Loan for Venture',
    'loan_intent_EDUCATION': 'Loan for Education',
    'loan_intent_HOMEIMPROVEMENT': 'Loan for Home Improvement',
    'loan_amnt': 'Amount of the loan',
    'person_emp_length': 'Employment length',
    'loan_intent_PERSONAL': 'Loan for Personal Reasons',
    'loan_intent_MEDICAL': 'Loan for Medical Reasons',
    'cb_person_cred_hist_length': 'Credit history length',
    'person_home_ownership_RENT': 'Home-Rent',
    'person_home_ownership_OWN': 'Home-Own',
    'cb_person_default_on_file_Y': 'Historical default',
    'person_home_ownership_OTHER': 'Home-Other'

}

X_train_named = X_train.copy()
X_train_named.columns = [feature_rename_dict.get(col, col) for col in X_train.columns]

X_test_named = X_test.copy()
X_test_named.columns = [feature_rename_dict.get(col, col) for col in X_test.columns]

explainer_shap = shap.Explainer(xgb, X_train_named)
shap_values = explainer_shap(X_test_named)

shap.summary_plot(shap_values, X_test_named, color=plt.get_cmap("Blues"))

#shap.summary_plot(shap_values, X_test,color=plt.get_cmap("Blues"))
plt.tight_layout()
#plt.savefig("/Users/wujiayi/Desktop/FTD/AML/graphs/shap_summary_plot.png", dpi=300)


#%%
# --- Neural Network on log-transformed features ---
# Use X_train and X_test already log-transformed
# Cast to float32 for torch
X_train_nn = X_train.astype('float32')
X_test_nn  = X_test.astype('float32')

X_train_tensor = torch.tensor(X_train_nn.values, dtype=torch.float32)
X_test_tensor  = torch.tensor(X_test_nn.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
y_test_tensor  = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

train_ds = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

class CreditRiskNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

model = CreditRiskNN(X_train_tensor.shape[1])
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train
for epoch in range(10):
    model.train()
    for xb, yb in train_loader:
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

model.eval()


# Confusion Matrix for NN
with torch.no_grad():
    probs_nn = model(X_test_tensor).squeeze().cpu().numpy()
    preds_nn = (probs_nn >= 0.2).astype(int)
cm_nn = confusion_matrix(y_test, preds_nn)

recall_nn = recall_score(y_test, preds_nn)
precision_nn = precision_score(y_test, preds_nn)
f1_score_nn = f1_score(y_test, preds_nn)
print(f"NN Recall: {recall_nn:.3f}")
print(f"NN Precision: {precision_nn:.3f}")
print(f"NN F1 Score: {f1_score_nn:.3f}")

plt.figure(figsize=(5,4))
sns.heatmap(cm_nn, annot=True, fmt='d', cmap='Blues')
plt.title("Neural Net Confusion Matrix")
plt.xlabel('Predicted')
plt.ylabel('Actual')
#plt.savefig(f"/Users/wujiayi/Desktop/FTD/AML/graphs/NN_confusion_matrix.png")
plt.show()

# Metrics for NN

auc = roc_auc_score(y_test, probs_nn)
mse = mean_squared_error(y_test, preds_nn)
mae = mean_absolute_error(y_test, preds_nn)

# Store in dictionary
nn_results = {
    'Model': 'Neural Network',
    'Accuracy': acc,
    'AUC-ROC': auc,
    'MSE': mse,
    'MAE': mae
}
print("Neural Network performance:\n", nn_results)


# --- Captum IG + SmoothGradSq for best attributions ---
# --- Captum Integrated Gradients on log-transformed inputs ---
ig = IntegratedGradients(model)
# Select first 10 samples
test_sample = X_test_tensor[:10]
# Compute attributions
attributions, delta = ig.attribute(
    test_sample,
    target=0,
    return_convergence_delta=True
)
# Convert attributions to numpy
attributions_np = attributions.detach().cpu().numpy()
feature_names = X_test.columns

# Display feature attributions for first sample
feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Attribution': attributions_np[0]
}).sort_values(by='Attribution', key=abs, ascending=False)

print("Feature importance for the first sample:")
print(feature_importance)

feature_importance['Feature'] = feature_importance['Feature'].map(
    feature_rename_dict
).fillna(feature_importance['Feature'])

# Plot top 15 attributions
plt.figure(figsize=(10, 6))
sns.barplot(
    x='Attribution',
    y='Feature',
    data=feature_importance.head(15)
)
plt.title("Top 15 Feature Attributions via Integrated Gradients")
plt.tight_layout()
#plt.savefig(f"/Users/wujiayi/Desktop/FTD/AML/graphs/Captum for NN.png")
plt.show()



#%%# LIME
from lime.lime_tabular import LimeTabularExplainer

# Create the explainer
lime_explainer = LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=X_train.columns.tolist(),
    class_names=['Not Default', 'Default'],
    mode='classification',
    random_state=42
)

# Pick one test sample to explain
i = 2  # index of the sample to explain
exp = lime_explainer.explain_instance(
    data_row=X_test.iloc[i],
    predict_fn=models['XGBoost'].predict_proba
)

# Show explanation in notebook or terminal
print("LIME explanation for XGBoost on test instance", i)

feature_rename_dict = {
    'loan_percent_income': 'Percentage of loan to income',
    'loan_int_rate': 'Loan interest rate',
    'person_age': 'Age',
    'person_income': 'Income',
    'loan_intent_VENTURE': 'Loan for Venture',
    'loan_intent_EDUCATION': 'Loan for Education',
    'loan_intent_HOMEIMPROVEMENT': 'Loan for Home Improvement',
    'loan_amnt': 'Amount of the loan',
    'person_emp_length': 'Employment length',
    'loan_intent_PERSONAL': 'Loan for Personal Reasons',
    'loan_intent_MEDICAL': 'Loan for Medical Reasons',
    'cb_person_cred_hist_length': 'Credit history length',
    'person_home_ownership_RENT': 'Home-Rent',
    'person_home_ownership_OWN': 'Home-Own',
    'cb_person_default_on_file_Y': 'Historical default',
    'person_home_ownership_OTHER': 'Home-Other'
}

explanation = exp.as_list()

# Replace feature names inside rule strings
renamed_explanation = []
for feature_str, weight in explanation:
    for raw, new in feature_rename_dict.items():
        if raw in feature_str:
            feature_str = feature_str.replace(raw, new)
    renamed_explanation.append((feature_str, weight))

# Create a custom bar plot
import pandas as pd
renamed_df = pd.DataFrame(renamed_explanation, columns=["Feature", "Weight"])

plt.figure(figsize=(8, 6))
sns.barplot(x="Weight", y="Feature", data=renamed_df, palette=["green" if w > 0 else "red" for w in renamed_df["Weight"]])
plt.title(f"LIME Explanation for Test Sample {i} (Renamed)")
plt.xlabel("Contribution to Prediction")
plt.tight_layout()
#plt.savefig("/Users/wujiayi/Desktop/FTD/AML/graphs/lime_summary_plot_renamed_3.png", dpi=300)
plt.show()


#%% Histogram of model metrics

metrics_data = {
    'Model': ['Logistic Regression', 'SVM', 'Random Forest', 'XGBoost'],
    'Recall': [recall_lg, recall_svm, recall_rf, recall_xgb],
    'Precision': [precision_lg, precision_svm, precision_rf, precision_xgb],
    'F1 Score': [f1_score_lg, f1_score_svm, f1_score_rf, f1_score_xgb]
}

metrics_df = pd.DataFrame(metrics_data)

##  1)
metrics_melted_1 = pd.melt(metrics_df, id_vars='Model', var_name='Metric', value_name='Score')

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(data=metrics_melted_1, x='Model', y='Score', hue='Metric')
plt.title("Recall, Precision, and F1 Score for Each Model (Threshold = 0.2)")
plt.ylim(0, 1)
plt.ylabel("Score")
plt.xlabel("Model")
plt.legend(title='Metric')
plt.tight_layout()
plt.show()


##  2)
metrics_melted_2 = pd.melt(metrics_df, id_vars='Model', var_name='Metric', value_name='Score')

# Plot grouped by Metric instead of Model
plt.figure(figsize=(10, 6))
sns.barplot(data=metrics_melted_2, x='Metric', y='Score', hue='Model')
plt.title("Model Comparison by Metric (Threshold = 0.2)")
plt.ylim(0, 1)
plt.ylabel("Score")
plt.xlabel("Metric")
plt.legend(title='Model')
plt.tight_layout()
plt.show()













