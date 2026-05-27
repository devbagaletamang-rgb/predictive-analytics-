# =====================================================
# TELCO CHURN PREDICTION - ENHANCED MODEL BUILDING
# =====================================================
# Best Practices:
# - Exploratory Data Analysis (EDA)
# - Data preprocessing & validation
# - Hyperparameter tuning
# - Cross-validation
# - Class imbalance handling
# - Feature importance analysis
# - Model persistence
# =====================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pickle
import joblib
from datetime import datetime

# Machine Learning Libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             roc_auc_score, confusion_matrix, roc_curve, auc, 
                             classification_report, matthews_corrcoef)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

warnings.filterwarnings('ignore')

# =====================================================
# 1. LOAD DATA & EXPLORATORY DATA ANALYSIS (EDA)
# =====================================================
print("=" * 60)
print("STEP 1: Loading Data & Exploratory Data Analysis")
print("=" * 60)

df = pd.read_csv("processed_telco_data.csv")

print(f"\n📊 Dataset Shape: {df.shape}")
print(f"\n📋 First Few Rows:\n{df.head()}")

print(f"\n🔍 Data Types:\n{df.dtypes}")
print(f"\n❌ Missing Values:\n{df.isnull().sum()}")

# Check class distribution
print(f"\n📈 Target Variable Distribution:")
print(df["Churn"].value_counts())
print(f"\nClass Imbalance Ratio: {df['Churn'].value_counts()[1] / df['Churn'].value_counts()[0]:.2%}")

# =====================================================
# 2. DATA PREPROCESSING
# =====================================================
print("\n" + "=" * 60)
print("STEP 2: Data Preprocessing")
print("=" * 60)

df = df.copy()
label_encoders = {}

# Encode categorical variables and store encoders for future use
categorical_cols = df.select_dtypes(include=['object']).columns
print(f"\n🔤 Encoding {len(categorical_cols)} categorical columns...")

for col in categorical_cols:
    label_enc = LabelEncoder()
    df[col] = label_enc.fit_transform(df[col])
    label_encoders[col] = label_enc
    print(f"  ✓ {col}")

# =====================================================
# 3. TRAIN-TEST SPLIT
# =====================================================
print("\n" + "=" * 60)
print("STEP 3: Train-Test Split")
print("=" * 60)

X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n✓ Training Set Size: {X_train.shape}")
print(f"✓ Test Set Size: {X_test.shape}")
print(f"✓ Training Set Class Distribution:\n{y_train.value_counts()}")

# =====================================================
# 4. FEATURE SCALING
# =====================================================
print("\n" + "=" * 60)
print("STEP 4: Feature Scaling")
print("=" * 60)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✓ Features scaled using StandardScaler")

# =====================================================
# 5. HANDLE CLASS IMBALANCE WITH SMOTE
# =====================================================
print("\n" + "=" * 60)
print("STEP 5: Handling Class Imbalance (SMOTE)")
print("=" * 60)

smote = SMOTE(random_state=42, k_neighbors=5)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

print(f"\n✓ Original Training Set Balance:\n{pd.Series(y_train).value_counts()}")
print(f"\n✓ After SMOTE Balancing:\n{pd.Series(y_train_balanced).value_counts()}")

# =====================================================
# 6. TRAIN BASELINE MODELS WITH HYPERPARAMETER TUNING
# =====================================================
print("\n" + "=" * 60)
print("STEP 6: Hyperparameter Tuning via GridSearchCV")
print("=" * 60)

# Define models with hyperparameter grids
model_params = {
    "Logistic Regression": {
        "model": LogisticRegression(max_iter=1000, random_state=42),
        "params": {
            'C': [0.001, 0.01, 0.1, 1, 10],
            'penalty': ['l2'],
            'solver': ['lbfgs']
        }
    },
    "Decision Tree": {
        "model": DecisionTreeClassifier(random_state=42),
        "params": {
            'max_depth': [3, 5, 7, 10, 15],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    },
    "Naive Bayes": {
        "model": GaussianNB(),
        "params": {
            'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
        }
    },
    "Random Forest": {
        "model": RandomForestClassifier(random_state=42, n_jobs=-1),
        "params": {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15],
            'min_samples_split': [2, 5]
        }
    },
    "Gradient Boosting": {
        "model": GradientBoostingClassifier(random_state=42),
        "params": {
            'n_estimators': [50, 100, 150],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7]
        }
    }
}

best_models = {}
results = []

for name, config in model_params.items():
    print(f"\n🔍 Tuning {name}...")
    
    grid_search = GridSearchCV(
        config['model'],
        config['params'],
        cv=5,
        scoring='f1',
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X_train_balanced, y_train_balanced)
    best_models[name] = grid_search.best_estimator_
    
    print(f"  ✓ Best Parameters: {grid_search.best_params_}")
    print(f"  ✓ Best CV Score: {grid_search.best_score_:.4f}")

# =====================================================
# 7. EVALUATE MODELS
# =====================================================
print("\n" + "=" * 60)
print("STEP 7: Model Evaluation on Test Set")
print("=" * 60)

for name, model in best_models.items():
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-Score": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_prob),
        "MCC": matthews_corrcoef(y_test, y_pred)
    })

results_df = pd.DataFrame(results).sort_values('F1-Score', ascending=False)
print("\n" + results_df.to_string(index=False))

# Save results to CSV
results_df.to_csv('model_results.csv', index=False)
print("\n✓ Results saved to 'model_results.csv'")

# =====================================================
# 8. CROSS-VALIDATION SCORES
# =====================================================
print("\n" + "=" * 60)
print("STEP 8: Cross-Validation Scores (5-Fold)")
print("=" * 60)

for name, model in best_models.items():
    cv_scores = cross_val_score(model, X_train_balanced, y_train_balanced, cv=5, scoring='f1')
    print(f"\n{name}:")
    print(f"  CV Scores: {cv_scores}")
    print(f"  Mean CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# =====================================================
# 9. FEATURE IMPORTANCE (for tree-based models)
# =====================================================
print("\n" + "=" * 60)
print("STEP 9: Feature Importance Analysis")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Feature Importance (Top 10)', fontsize=16, fontweight='bold')

tree_models = {
    "Decision Tree": best_models["Decision Tree"],
    "Random Forest": best_models["Random Forest"],
    "Gradient Boosting": best_models["Gradient Boosting"]
}

for idx, (name, model) in enumerate(tree_models.items()):
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importances
    }).sort_values('Importance', ascending=False).head(10)
    
    axes[idx].barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
    axes[idx].set_title(name)
    axes[idx].set_xlabel('Importance')

plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
print("✓ Feature importance plot saved as 'feature_importance.png'")
plt.show()

# =====================================================
# 10. CONFUSION MATRICES
# =====================================================
print("\n" + "=" * 60)
print("STEP 10: Confusion Matrices")
print("=" * 60)

plt.figure(figsize=(16, 10))

for i, (name, model) in enumerate(best_models.items()):
    plt.subplot(2, 3, i+1)
    
    y_pred = model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred)
    
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"{name}\nConfusion Matrix", fontweight='bold')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
print("✓ Confusion matrices saved as 'confusion_matrices.png'")
plt.show()

# =====================================================
# 11. ROC CURVES
# =====================================================
print("\n" + "=" * 60)
print("STEP 11: ROC Curves Comparison")
print("=" * 60)

plt.figure(figsize=(10, 8))

for name, model in best_models.items():
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})", linewidth=2)

plt.plot([0, 1], [0, 1], "k--", label="Random Classifier", linewidth=2)
plt.xlabel("False Positive Rate", fontsize=12)
plt.ylabel("True Positive Rate", fontsize=12)
plt.title("ROC Curves - Model Comparison", fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=10)
plt.grid(True, alpha=0.3)
plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
print("✓ ROC curves saved as 'roc_curves.png'")
plt.show()

# =====================================================
# 12. DETAILED CLASSIFICATION REPORTS
# =====================================================
print("\n" + "=" * 60)
print("STEP 12: Detailed Classification Reports")
print("=" * 60)

for name, model in best_models.items():
    y_pred = model.predict(X_test_scaled)
    print(f"\n{'='*60}")
    print(f"{name}")
    print(f"{'='*60}")
    print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))

# =====================================================
# 13. SAVE BEST MODEL & ARTIFACTS
# =====================================================
print("\n" + "=" * 60)
print("STEP 13: Saving Models & Artifacts")
print("=" * 60)

# Get best model
best_model_name = results_df.iloc[0]['Model']
best_model = best_models[best_model_name]

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Save best model
model_path = f"best_model_{best_model_name.replace(' ', '_')}_{timestamp}.pkl"
joblib.dump(best_model, model_path)
print(f"✓ Best Model ({best_model_name}) saved as '{model_path}'")

# Save scaler
scaler_path = f"scaler_{timestamp}.pkl"
joblib.dump(scaler, scaler_path)
print(f"✓ Scaler saved as '{scaler_path}'")

# Save label encoders
encoders_path = f"label_encoders_{timestamp}.pkl"
joblib.dump(label_encoders, encoders_path)
print(f"✓ Label Encoders saved as '{encoders_path}'")

# Save feature names
features_path = f"feature_names_{timestamp}.pkl"
joblib.dump(X.columns.tolist(), features_path)
print(f"✓ Feature names saved as '{features_path}'")

print("\n" + "=" * 60)
print("✅ MODEL BUILDING COMPLETE!")
print("=" * 60)
print(f"\n🏆 Best Model: {best_model_name}")
print(f"📊 Best F1-Score: {results_df.iloc[0]['F1-Score']:.4f}")
print(f"📈 Best ROC-AUC: {results_df.iloc[0]['ROC-AUC']:.4f}")
