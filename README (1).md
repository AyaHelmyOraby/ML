# Near-Earth Object (NEO) Hazard Classification

## Overview
This project analyzes Near-Earth Objects (NEOs) to determine their potential hazard risk. Using data preprocessing, feature engineering, and machine learning models, we classify whether an asteroid is hazardous or not based on various attributes.

## Dataset
The dataset contains information about NEOs, including:
- **absolute_magnitude**: Brightness of the object
- **estimated_diameter_min/max**: Size range in kilometers
- **orbiting_body**: The celestial body the NEO orbits
- **relative_velocity**: Speed relative to Earth
- **miss_distance**: Distance from Earth
- **is_hazardous**: Target variable (True/False)

## Data Preprocessing
- **Handling Missing Values**: Imputed missing data using mean for numerical columns.
- **Feature Scaling**: Normalized numerical features using StandardScaler.
- **Encoding Categorical Data**: Used Label Encoding for categorical variables.
- **Handling Imbalanced Classes**: Applied SMOTE (Synthetic Minority Over-sampling Technique) to balance the dataset.

## Exploratory Data Analysis (EDA)
We conducted thorough data visualization using **Matplotlib**, **Seaborn**, and **Plotly** to identify trends:
- Heatmaps to examine correlations between features.
- Scatter plots and histograms to visualize distributions.
- Box plots to detect outliers.

## Model Training and Evaluation
We trained multiple machine learning models:
1. **Logistic Regression**
2. **Random Forest Classifier**
3. **Gradient Boosting Classifier**

### Performance Metrics:
- **Accuracy**
- **Precision & Recall**
- **F1-Score**
- **ROC-AUC Score**

The **Random Forest Classifier** performed best with an F1-score of **0.91** and an AUC-ROC of **0.949**.

---

## Code Sections

### 1️⃣ Load Data
```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve

# df = pd.read_csv('your_dataset.csv')  # Uncomment and update with your dataset path
```

### 2️⃣ Feature Selection & Preprocessing
```python
df.drop(columns=["neo_id", "name", "orbiting_body"], errors='ignore', inplace=True)
df.fillna(df.median(), inplace=True)
df["is_hazardous"] = df["is_hazardous"].astype(int)

X = df.drop(columns=["is_hazardous"])
y = df["is_hazardous"]
```

### 3️⃣ Train-Test Split
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
```

### 4️⃣ Normalize Features
```python
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

### 5️⃣ Handle Imbalance (SMOTE)
```python
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)
```

### 6️⃣ Define Models
```python
models = {
    "Logistic Regression": LogisticRegression(class_weight="balanced", random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", scale_pos_weight=y_train.value_counts()[0] / y_train.value_counts()[1])
}
```

### 7️⃣ Train & Evaluate Each Model
```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve



# **2️⃣ Feature Selection & Preprocessing**
df.drop(columns=["neo_id", "name", "orbiting_body"], errors='ignore', inplace=True)
df.fillna(df.median(), inplace=True)
df["is_hazardous"] = df["is_hazardous"].astype(int)

X = df.drop(columns=["is_hazardous"])
y = df["is_hazardous"]

# **3️⃣ Train-Test Split**
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# **4️⃣ Normalize Features**
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# **5️⃣ Handle Imbalance (SMOTE)**
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# **6️⃣ Define Models**
models = {
    "Logistic Regression": LogisticRegression(class_weight="balanced", random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", scale_pos_weight=y_train.value_counts()[0] / y_train.value_counts()[1])
}

# **7️⃣ Train & Evaluate Each Model**
best_model = None
best_auc = 0
results = {}

plt.figure(figsize=(8, 6))
for name, model in models.items():
    print(f"🔹 Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # **Metrics**
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    report = classification_report(y_test, y_pred)
    
    # Store results
    results[name] = {
        "Accuracy": accuracy,
        "AUC": auc,
        "Report": report
    }

    # Select best model based on AUC
    if auc > best_auc:
        best_auc = auc
        best_model = model

    # **AUC-ROC Curve**
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")


```

### 8️⃣ Print Results
```python
for name, result in results.items():
    print(f"\n🔹 Model: {name}")
    print(f"✅ Accuracy: {result['Accuracy']:.3f}")
    print(f"✅ AUC Score: {result['AUC']:.3f}")
    print(f"✅ Classification Report:\n{result['Report']}")
```

### 9️⃣ Confusion Matrix for Best Model
```python
best_model_name = [name for name, model in models.items() if model == best_model][0]
print(f"\n🎯 Best Model: {best_model_name} (AUC = {best_auc:.3f})")

plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("AUC-ROC Curve")
plt.legend()
plt.show()
```

---

## Key Findings
- **Larger asteroids tend to be more hazardous**.
- **Relative velocity has a moderate correlation with hazard classification**.
- **Miss distance alone is not a strong predictor of hazard risk**.

## Next Steps
- Experiment with deep learning models (Neural Networks).
- Explore additional feature engineering.
- Deploy the best-performing model as a web service.

## Contributors
- **[Aya Oraby]** – Data Science & Machine Learning

## License
This project is licensed under the MIT License.

