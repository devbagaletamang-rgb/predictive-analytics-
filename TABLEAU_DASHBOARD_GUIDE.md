````markdown name=TABLEAU_DASHBOARD_GUIDE.md url=https://github.com/devbagaletamang-rgb/predictive-analytics-/blob/main/TABLEAU_DASHBOARD_GUIDE.md
# 📊 Tableau Dashboard Setup Guide
## Telco Churn Prediction - Model Comparison & Results Visualization

---

## Part 1: Data Preparation for Tableau

### Step 1.1: Combine Model Results

After running both Vertex AI and Azure AutoML trainers, you'll have:
- `vertex_ai_model_results_*.csv`
- `azure_automl_model_results_*.csv`

**Create a unified results file:**

```python
import pandas as pd
import glob

# Load all Vertex AI results
vertex_files = glob.glob("vertex_ai_model_results_*.csv")
vertex_df = pd.concat([pd.read_csv(f) for f in vertex_files], ignore_index=True)
vertex_df['Platform'] = 'Vertex AI'

# Load all Azure results
azure_files = glob.glob("azure_automl_model_results_*.csv")
azure_df = pd.concat([pd.read_csv(f) for f in azure_files], ignore_index=True)
azure_df['Platform'] = 'Azure AutoML'

# Combine
combined_results = pd.concat([vertex_df, azure_df], ignore_index=True)
combined_results.to_csv('combined_model_results.csv', index=False)

print("✓ Combined results saved to: combined_model_results.csv")
```

### Step 1.2: Create Confusion Matrix Dataset

```python
import pandas as pd

# Sample confusion matrices (replace with actual values from your models)
confusion_matrices = pd.DataFrame({
    'Model': ['Vertex AI', 'Azure AutoML'],
    'True_Negatives': [8500, 8600],
    'False_Positives': [800, 700],
    'False_Negatives': [450, 380],
    'True_Positives': [1250, 1320]
})

confusion_matrices.to_csv('confusion_matrices.csv', index=False)
print("✓ Confusion matrices saved")
```

### Step 1.3: Create Feature Importance Dataset

```python
import pandas as pd

# Feature importance data (example)
feature_importance = pd.DataFrame({
    'Feature': ['Contract', 'Tenure', 'MonthlyCharges', 'InternetService', 
                'OnlineSecurity', 'TechSupport', 'PaymentMethod', 'Total_Charges'],
    'Vertex_AI_Importance': [0.28, 0.22, 0.18, 0.12, 0.08, 0.06, 0.04, 0.02],
    'Azure_Importance': [0.26, 0.24, 0.16, 0.14, 0.09, 0.05, 0.04, 0.02]
})

feature_importance.to_csv('feature_importance.csv', index=False)
print("✓ Feature importance data saved")
```

---

## Part 2: Setting Up Tableau

### Step 2.1: Connect Data Source

1. **Open Tableau Desktop or Online**
2. **Create New Workbook**
3. **Connect to Data:**
   - Click "Connect to Data"
   - Select "Text File" or upload CSV
   - Load `combined_model_results.csv`

### Step 2.2: Data Structure

Ensure your data has these columns:

| Column | Type | Example |
|--------|------|---------|
| Platform | String | "Vertex AI", "Azure AutoML" |
| Accuracy | Decimal | 0.8234 |
| Precision | Decimal | 0.7856 |
| Recall | Decimal | 0.8901 |
| F1_Score | Decimal | 0.8356 |
| ROC_AUC | Decimal | 0.9123 |
| Training_Date | Date | 2026-05-28 |

---

## Part 3: Building Dashboard Visualizations

### Dashboard Layout

```
┌─────────────────────────────────────────────────────┐
│           TELCO CHURN PREDICTION DASHBOARD          │
├──────────────────────────────────────────────────────┤
│  ┌──────────────────┐  ┌──────────────────────────┐ │
│  │  KPI Cards       │  │  Model Comparison       │ │
│  │  • Accuracy      │  │  (Bar Chart)            │ │
│  │  • Recall        │  └──────────────────────────┘ │
│  │  • F1-Score      │                               │
│  └──────────────────┘                               │
├──────────────────────────────────────────────────────┤
│  ┌──────────────────┐  ┌──────────────────────────┐ │
│  │ Confusion Matrices│  │  ROC Curves             │ │
│  │ (Heatmaps)       │  │  (Line Chart)           │ │
│  └──────────────────┘  └──────────────────────────┘ │
├──────────────────────────────────────────────────────┤
│  ┌────────────────────────────────────────────────┐ │
│  │  Feature Importance Ranking                    │ │
│  │  (Horizontal Bar Chart)                        │ │
│  └────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────┘
```

### Visualization 1: KPI Cards

**Sheet: Model Metrics Summary**

1. Create calculated fields for each metric:
   ```
   Metric Type: Accuracy, Precision, Recall, F1-Score
   ```

2. Create filters:
   - Platform filter (Vertex AI / Azure AutoML)
   - Metric type selector

3. Display using:
   - Text tables with conditional formatting
   - Color scale (Green for >0.85, Yellow for 0.75-0.85, Red for <0.75)

**Result:**
```
┌─────────────────────────────────────────────┐
│ VERTEX AI AUTOML                            │
├─────────────────────────────────────────────┤
│ Accuracy    │ 85.2%  │ ✅ Excellent        │
│ Precision   │ 78.6%  │ ✅ Good             │
│ Recall      │ 89.0%  │ ✅ Excellent        │
│ F1-Score    │ 83.6%  │ ✅ Good             │
│ ROC-AUC     │ 91.2%  │ ✅ Excellent        │
└─────────────────────────────────────────────┘
```

### Visualization 2: Model Comparison Bar Chart

**Sheet: Metrics Comparison**

1. **Rows:** Metric (Accuracy, Precision, Recall, F1-Score, ROC-AUC)
2. **Columns:** Platform
3. **Values:** Metric Value (SUM)
4. **Color:** By Platform
5. **Sort:** Descending by metric value

**Chart Type:** Grouped Bar Chart

```
Accuracy
├─ Vertex AI:    ████████░░  85.2%
└─ Azure AutoML: █████████░  86.1%

Precision
├─ Vertex AI:    ███████░░░  78.6%
└─ Azure AutoML: ████████░░  81.2%

Recall
├─ Vertex AI:    █████████░  89.0%
└─ Azure AutoML: ████████░░  87.3%

[Chart continues...]
```

### Visualization 3: Confusion Matrix Heatmaps

**Sheet: Confusion Matrices**

1. **Create two separate sheets** (one per model)
2. **Data structure:**

| Model | Classification | Predicted_Yes | Predicted_No |
|-------|---|---|---|
| Vertex AI | Actual_Yes | 1250 (TP) | 450 (FN) |
| Vertex AI | Actual_No | 800 (FP) | 8500 (TN) |
| Azure AutoML | Actual_Yes | 1320 (TP) | 380 (FN) |
| Azure AutoML | Actual_No | 700 (FP) | 8600 (TN) |

3. **Tableau Setup:**
   - Rows: Actual Classification
   - Columns: Predicted Classification
   - Values: Count (as text annotation)
   - Color: SUM of Count (darker = higher)
   - Mark type: Square/Heatmap

**Result:**
```
              VERTEX AI
              Predicted Churn | Predicted No-Churn
Actual Churn  ████████░░ 1250 | ██░░░░░░░░ 450
Actual No-Ch  ███░░░░░░░ 800  | ████████████ 8500

              AZURE AUTOML
              Predicted Churn | Predicted No-Churn
Actual Churn  ████████░░ 1320 | ██░░░░░░░░ 380
Actual No-Ch  ██░░░░░░░░ 700  | ████████████ 8600
```

### Visualization 4: ROC Curve Comparison

**Sheet: ROC Curves**

Preparation (in Python):
```python
from sklearn.metrics import roc_curve, auc
import pandas as pd

# For Vertex AI
fpr_va, tpr_va, _ = roc_curve(y_test, y_pred_proba_va)
roc_auc_va = auc(fpr_va, tpr_va)

# For Azure
fpr_az, tpr_az, _ = roc_curve(y_test, y_pred_proba_az)
roc_auc_az = auc(fpr_az, tpr_az)

# Create ROC data
roc_data = pd.DataFrame({
    'False_Positive_Rate': list(fpr_va) + list(fpr_az),
    'True_Positive_Rate': list(tpr_va) + list(tpr_az),
    'Model': ['Vertex AI']*len(fpr_va) + ['Azure AutoML']*len(fpr_az),
    'AUC': [roc_auc_va]*len(fpr_va) + [roc_auc_az]*len(fpr_az)
})
roc_data.to_csv('roc_curves.csv', index=False)
```

**Tableau Setup:**
- Rows: True_Positive_Rate
- Columns: False_Positive_Rate
- Color: Model
- Line chart connecting points
- Add reference line (diagonal - random classifier)

**Result:**
```
    True Positive Rate
    1.0 │     ╱        ╱
        │    ╱ Vertex  ╱ Azure
        │   ╱ AI(91%)╱ AutoML(89%)
    0.5 │  ╱     ╱
        │ ╱   ╱
    0.0 └─────────────────
        0.0      0.5     1.0
        False Positive Rate
```

### Visualization 5: Feature Importance Ranking

**Sheet: Feature Importance**

1. **Rows:** Feature
2. **Columns:** Values
3. **Create two calculations:**
   ```
   Max Importance = MAX(Vertex_AI_Importance, Azure_Importance)
   Difference = ABS(Vertex_AI_Importance - Azure_Importance)
   ```

**Chart Type:** Horizontal Bar Chart

```
Feature Importance Ranking
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Contract          ████████████████████░░  28.0%
Tenure            ██████████████████░░░░  24.0%
MonthlyCharges    ████████████████░░░░░░  19.0%
InternetService   ███████████░░░░░░░░░░░  14.0%
OnlineSecurity    ████████░░░░░░░░░░░░░░  10.0%
TechSupport       ██████░░░░░░░░░░░░░░░░   7.0%
[continues...]
```

### Visualization 6: Business Impact Analysis

**Sheet: Business Metrics**

Create calculated fields:
```
Customers Saved = Recall * Total_Churners
Cost per Prevention = Retention_Cost / Customers_Saved
Revenue Recovered = Customers_Saved * Average_CLV
ROI = Revenue_Recovered / Total_Cost
```

**Display:**
- KPI tiles showing:
  - Total Churners Identified
  - Retention Success Rate
  - Estimated Revenue Recovered
  - Model ROI

---

## Part 4: Dashboard Assembly

### Step 4.1: Create Dashboard

1. Click "New Dashboard"
2. Set size: **Automatic** or **1920 x 1080**
3. Title: **"Telco Churn Prediction - Vertex AI vs Azure AutoML"**

### Step 4.2: Add Sheets

1. **Row 1:** KPI Cards + Model Comparison (side-by-side)
2. **Row 2:** Confusion Matrices (tabs for each model)
3. **Row 3:** ROC Curves Overlay
4. **Row 4:** Feature Importance
5. **Row 5:** Business Impact Metrics

### Step 4.3: Add Filters

Create sheet-level filters:
```
- Platform: Vertex AI / Azure AutoML
- Date Range: Training date filter
- Metric Type: Select specific metrics
```

### Step 4.4: Add Interactivity

1. **Dashboard Filters:**
   - Click on bar in comparison chart → updates all visualizations
   - Platform selector → filters all sheets

2. **Highlighting:**
   - Hover over model → highlights across dashboard
   - Click metric → shows detailed breakdown

3. **Tooltips:**
   - Show actual values on hover
   - Display calculation details
   - Show data quality metrics

---

## Part 5: Formatting & Styling

### Color Scheme

```
Primary Colors:
  • Vertex AI:    #1f77b4 (Blue)
  • Azure AutoML: #ff7f0e (Orange)
  • Neutral:      #999999 (Gray)

Metric Colors (Performance):
  • Excellent (>0.85): #2ca02c (Green)
  • Good (0.75-0.85):  #ffd700 (Gold)
  • Needs Improvement: #d62728 (Red)
```

### Font & Layout

```
Title:        Tableau Black, 24pt, Bold
Subtitle:     Tableau Regular, 14pt
Labels:       Tableau Regular, 11pt
Values:       Tableau Book, 12pt, Bold
Background:   White (#FFFFFF)
Borders:      Light Gray (#E8E8E8)
```

---

## Part 6: Sharing & Publishing

### Export Options

1. **PDF Report:**
   - File → Export as PDF
   - Include all 5 visualizations
   - Add cover page with summary

2. **Web Publishing (Tableau Server/Online):**
   ```
   File → Publish to Tableau Server
   Select Workbook: telco_churn_prediction
   Project: Analytics
   Permissions: View-only for stakeholders
   ```

3. **Interactive HTML:**
   ```
   File → Export as Image
   Or use Tableau Public for web sharing
   ```

### Sample Output Filename
```
Telco_Churn_Prediction_Dashboard_VertexAI_vs_AzureAutoML_2026-05-28.twbx
```

---

## Part 7: Refresh & Monitoring

### Automatic Data Refresh

1. **Tableau Server:**
   - Right-click Data Source
   - Schedule Refresh
   - Set frequency: Daily / Weekly

2. **Update CSV Files:**
   ```bash
   # Quarterly retraining
   python vertex_ai_automl_trainer.py
   python azure_automl_trainer.py
   
   # Combine & refresh
   python combine_results.py
   ```

### Dashboard Refresh Checklist

- [ ] Data files updated from both platforms
- [ ] New metrics calculated correctly
- [ ] All filters working
- [ ] Tooltip values accurate
- [ ] Performance optimized (<2s load time)

---

## Part 8: Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| Data not updating | Check file path, refresh data source |
| Calculations failing | Verify column names match exactly |
| Performance slow | Reduce data size, filter test set only |
| Filters not working | Check filter scope, ensure sheets linked |
| Colors not matching | Reset color palette, check formatting rules |

---

## Sample Dashboard Narrative

Use text boxes to guide viewers:

> ### 📊 Model Performance Summary
> 
> **Key Finding:** Both platforms achieved >85% accuracy.
> - **Vertex AI** excels at **Recall (89%)** - catches more churners
> - **Azure AutoML** shows better **Precision (81%)** - fewer false alarms
> 
> **Recommendation:** 
> Deploy Vertex AI for customer retention (prioritize recall)
> 
> **Business Impact:**
> - Identify ~1,250 at-risk customers/month
> - Potential revenue recovery: $2.5M annually

---

## Deliverables Checklist

- [ ] Combined model results CSV
- [ ] Confusion matrices CSV
- [ ] Feature importance CSV
- [ ] ROC curves data
- [ ] Tableau workbook (.twbx)
- [ ] Exported PDF dashboard
- [ ] Published to Tableau Server/Online
- [ ] Refresh schedule configured
- [ ] Access granted to stakeholders

````
