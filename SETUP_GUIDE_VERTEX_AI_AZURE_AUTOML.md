# 🚀 Complete Setup Guide: Vertex AI + Azure AutoML + Tableau Dashboard

## Table of Contents
1. [Vertex AI Setup](#vertex-ai-setup)
2. [Azure AutoML Setup](#azure-automl-setup)
3. [Model Comparison](#model-comparison)
4. [Tableau Dashboard Setup](#tableau-dashboard-setup)

---

## Part 1: Vertex AI Setup

### Prerequisites
- Google Cloud Project with billing enabled
- Service account with appropriate permissions
- `gcloud` CLI installed

### Step 1.1: Create Google Cloud Project

```bash
# Set your project ID
export PROJECT_ID="your-project-id"
export REGION="us-central1"

# Create project (or use existing)
gcloud projects create $PROJECT_ID

# Set as active project
gcloud config set project $PROJECT_ID

# Enable required APIs
gcloud services enable \
  aiplatform.googleapis.com \
  compute.googleapis.com \
  container.googleapis.com \
  bigquery.googleapis.com \
  storage-api.googleapis.com
```

### Step 1.2: Create Service Account

```bash
# Create service account
gcloud iam service-accounts create vertex-ai-sa \
  --display-name="Vertex AI Service Account"

# Set project number
export PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format='value(projectNumber)')

# Grant permissions
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:vertex-ai-sa@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/aiplatform.admin"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:vertex-ai-sa@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/bigquery.admin"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:vertex-ai-sa@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/storage.admin"

# Create and download key
gcloud iam service-accounts keys create vertex-ai-key.json \
  --iam-account=vertex-ai-sa@${PROJECT_ID}.iam.gserviceaccount.com

# Set authentication
export GOOGLE_APPLICATION_CREDENTIALS="$(pwd)/vertex-ai-key.json"
```

### Step 1.3: Upload Data to BigQuery

```bash
# Create BigQuery dataset
bq mk --dataset \
  --location=$REGION \
  telco_churn_dataset

# Load CSV to BigQuery
bq load \
  --source_format=CSV \
  --skip_leading_rows=1 \
  telco_churn_dataset.telco_churn \
  processed_telco_data.csv
```

---

## Part 2: Azure AutoML Setup

### Prerequisites
- Azure subscription with billing enabled
- Azure Machine Learning workspace created
- Azure CLI installed

### Step 2.1: Create Azure ML Workspace

```bash
# Set variables
export SUBSCRIPTION_ID="your-subscription-id"
export RESOURCE_GROUP="telco-churn-rg"
export WORKSPACE_NAME="telco-churn-workspace"
export LOCATION="eastus"

# Create resource group
az group create \
  --name $RESOURCE_GROUP \
  --location $LOCATION

# Create ML workspace
az ml workspace create \
  --name $WORKSPACE_NAME \
  --resource-group $RESOURCE_GROUP \
  --location $LOCATION
```

### Step 2.2: Upload Data to Azure ML

```bash
# Create datastore reference
az ml datastore create \
  --name churn_datastore \
  --type azure_blob \
  --account-name your_storage_account \
  --container-name churn_container \
  --account-key your_storage_key

# Create dataset
az ml dataset register \
  --name telco_churn_data \
  --path telco_churn_dataset/processed_telco_data.csv \
  --datastore churn_datastore
```

---

## Part 3: Model Training & Comparison

See `vertex_ai_automl_trainer.py` and `azure_automl_trainer.py` for implementation.

---

## Part 4: Tableau Dashboard Setup

### Prerequisites
- Tableau Desktop or Tableau Server/Online
- Export files from model training:
  - `vertex_ai_model_results.csv`
  - `azure_automl_model_results.csv`
  - `model_comparison_metrics.csv`

### Step 4.1: Prepare Data for Tableau

1. **Combine Model Results**
   - Merge Vertex AI and Azure results
   - Normalize metric columns
   - Add model source column

2. **Create Supporting Datasets**
   - Confusion matrices (CSV format)
   - Feature importance data
   - Prediction probabilities (sample)

### Step 4.2: Build Tableau Dashboard

**Dashboard Components:**
1. Model Performance Comparison (bar charts)
2. Confusion Matrix Heatmaps
3. ROC Curve Overlay
4. Key Metrics KPIs
5. Feature Importance Rankings
6. Business Impact Analysis

See `TABLEAU_DASHBOARD_GUIDE.md` for detailed instructions.

---

## Next Steps

1. ✅ Complete Vertex AI setup (10-15 minutes)
2. ✅ Complete Azure AutoML setup (10-15 minutes)
3. ✅ Train models using provided Python scripts
4. ✅ Export results to CSV files
5. ✅ Create Tableau dashboard
6. ✅ Share dashboard with stakeholders

---

## Important Notes

- **Cost Monitoring**: Both Vertex AI and Azure charge for compute time. Monitor your spending in GCP/Azure consoles.
- **Model Training Time**: AutoML models can take 30-60 minutes to train depending on dataset size.
- **API Keys**: Keep your credentials secure. Never commit keys to GitHub.
- **Tableau Licensing**: Ensure you have appropriate Tableau licenses for your team.

