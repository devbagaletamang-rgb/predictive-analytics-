"""
=====================================================
VERTEX AI AUTOML - TELCO CHURN PREDICTION
=====================================================
Automated Machine Learning on Google Cloud Platform

Features:
- AutoML automatic model selection and tuning
- Distributed training on Google Cloud
- Built-in hyperparameter optimization
- Real-time predictions API
"""

import os
import time
import pandas as pd
import numpy as np
from datetime import datetime
from google.cloud import aiplatform
from google.cloud import bigquery
from google.cloud import storage
import json

# =====================================================
# CONFIGURATION
# =====================================================
PROJECT_ID = os.getenv("GCP_PROJECT_ID", "your-project-id")
REGION = "us-central1"
BUCKET_NAME = f"{PROJECT_ID}-telco-churn"
DATASET_ID = "telco_churn_dataset"
TABLE_ID = "telco_churn"
DISPLAY_NAME = "telco-churn-automl"

print("=" * 80)
print("VERTEX AI AUTOML - TELCO CHURN PREDICTION")
print("=" * 80)
print(f"\n📊 Configuration:")
print(f"   • Project ID: {PROJECT_ID}")
print(f"   • Region: {REGION}")
print(f"   • Bucket: {BUCKET_NAME}")
print(f"   • Dataset: {DATASET_ID}.{TABLE_ID}")

# =====================================================
# INITIALIZE VERTEX AI
# =====================================================
print("\n" + "=" * 80)
print("STEP 1: Initializing Vertex AI")
print("-" * 80)

aiplatform.init(project=PROJECT_ID, location=REGION)
bq_client = bigquery.Client(project=PROJECT_ID)
storage_client = storage.Client(project=PROJECT_ID)

print("✓ Vertex AI initialized")
print("✓ BigQuery client ready")
print("✓ Storage client ready")

# =====================================================
# CREATE STORAGE BUCKET
# =====================================================
print("\n" + "=" * 80)
print("STEP 2: Setting up Cloud Storage")
print("-" * 80)

try:
    bucket = storage_client.get_bucket(BUCKET_NAME)
    print(f"✓ Using existing bucket: {BUCKET_NAME}")
except Exception:
    bucket = storage_client.create_bucket(BUCKET_NAME, location=REGION)
    print(f"✓ Created new bucket: {BUCKET_NAME}")

# =====================================================
# UPLOAD TRAINING DATA TO GCS
# =====================================================
print("\n" + "=" * 80)
print("STEP 3: Uploading Training Data")
print("-" * 80)

local_file = "processed_telco_data.csv"
gcs_file = f"gs://{BUCKET_NAME}/data/{local_file}"

if os.path.exists(local_file):
    blob = bucket.blob(f"data/{local_file}")
    blob.upload_from_filename(local_file)
    print(f"✓ Uploaded training data to: {gcs_file}")
else:
    print(f"⚠️  File not found: {local_file}")
    print("   Please ensure processed_telco_data.csv is in the current directory")

# =====================================================
# CREATE DATASET IN VERTEX AI
# =====================================================
print("\n" + "=" * 80)
print("STEP 4: Creating Vertex AI Dataset")
print("-" * 80)

dataset_display_name = f"{DISPLAY_NAME}-dataset-{datetime.now().strftime('%Y%m%d%H%M%S')}"

# Create dataset from CSV
try:
    dataset = aiplatform.TabularDataset.create(
        display_name=dataset_display_name,
        gcs_source=gcs_file,
    )
    print(f"✓ Created Vertex AI dataset: {dataset_display_name}")
    print(f"   Resource Name: {dataset.resource_name}")
    
    dataset.wait()
    print("✓ Dataset import completed")
except Exception as e:
    print(f"⚠️  Error creating dataset: {e}")

# =====================================================
# TRAIN AUTOML MODEL
# =====================================================
print("\n" + "=" * 80)
print("STEP 5: Training AutoML Classification Model")
print(f"-" * 80)

model_display_name = f"{DISPLAY_NAME}-model-{datetime.now().strftime('%Y%m%d%H%M%S')}"

print(f"\n🤖 Model Configuration:")
print(f"   • Display Name: {model_display_name}")
print(f"   • Problem Type: Classification")
print(f"   • Target Column: Churn")
print(f"   • Training Budget: 1 hour")
print(f"\n   Training may take 30-60 minutes...")
print(f"   You can monitor progress in Vertex AI Console")

try:
    job = aiplatform.AutoMLTabularTrainingJob(
        display_name=f"train-{DISPLAY_NAME}",
        objective="minimize-log-loss",  # For binary classification
        budget_milli_node_hours=3600,  # 1 hour training
    )
    
    model = job.run(
        dataset=dataset,
        target_column="Churn",
        model_display_name=model_display_name,
        training_fraction_split=0.8,
        validation_fraction_split=0.1,
        test_fraction_split=0.1,
        enable_early_stopping=True,
    )
    
    print(f"\n✅ Model Training Completed!")
    print(f"   Model Name: {model_display_name}")
    print(f"   Model Resource: {model.resource_name}")
    
except Exception as e:
    print(f"⚠️  Error during training: {e}")
    model = None

# =====================================================
# EVALUATE MODEL PERFORMANCE
# =====================================================
if model:
    print("\n" + "=" * 80)
    print("STEP 6: Model Evaluation")
    print("-" * 80)
    
    try:
        # Get evaluation metrics
        evaluations = model.list_model_evaluations()
        
        print(f"\n📊 Model Evaluation Results:")
        
        metrics_list = []
        
        for evaluation in evaluations:
            print(f"\n   Evaluation ID: {evaluation.name.split('/')[-1]}")
            
            metrics_dict = {}
            
            if hasattr(evaluation, 'metrics'):
                for key, value in evaluation.metrics.items():
                    print(f"   • {key}: {value:.4f}" if isinstance(value, float) else f"   • {key}: {value}")
                    metrics_dict[key] = value
            
            metrics_list.append(metrics_dict)
        
        # Save evaluation results
        results_df = pd.DataFrame(metrics_list)
        results_csv = f"vertex_ai_model_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        results_df.to_csv(results_csv, index=False)
        print(f"\n✓ Results saved to: {results_csv}")
        
    except Exception as e:
        print(f"⚠️  Could not retrieve evaluations: {e}")

# =====================================================
# DEPLOY MODEL FOR PREDICTIONS
# =====================================================
if model:
    print("\n" + "=" * 80)
    print("STEP 7: Deploying Model (Optional)")
    print("-" * 80)
    
    deploy_choice = input("\n🚀 Deploy model for predictions? (y/n): ").strip().lower()
    
    if deploy_choice == 'y':
        try:
            endpoint_display_name = f"{DISPLAY_NAME}-endpoint-{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            endpoint = model.deploy(
                machine_type="n1-standard-2",
                deployed_model_display_name=endpoint_display_name,
            )
            
            print(f"✅ Model deployed successfully!")
            print(f"   Endpoint: {endpoint.resource_name}")
            print(f"   Predictions endpoint: {endpoint.predict_sampleson_uri}")
            
            # Save endpoint info for later use
            endpoint_info = {
                "endpoint_name": endpoint.resource_name,
                "endpoint_id": endpoint.name,
                "project": PROJECT_ID,
                "region": REGION,
                "timestamp": datetime.now().isoformat()
            }
            
            with open("vertex_ai_endpoint_info.json", "w") as f:
                json.dump(endpoint_info, f, indent=2)
            
            print(f"   ✓ Endpoint info saved to: vertex_ai_endpoint_info.json")
            
        except Exception as e:
            print(f"⚠️  Error deploying model: {e}")

# =====================================================
# EXPORT MODEL PERFORMANCE SUMMARY
# =====================================================
print("\n" + "=" * 80)
print("STEP 8: Generating Performance Summary")
print("-" * 80)

summary = {
    "Platform": "Google Cloud Vertex AI",
    "Model Type": "AutoML Tabular",
    "Training Date": datetime.now().isoformat(),
    "Project ID": PROJECT_ID,
    "Region": REGION,
    "Status": "Training Completed" if model else "Training Failed",
    "Model Resource": model.resource_name if model else "N/A",
}

summary_df = pd.DataFrame([summary])
summary_csv = f"vertex_ai_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
summary_df.to_csv(summary_csv, index=False)

print(f"\n✓ Summary saved to: {summary_csv}")
print(f"\n📋 Summary:")
for key, value in summary.items():
    print(f"   • {key}: {value}")

# =====================================================
# NEXT STEPS
# =====================================================
print("\n" + "=" * 80)
print("✅ VERTEX AI AUTOML SETUP COMPLETE!")
print("=" * 80)

print("""
📝 NEXT STEPS:

1. Monitor training progress:
   • Go to Vertex AI > Training > Custom Jobs in GCP Console
   • Training typically takes 30-60 minutes

2. View model evaluation:
   • Navigate to Vertex AI > Models
   • Select your model to see detailed metrics

3. Make predictions:
   • Use the endpoint for batch or real-time predictions
   • See predict_telco_churn.py for example usage

4. Export for Tableau:
   • Download evaluation metrics from GCP Console
   • Combine with Azure AutoML results
   • Create comprehensive comparison dashboard

📊 Files Generated:
   • vertex_ai_model_results_[timestamp].csv
   • vertex_ai_summary_[timestamp].csv
   • vertex_ai_endpoint_info.json (if deployed)

💡 IMPORTANT NOTES:
   • Monitor GCP billing - AutoML training incurs costs
   • Keep your credentials secure
   • Test model predictions before production use
""")

print("=" * 80)
