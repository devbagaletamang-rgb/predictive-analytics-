"""
=====================================================
AZURE AUTOML - TELCO CHURN PREDICTION
=====================================================
Automated Machine Learning on Microsoft Azure

Features:
- AutoML automatic algorithm selection and tuning
- Distributed training on Azure Compute
- Built-in ensemble methods
- Real-time prediction endpoints
- MLflow integration
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
import json
import logging
from azureml.core import Workspace, Dataset, Environment, ScriptRunConfig
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.train.automl import AutoMLConfig
from azureml.train.automl.run import AutoMLRun
from azure.storage.blob import BlobServiceClient

# =====================================================
# CONFIGURATION
# =====================================================
SUBSCRIPTION_ID = os.getenv("AZURE_SUBSCRIPTION_ID", "your-subscription-id")
RESOURCE_GROUP = os.getenv("AZURE_RESOURCE_GROUP", "telco-churn-rg")
WORKSPACE_NAME = os.getenv("AZURE_WORKSPACE_NAME", "telco-churn-workspace")
COMPUTE_NAME = "automl-cluster"
STORAGE_ACCOUNT = os.getenv("AZURE_STORAGE_ACCOUNT", "telcochurndata")
CONTAINER_NAME = "training-data"
LOCAL_FILE = "processed_telco_data.csv"

print("=" * 80)
print("AZURE AUTOML - TELCO CHURN PREDICTION")
print("=" * 80)
print(f"\n📊 Configuration:")
print(f"   • Subscription ID: {SUBSCRIPTION_ID[:20]}...")
print(f"   • Resource Group: {RESOURCE_GROUP}")
print(f"   • Workspace: {WORKSPACE_NAME}")
print(f"   • Compute Cluster: {COMPUTE_NAME}")

# =====================================================
# INITIALIZE LOGGING
# =====================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =====================================================
# STEP 1: CONNECT TO AZURE ML WORKSPACE
# =====================================================
print("\n" + "=" * 80)
print("STEP 1: Connecting to Azure ML Workspace")
print("-" * 80)

try:
    ws = Workspace(
        subscription_id=SUBSCRIPTION_ID,
        resource_group=RESOURCE_GROUP,
        workspace_name=WORKSPACE_NAME
    )
    print(f"✓ Connected to workspace: {ws.name}")
    print(f"   Location: {ws.location}")
    print(f"   Resource Group: {ws.resource_group}")
except Exception as e:
    logger.error(f"Failed to connect to workspace: {e}")
    print("⚠️  Could not connect to workspace")
    print("   Ensure you have configured Azure ML workspace")
    exit(1)

# =====================================================
# STEP 2: SETUP COMPUTE CLUSTER
# =====================================================
print("\n" + "=" * 80)
print("STEP 2: Setting up Compute Cluster")
print("-" * 80)

try:
    compute_target = ComputeTarget(workspace=ws, name=COMPUTE_NAME)
    print(f"✓ Using existing compute cluster: {COMPUTE_NAME}")
except ComputeTargetException:
    print(f"   Creating new compute cluster: {COMPUTE_NAME}")
    
    compute_config = AmlCompute.provisioning_configuration(
        vm_size="STANDARD_D2_V2",
        min_nodes=0,
        max_nodes=4,
        idle_seconds_before_scaledown=300
    )
    
    compute_target = ComputeTarget.create(ws, COMPUTE_NAME, compute_config)
    compute_target.wait_for_completion(show_output=True)
    print(f"✓ Created compute cluster: {COMPUTE_NAME}")

# =====================================================
# STEP 3: UPLOAD TRAINING DATA
# =====================================================
print("\n" + "=" * 80)
print("STEP 3: Uploading Training Data")
print("-" * 80)

if not os.path.exists(LOCAL_FILE):
    print(f"⚠️  File not found: {LOCAL_FILE}")
    print("   Please ensure processed_telco_data.csv is in the current directory")
else:
    try:
        # Upload to Azure blob storage
        try:
            blob_service_client = BlobServiceClient.from_connection_string(
                os.getenv("AZURE_STORAGE_CONNECTION_STRING")
            )
            container_client = blob_service_client.get_container_client(CONTAINER_NAME)
            
            with open(LOCAL_FILE, "rb") as data:
                container_client.upload_blob(LOCAL_FILE, data, overwrite=True)
            
            print(f"✓ Uploaded to blob storage: {LOCAL_FILE}")
        except Exception as e:
            logger.warning(f"Could not upload to blob storage: {e}")
        
        # Create dataset in Azure ML
        datastore = ws.get_default_datastore()
        dataset = Dataset.Tabular.from_delimited_files(
            path=(datastore, LOCAL_FILE)
        )
        dataset.register(workspace=ws, name="telco-churn-data", create_new_version=True)
        print(f"✓ Registered dataset: telco-churn-data")
        
    except Exception as e:
        logger.error(f"Error uploading data: {e}")

# =====================================================
# STEP 4: CONFIGURE AUTOML
# =====================================================
print("\n" + "=" * 80)
print("STEP 4: Configuring AutoML")
print("-" * 80)

# Get registered dataset
dataset = Dataset.get_by_name(ws, name="telco-churn-data")

automl_config = AutoMLConfig(
    compute_target=compute_target,
    task="classification",
    primary_metric="AUC_weighted",
    experiment_name="telco-churn-prediction",
    training_data=dataset,
    label_column_name="Churn",
    validation_size=0.2,
    test_size=0.2,
    n_cross_validations=5,
    max_cores_per_iteration=-1,
    max_concurrent_iterations=4,
    iteration_timeout_minutes=30,
    max_iterations=10,
    enable_early_stopping=True,
    featurization="auto",
    enable_voting_ensemble=True,
    enable_stack_ensemble=False,
    ensemble_models=["LightGBM", "XGBoost", "LogisticRegression"],
    blocked_models=["KNN", "LinearSVM"],
    model_explainability=True,
    verbosity=logging.INFO,
)

print("✓ AutoML configuration created")
print(f"   • Task: Classification")
print(f"   • Primary Metric: AUC_weighted")
print(f"   • Max Iterations: 10")
print(f"   • Cross Validations: 5")
print(f"   • Early Stopping: Enabled")

# =====================================================
# STEP 5: SUBMIT AUTOML RUN
# =====================================================
print("\n" + "=" * 80)
print("STEP 5: Submitting AutoML Training Job")
print("-" * 80)

print("\n🤖 Starting AutoML training...")
print("   This will automatically:")
print("   • Test multiple algorithms")
print("   • Tune hyperparameters")
print("   • Perform cross-validation")
print("   • Select best model")
print("\n   Training typically takes 20-40 minutes...")
print("   Monitor progress in Azure ML Studio\n")

try:
    run = ws.experiments["telco-churn-prediction"].submit(
        automl_config, show_output=False
    )
    
    print(f"✅ AutoML job submitted!")
    print(f"   Run ID: {run.id}")
    print(f"   Experiment: {run.experiment.name}")
    
    # Wait for run to complete
    run.wait_for_completion(show_output=False)
    
    print(f"\n✅ AutoML training completed!")
    
except Exception as e:
    logger.error(f"Error submitting AutoML run: {e}")
    print(f"⚠️  Error during AutoML submission: {e}")

# =====================================================
# STEP 6: RETRIEVE BEST MODEL
# =====================================================
print("\n" + "=" * 80)
print("STEP 6: Retrieving Best Model")
print("-" * 80)

try:
    best_run, fitted_model = run.get_output()
    
    print(f"✓ Best Model Retrieved")
    print(f"   Algorithm: {best_run.properties.get('model_summary')}")
    print(f"   Run ID: {best_run.id}")
    
except Exception as e:
    logger.error(f"Error retrieving best model: {e}")

# =====================================================
# STEP 7: EVALUATE BEST MODEL
# =====================================================
print("\n" + "=" * 80)
print("STEP 7: Model Evaluation")
print("-" * 80)

try:
    # Get metrics from best run
    metrics = best_run.get_metrics()
    
    print(f"\n📊 Best Model Performance Metrics:")
    
    metrics_dict = {}
    for metric_name, metric_value in metrics.items():
        if isinstance(metric_value, (int, float)):
            print(f"   • {metric_name}: {metric_value:.4f}")
            metrics_dict[metric_name] = metric_value
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame([metrics_dict])
    results_csv = f"azure_automl_model_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    metrics_df.to_csv(results_csv, index=False)
    print(f"\n✓ Results saved to: {results_csv}")
    
except Exception as e:
    logger.error(f"Error retrieving metrics: {e}")

# =====================================================
# STEP 8: REGISTER MODEL
# =====================================================
print("\n" + "=" * 80)
print("STEP 8: Registering Model")
print("-" * 80)

try:
    model_name = f"telco-churn-automl-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    model = best_run.register_model(
        model_name=model_name,
        model_path="outputs/model.pkl",
        tags={
            "platform": "Azure AutoML",
            "task": "churn_prediction",
            "timestamp": datetime.now().isoformat(),
        }
    )
    
    print(f"✓ Model registered: {model_name}")
    print(f"   Model ID: {model.id}")
    print(f"   Version: {model.version}")
    
except Exception as e:
    logger.error(f"Error registering model: {e}")

# =====================================================
# STEP 9: GENERATE PREDICTIONS ON TEST SET
# =====================================================
print("\n" + "=" * 80)
print("STEP 9: Generating Test Predictions")
print("-" * 80)

try:
    # Get test set predictions
    test_predictions = best_run.get_metrics().get("test_predictions")
    
    if test_predictions:
        predictions_df = pd.DataFrame(test_predictions)
        predictions_csv = f"azure_automl_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        predictions_df.to_csv(predictions_csv, index=False)
        print(f"✓ Predictions saved to: {predictions_csv}")
    
except Exception as e:
    logger.warning(f"Could not retrieve test predictions: {e}")

# =====================================================
# STEP 10: GENERATE SUMMARY REPORT
# =====================================================
print("\n" + "=" * 80)
print("STEP 10: Generating Summary Report")
print("-" * 80)

summary = {
    "Platform": "Microsoft Azure AutoML",
    "Training Date": datetime.now().isoformat(),
    "Subscription ID": SUBSCRIPTION_ID[:20] + "...",
    "Workspace": WORKSPACE_NAME,
    "Resource Group": RESOURCE_GROUP,
    "Compute Cluster": COMPUTE_NAME,
    "Experiment Name": "telco-churn-prediction",
    "Status": "Completed" if run.get_status() == "Completed" else run.get_status(),
}

summary_df = pd.DataFrame([summary])
summary_csv = f"azure_automl_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
summary_df.to_csv(summary_csv, index=False)

print(f"\n✓ Summary saved to: {summary_csv}")
print(f"\n📋 Summary:")
for key, value in summary.items():
    print(f"   • {key}: {value}")

# =====================================================
# STEP 11: DEPLOYMENT (OPTIONAL)
# =====================================================
print("\n" + "=" * 80)
print("STEP 11: Model Deployment (Optional)")
print("-" * 80)

deploy_choice = input("\n🚀 Deploy model to real-time endpoint? (y/n): ").strip().lower()

if deploy_choice == 'y':
    try:
        from azureml.core.webservice import AciWebservice, Webservice
        from azureml.core.model import InferenceConfig
        
        inference_config = InferenceConfig(
            environment=Environment.get_default(),
            entry_script="score.py"
        )
        
        service_name = f"telco-churn-endpoint-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        service = Model.deploy(
            ws,
            service_name,
            [model],
            inference_config,
            deployment_config=AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1)
        )
        
        service.wait_for_deployment(show_output=True)
        
        print(f"✅ Model deployed successfully!")
        print(f"   Service Name: {service_name}")
        print(f"   Scoring URI: {service.scoring_uri}")
        
        # Save endpoint info
        endpoint_info = {
            "service_name": service_name,
            "scoring_uri": service.scoring_uri,
            "swagger_uri": service.swagger_uri if hasattr(service, 'swagger_uri') else "N/A",
            "timestamp": datetime.now().isoformat()
        }
        
        with open("azure_automl_endpoint_info.json", "w") as f:
            json.dump(endpoint_info, f, indent=2)
        
        print(f"   ✓ Endpoint info saved to: azure_automl_endpoint_info.json")
        
    except Exception as e:
        logger.error(f"Error deploying model: {e}")
        print(f"⚠️  Error deploying model: {e}")

# =====================================================
# COMPLETION MESSAGE
# =====================================================
print("\n" + "=" * 80)
print("✅ AZURE AUTOML SETUP COMPLETE!")
print("=" * 80)

print("""
📝 NEXT STEPS:

1. Monitor training in Azure ML Studio:
   • Go to Azure ML Studio
   • Navigate to Experiments > telco-churn-prediction
   • View detailed metrics and model comparison

2. Review best model details:
   • Access model registry in Azure ML Studio
   • View model lineage and metrics
   • Download model artifacts if needed

3. Export metrics for Tableau:
   • Download CSV files from Azure ML
   • Combine with Vertex AI results
   • Prepare for comprehensive dashboard

4. Compare with Vertex AI results:
   • Combine azure_automl_model_results_*.csv
   • Compare performance metrics
   • Identify best platform for your use case

📊 Files Generated:
   • azure_automl_model_results_[timestamp].csv
   • azure_automl_summary_[timestamp].csv
   • azure_automl_predictions_[timestamp].csv (optional)
   • azure_automl_endpoint_info.json (if deployed)

💡 IMPORTANT NOTES:
   • Monitor Azure costs - AutoML training incurs charges
   • Keep credentials secure in environment variables
   • Test predictions before production deployment
   • Consider scaling settings for your use case
""")

print("=" * 80)
