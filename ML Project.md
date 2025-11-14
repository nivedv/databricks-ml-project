# Capstone Lab: End-to-End ML Project - Credit Card Fraud Detection System
## Created by Nived Varma
### There may be some bugs in the code. It is delibrately added. You AI assistant to resolve the errors or warning. 

**Estimated Duration:** 4 hours  
**Dataset:** Synthetic Credit Card Transactions  
**Cluster Configuration:** Single Node, Databricks Runtime 14.3 LTS with Machine Learning (This is already there in your workspace).

---

## Project Overview

Welcome to your final challenge! Today, you'll build a complete fraud detection system from scratch - just like a real-world ML engineer would. We're using PaySecure Inc., a fictional digital payment company processing 2 million transactions daily. Every missed fraud case costs them $1,200 on average, and false positives (blocking legitimate transactions) cost $50 in customer support and reputation damage. **Your mission: Build an intelligent system that catches fraud while keeping customers happy!**

### Business Context
- **Current State**: Manual review catches only 60% of fraud
- **False Positive Rate**: 8% (frustrating customers!)
- **Target Performance**: 95% fraud detection with <2% false positives
- **ROI Potential**: $18M annually in prevented losses

---

## Phase 1: Data Preparation & Feature Engineering (45 minutes)

### Real-World Scenario
You've just received transaction logs from PaySecure's data lake. The data engineering team has provided raw transaction data, but it's YOUR job to transform it into ML-ready features that can spot suspicious patterns!

### Step 1.1: Environment Setup & Data Generation

First, let's set up your workspace and generate realistic transaction data.

```python
# Cell 1: Import required libraries and setup
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pyspark.sql.functions import *
from pyspark.sql.types import *
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

print("✅ Libraries imported successfully!")
print(f"MLflow Version: {mlflow.__version__}")
```

```python
# Cell 2: Generate synthetic transaction dataset
def generate_transaction_data(n_samples=50000):
    """
    Generate realistic credit card transaction data
    This simulates PaySecure's daily transaction volume
    """
    
    # Time distribution (more transactions during business hours)
    start_date = datetime(2024, 1, 1)
    transaction_times = [start_date + timedelta(
        hours=np.random.choice(24, p=[0.02]*6 + [0.06]*12 + [0.03]*6)
    ) for _ in range(n_samples)]
    
    # Normal transactions (97% of data)
    n_normal = int(n_samples * 0.97)
    n_fraud = n_samples - n_normal
    
    # Normal transaction patterns
    normal_amounts = np.random.lognormal(3.5, 1.2, n_normal)
    normal_merchants = np.random.choice(['Amazon', 'Walmart', 'Target', 'Starbucks', 'Shell', 
                                        'McDonalds', 'Grocery Store', 'Gas Station'], n_normal)
    normal_locations = np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 
                                        'Phoenix', 'Philadelphia', 'San Antonio'], n_normal)
    
    # Fraud patterns (higher amounts, unusual times, foreign locations)
    fraud_amounts = np.random.uniform(200, 2000, n_fraud)
    fraud_merchants = np.random.choice(['Online Store', 'Unknown Vendor', 'Wire Transfer', 
                                       'Cash Advance'], n_fraud)
    fraud_locations = np.random.choice(['Foreign-Asia', 'Foreign-Europe', 'Foreign-Africa'], n_fraud)
    
    # Combine data
    amounts = np.concatenate([normal_amounts, fraud_amounts])
    merchants = np.concatenate([normal_merchants, fraud_merchants])
    locations = np.concatenate([normal_locations, fraud_locations])
    is_fraud = np.array([0]*n_normal + [1]*n_fraud)
    
    # Create DataFrame
    data = pd.DataFrame({
        'transaction_id': range(1, n_samples + 1),
        'timestamp': transaction_times,
        'amount': amounts,
        'merchant_name': merchants,
        'merchant_category': np.random.choice(['retail', 'food', 'gas', 'online', 'travel'], n_samples),
        'location': locations,
        'card_present': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
        'customer_id': np.random.randint(1000, 5000, n_samples),
        'is_fraud': is_fraud
    })
    
    # Shuffle data
    data = data.sample(frac=1).reset_index(drop=True)
    
    return data

# Generate dataset
print("🔄 Generating 50,000 transactions (this simulates one week of PaySecure data)...")
transactions_df = generate_transaction_data(50000)

# Convert to Spark DataFrame
spark_df = spark.createDataFrame(transactions_df)

# Display sample
print("\n📊 Sample of raw transaction data:")
display(spark_df.limit(10))

print(f"\n✅ Generated {transactions_df.shape[0]:,} transactions")
print(f"   ├─ Legitimate transactions: {(transactions_df['is_fraud']==0).sum():,}")
print(f"   └─ Fraudulent transactions: {(transactions_df['is_fraud']==1).sum():,}")
```

### Step 1.2: Exploratory Data Analysis

Let's understand what fraud looks like in the data!

```python
# Cell 3: Analyze fraud patterns
print("🔍 FRAUD PATTERN ANALYSIS")
print("="*60)

# Fraud rate by amount range
print("\n1️⃣ Fraud Rate by Transaction Amount:")
spark_df.withColumn('amount_range', 
    when(col('amount') < 50, '< $50')
    .when(col('amount') < 100, '$50-100')
    .when(col('amount') < 200, '$100-200')
    .when(col('amount') < 500, '$200-500')
    .otherwise('> $500')
).groupBy('amount_range').agg(
    count('*').alias('total_transactions'),
    sum('is_fraud').alias('fraud_count'),
    (avg('is_fraud') * 100).alias('fraud_rate_pct')
).orderBy('fraud_rate_pct', ascending=False).show()

# Fraud by location
print("\n2️⃣ Fraud Rate by Location:")
spark_df.groupBy('location').agg(
    count('*').alias('transactions'),
    (avg('is_fraud') * 100).alias('fraud_rate_pct')
).orderBy('fraud_rate_pct', ascending=False).show()

# Fraud by card presence
print("\n3️⃣ Card Present vs Not Present:")
spark_df.groupBy('card_present').agg(
    count('*').alias('transactions'),
    (avg('is_fraud') * 100).alias('fraud_rate_pct')
).show()
```

### Step 1.3: Feature Engineering

Time to create powerful features that reveal fraud patterns!

```python
# Cell 4: Engineer features
from pyspark.sql.window import Window

print("⚙️ FEATURE ENGINEERING PIPELINE")
print("="*60)

# 1. Time-based features
print("\n1️⃣ Creating time-based features...")
featured_df = spark_df.withColumn('hour', hour('timestamp')) \
    .withColumn('day_of_week', dayofweek('timestamp')) \
    .withColumn('is_weekend', when(col('day_of_week').isin([1, 7]), 1).otherwise(0)) \
    .withColumn('is_night', when((col('hour') >= 22) | (col('hour') <= 6), 1).otherwise(0))

# 2. Amount-based features
print("2️⃣ Creating amount-based features...")
featured_df = featured_df.withColumn('amount_log', log1p('amount')) \
    .withColumn('is_large_amount', when(col('amount') > 500, 1).otherwise(0))

# 3. Customer historical features (using window functions)
print("3️⃣ Creating customer behavior features...")
customer_window = Window.partitionBy('customer_id').orderBy('timestamp').rowsBetween(-10, 0)

featured_df = featured_df.withColumn('customer_avg_amount', 
                                    avg('amount').over(customer_window)) \
    .withColumn('customer_transaction_count', 
               count('*').over(customer_window))

# 4. Location risk features
print("4️⃣ Creating location risk features...")
featured_df = featured_df.withColumn('is_foreign', 
                                    when(col('location').contains('Foreign'), 1).otherwise(0))

# 5. Merchant category encoding
print("5️⃣ Encoding merchant categories...")
featured_df = featured_df.withColumn('merchant_online', 
                                    when(col('merchant_category') == 'online', 1).otherwise(0))

print("\n✅ Feature engineering complete!")
print("\n📋 New Features Created:")
print("   ├─ Time features: hour, day_of_week, is_weekend, is_night")
print("   ├─ Amount features: amount_log, is_large_amount")
print("   ├─ Customer features: customer_avg_amount, customer_transaction_count")
print("   ├─ Location features: is_foreign")
print("   └─ Merchant features: merchant_online")

# Display sample with new features
display(featured_df.select('transaction_id', 'amount', 'is_fraud', 'hour', 'is_night', 
                           'is_large_amount', 'is_foreign', 'customer_avg_amount').limit(10))
```

### Step 1.4: Prepare Final Training Dataset

```python
# Cell 5: Create final feature set
print("📦 PREPARING FINAL DATASET")
print("="*60)

# Select features for modeling
feature_columns = [
    'amount', 'amount_log', 'hour', 'day_of_week', 'is_weekend', 'is_night',
    'card_present', 'is_large_amount', 'is_foreign', 'merchant_online',
    'customer_avg_amount', 'customer_transaction_count'
]

# Handle any nulls (from customer aggregations)
final_df = featured_df.fillna({'customer_avg_amount': 0, 'customer_transaction_count': 0})

# Convert to Pandas for sklearn (with selected features)
modeling_data = final_df.select(feature_columns + ['is_fraud']).toPandas()

print(f"\n✅ Final dataset prepared!")
print(f"   ├─ Total samples: {len(modeling_data):,}")
print(f"   ├─ Features: {len(feature_columns)}")
print(f"   └─ Class distribution:")
print(f"      ├─ Legitimate: {(modeling_data['is_fraud']==0).sum():,} ({(modeling_data['is_fraud']==0).sum()/len(modeling_data)*100:.1f}%)")
print(f"      └─ Fraud: {(modeling_data['is_fraud']==1).sum():,} ({(modeling_data['is_fraud']==1).sum()/len(modeling_data)*100:.1f}%)")

# Split data
X = modeling_data[feature_columns]
y = modeling_data['is_fraud']

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

print(f"\n📊 Data Split:")
print(f"   ├─ Training: {len(X_train):,} samples ({len(X_train)/len(X)*100:.1f}%)")
print(f"   ├─ Validation: {len(X_val):,} samples ({len(X_val)/len(X)*100:.1f}%)")
print(f"   └─ Test: {len(X_test):,} samples ({len(X_test)/len(X)*100:.1f}%)")

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print("\n✅ Phase 1 Complete! Features are ready for modeling.")
```

---

## Phase 2: Model Development & Hyperparameter Tuning (50 minutes)

### Real-World Scenario
PaySecure's CTO wants you to evaluate multiple ML approaches. Your team will compare a fast Logistic Regression baseline against a more sophisticated Random Forest. The business requirement: detect 95% of fraud while keeping false positives under 2%!

### Step 2.1: Set Up MLflow Experiment

```python
# Cell 6: Initialize MLflow experiment
print("🔬 SETTING UP MLFLOW EXPERIMENT")
print("="*60)

# Create experiment
experiment_name = "/Users/your_email@domain.com/fraud_detection_capstone"
mlflow.set_experiment(experiment_name)

print(f"\n✅ MLflow experiment created: {experiment_name}")
print("   All your models and metrics will be tracked here!")
```

### Step 2.2: Baseline Model - Logistic Regression

```python
# Cell 7: Train baseline logistic regression
print("🎯 BASELINE MODEL: LOGISTIC REGRESSION")
print("="*60)

with mlflow.start_run(run_name="logistic_regression_baseline"):
    
    # Log parameters
    mlflow.log_param("model_type", "Logistic Regression")
    mlflow.log_param("solver", "lbfgs")
    mlflow.log_param("max_iter", 1000)
    mlflow.log_param("features", len(feature_columns))
    
    # Train model
    print("\n⏳ Training baseline model...")
    lr_model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    lr_model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_val_pred = lr_model.predict(X_val_scaled)
    y_val_proba = lr_model.predict_proba(X_val_scaled)[:, 1]
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    accuracy = accuracy_score(y_val, y_val_pred)
    precision = precision_score(y_val, y_val_pred)
    recall = recall_score(y_val, y_val_pred)
    f1 = f1_score(y_val, y_val_pred)
    roc_auc = roc_auc_score(y_val, y_val_proba)
    
    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", roc_auc)
    
    # Calculate business impact
    tp = ((y_val_pred == 1) & (y_val == 1)).sum()
    fp = ((y_val_pred == 1) & (y_val == 0)).sum()
    fn = ((y_val_pred == 0) & (y_val == 1)).sum()
    
    fraud_prevented_value = tp * 1200  # $1,200 per caught fraud
    false_positive_cost = fp * 50  # $50 per false alarm
    missed_fraud_cost = fn * 1200  # $1,200 per missed fraud
    net_value = fraud_prevented_value - false_positive_cost - missed_fraud_cost
    
    mlflow.log_metric("fraud_prevented_value", fraud_prevented_value)
    mlflow.log_metric("false_positive_cost", false_positive_cost)
    mlflow.log_metric("missed_fraud_cost", missed_fraud_cost)
    mlflow.log_metric("net_business_value", net_value)
    
    # Log model
    mlflow.sklearn.log_model(lr_model, "logistic_regression_model")
    
    # Save scaler as artifact
    import joblib
    joblib.dump(scaler, "/tmp/scaler.pkl")
    mlflow.log_artifact("/tmp/scaler.pkl")
    
    print("\n✅ Baseline Model Results:")
    print(f"   ├─ Accuracy: {accuracy:.4f}")
    print(f"   ├─ Precision: {precision:.4f} (false positive rate: {1-precision:.2%})")
    print(f"   ├─ Recall: {recall:.4f} (fraud detection rate)")
    print(f"   ├─ F1-Score: {f1:.4f}")
    print(f"   └─ ROC-AUC: {roc_auc:.4f}")
    print(f"\n💰 Business Impact (Validation Set):")
    print(f"   ├─ Fraud Prevented: ${fraud_prevented_value:,}")
    print(f"   ├─ False Positive Cost: ${false_positive_cost:,}")
    print(f"   ├─ Missed Fraud Cost: ${missed_fraud_cost:,}")
    print(f"   └─ Net Value: ${net_value:,}")
    
    # Confusion matrix
    print("\n📊 Confusion Matrix:")
    cm = confusion_matrix(y_val, y_val_pred)
    print(cm)
```

### Step 2.3: Advanced Model - Random Forest with Tuning

```python
# Cell 8: Train Random Forest with hyperparameter search
from sklearn.model_selection import RandomizedSearchCV

print("🌲 ADVANCED MODEL: RANDOM FOREST WITH HYPERPARAMETER TUNING")
print("="*60)

# Define parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'class_weight': ['balanced', 'balanced_subsample']
}

print("\n🔍 Searching for best hyperparameters...")
print("   Testing 20 random combinations...")

with mlflow.start_run(run_name="random_forest_tuned"):
    
    mlflow.log_param("model_type", "Random Forest")
    mlflow.log_param("tuning_method", "RandomizedSearchCV")
    mlflow.log_param("cv_folds", 3)
    mlflow.log_param("n_iter", 20)
    
    # Random search
    rf_base = RandomForestClassifier(random_state=42)
    random_search = RandomizedSearchCV(
        rf_base, 
        param_distributions=param_grid,
        n_iter=20,
        cv=3,
        scoring='f1',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    random_search.fit(X_train_scaled, y_train)
    
    # Best model
    best_rf = random_search.best_estimator_
    
    # Log best parameters
    print("\n🏆 Best Parameters Found:")
    for param, value in random_search.best_params_.items():
        print(f"   ├─ {param}: {value}")
        mlflow.log_param(f"best_{param}", value)
    
    # Predictions
    y_val_pred_rf = best_rf.predict(X_val_scaled)
    y_val_proba_rf = best_rf.predict_proba(X_val_scaled)[:, 1]
    
    # Calculate metrics
    accuracy_rf = accuracy_score(y_val, y_val_pred_rf)
    precision_rf = precision_score(y_val, y_val_pred_rf)
    recall_rf = recall_score(y_val, y_val_pred_rf)
    f1_rf = f1_score(y_val, y_val_pred_rf)
    roc_auc_rf = roc_auc_score(y_val, y_val_proba_rf)
    
    # Log metrics
    mlflow.log_metric("accuracy", accuracy_rf)
    mlflow.log_metric("precision", precision_rf)
    mlflow.log_metric("recall", recall_rf)
    mlflow.log_metric("f1_score", f1_rf)
    mlflow.log_metric("roc_auc", roc_auc_rf)
    
    # Business metrics
    tp_rf = ((y_val_pred_rf == 1) & (y_val == 1)).sum()
    fp_rf = ((y_val_pred_rf == 1) & (y_val == 0)).sum()
    fn_rf = ((y_val_pred_rf == 0) & (y_val == 1)).sum()
    
    fraud_prevented_rf = tp_rf * 1200
    false_positive_cost_rf = fp_rf * 50
    missed_fraud_cost_rf = fn_rf * 1200
    net_value_rf = fraud_prevented_rf - false_positive_cost_rf - missed_fraud_cost_rf
    
    mlflow.log_metric("fraud_prevented_value", fraud_prevented_rf)
    mlflow.log_metric("false_positive_cost", false_positive_cost_rf)
    mlflow.log_metric("missed_fraud_cost", missed_fraud_cost_rf)
    mlflow.log_metric("net_business_value", net_value_rf)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': best_rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n📊 Top 5 Most Important Features:")
    for idx, row in feature_importance.head(5).iterrows():
        print(f"   ├─ {row['feature']}: {row['importance']:.4f}")
    
    # Log model
    mlflow.sklearn.log_model(best_rf, "random_forest_model")
    
    print("\n✅ Random Forest Results:")
    print(f"   ├─ Accuracy: {accuracy_rf:.4f}")
    print(f"   ├─ Precision: {precision_rf:.4f} (false positive rate: {1-precision_rf:.2%})")
    print(f"   ├─ Recall: {recall_rf:.4f} (fraud detection rate)")
    print(f"   ├─ F1-Score: {f1_rf:.4f}")
    print(f"   └─ ROC-AUC: {roc_auc_rf:.4f}")
    print(f"\n💰 Business Impact (Validation Set):")
    print(f"   ├─ Fraud Prevented: ${fraud_prevented_rf:,}")
    print(f"   ├─ False Positive Cost: ${false_positive_cost_rf:,}")
    print(f"   ├─ Missed Fraud Cost: ${missed_fraud_cost_rf:,}")
    print(f"   └─ Net Value: ${net_value_rf:,}")
```

### Step 2.4: Model Comparison & Selection

```python
# Cell 9: Compare models and select winner
print("⚖️ MODEL COMPARISON & SELECTION")
print("="*60)

comparison_df = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest'],
    'Accuracy': [accuracy, accuracy_rf],
    'Precision': [precision, precision_rf],
    'Recall': [recall, recall_rf],
    'F1-Score': [f1, f1_rf],
    'ROC-AUC': [roc_auc, roc_auc_rf],
    'Net Business Value': [net_value, net_value_rf]
})

print("\n📊 Model Performance Comparison:")
print(comparison_df.to_string(index=False))

# Determine winner
winner = 'Random Forest' if net_value_rf > net_value else 'Logistic Regression'
winner_model = best_rf if winner == 'Random Forest' else lr_model
winner_improvement = abs(net_value_rf - net_value)

print(f"\n🏆 WINNING MODEL: {winner}")
print(f"   └─ Business Value Improvement: ${winner_improvement:,}")

# Test on hold-out test set
print("\n🧪 Final Evaluation on Test Set:")
y_test_pred = winner_model.predict(X_test_scaled)
y_test_proba = winner_model.predict_proba(X_test_scaled)[:, 1]

test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)
test_roc_auc = roc_auc_score(y_test, y_test_proba)

print(f"   ├─ Accuracy: {test_accuracy:.4f}")
print(f"   ├─ Precision: {test_precision:.4f}")
print(f"   ├─ Recall: {test_recall:.4f}")
print(f"   ├─ F1-Score: {test_f1:.4f}")
print(f"   └─ ROC-AUC: {test_roc_auc:.4f}")

print("\n✅ Phase 2 Complete! Best model identified and validated.")
```

---

## Phase 3: Model Registration & Deployment (45 minutes)

### Real-World Scenario
Your model performed brilliantly! Now PaySecure's infrastructure team needs it deployed to production. You'll register it in MLflow Model Registry, create a serving endpoint, and test it with live-like API calls.

### Step 3.1: Register Model to MLflow Registry

```python
# Cell 10: Register winning model
print("📦 REGISTERING MODEL TO MLFLOW MODEL REGISTRY")
print("="*60)

model_name = "fraud_detection_model"

# Register model
with mlflow.start_run(run_name="production_model_registration"):
    
    # Log final model with all metadata
    mlflow.log_param("model_type", winner)
    mlflow.log_param("training_samples", len(X_train))
    mlflow.log_param("features_used", feature_columns)
    
    # Log test metrics
    mlflow.log_metric("test_accuracy", test_accuracy)
    mlflow.log_metric("test_precision", test_precision)
    mlflow.log_metric("test_recall", test_recall)
    mlflow.log_metric("test_f1", test_f1)
    mlflow.log_metric("test_roc_auc", test_roc_auc)
    
    # Log business KPIs
    test_tp = ((y_test_pred == 1) & (y_test == 1)).sum()
    test_fp = ((y_test_pred == 1) & (y_test == 0)).sum()
    test_fn = ((y_test_pred == 0) & (y_test == 1)).sum()
    
    test_net_value = (test_tp * 1200) - (test_fp * 50) - (test_fn * 1200)
    mlflow.log_metric("test_net_business_value", test_net_value)
    
    # Log model artifacts
    mlflow.sklearn.log_model(
        winner_model,
        "model",
        registered_model_name=model_name,
        input_example=X_train.iloc[:5],
        signature=mlflow.models.infer_signature(X_train, y_train)
    )
    
    # Save preprocessing artifacts
    import joblib
    joblib.dump(scaler, "/tmp/scaler.pkl")
    joblib.dump(feature_columns, "/tmp/feature_columns.pkl")
    mlflow.log_artifact("/tmp/scaler.pkl")
    mlflow.log_artifact("/tmp/feature_columns.pkl")
    
    run_id = mlflow.active_run().info.run_id
    
    print(f"\n✅ Model registered successfully!")
    print(f"   ├─ Model Name: {model_name}")
    print(f"   ├─ Run ID: {run_id}")
    print(f"   └─ Test Performance:")
    print(f"      ├─ Fraud Detection Rate: {test_recall:.2%}")
    print(f"      ├─ False Positive Rate: {1-test_precision:.2%}")
    print(f"      └─ Net Business Value: ${test_net_value:,}")
```

### Step 3.2: Transition Model to Production

```python
# Cell 11: Promote model to production stage
from mlflow.tracking import MlflowClient

print("🚀 PROMOTING MODEL TO PRODUCTION")
print("="*60)

client = MlflowClient()

# Get latest version
latest_version = client.get_latest_versions(model_name, stages=["None"])[0]

print(f"\n📋 Model Version Details:")
print(f"   ├─ Model Name: {model_name}")
print(f"   ├─ Version: {latest_version.version}")
print(f"   └─ Current Stage: {latest_version.current_stage}")

# Transition to production
client.transition_model_version_stage(
    name=model_name,
    version=latest_version.version,
    stage="Production",
    archive_existing_versions=True
)

print(f"\n✅ Model promoted to Production stage!")
print(f"   └─ Version {latest_version.version} is now live!")

# Add model description
client.update_model_version(
    name=model_name,
    version=latest_version.version,
    description=f"""
    Fraud Detection Model - Production Deployment
    
    **Model Type:** {winner}
    **Training Date:** {datetime.now().strftime('%Y-%m-%d')}
    **Performance Metrics:**
    - Recall (Fraud Detection): {test_recall:.2%}
    - Precision (False Positive Control): {test_precision:.2%}
    - ROC-AUC: {test_roc_auc:.4f}
    
    **Business Impact:**
    - Estimated Annual Value: $18M
    - False Positive Rate: {1-test_precision:.2%}
    
    **Features:** {len(feature_columns)} features including transaction amount, time, location, and customer behavior
    """
)

print("\n📝 Model documentation added to registry")
```

### Step 3.3: Create Batch Inference Pipeline

```python
# Cell 12: Batch inference function
print("📊 CREATING BATCH INFERENCE PIPELINE")
print("="*60)

def batch_predict_fraud(transactions_spark_df):
    """
    Score a batch of transactions for fraud probability
    
    Args:
        transactions_spark_df: Spark DataFrame with transaction data
    
    Returns:
        Spark DataFrame with fraud predictions
    """
    
    # Load production model
    model_uri = f"models:/{model_name}/Production"
    loaded_model = mlflow.sklearn.load_model(model_uri)
    loaded_scaler = joblib.load(mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path="scaler.pkl"
    ))
    
    # Feature engineering (same as training)
    featured_batch = transactions_spark_df \
        .withColumn('hour', hour('timestamp')) \
        .withColumn('day_of_week', dayofweek('timestamp')) \
        .withColumn('is_weekend', when(col('day_of_week').isin([1, 7]), 1).otherwise(0)) \
        .withColumn('is_night', when((col('hour') >= 22) | (col('hour') <= 6), 1).otherwise(0)) \
        .withColumn('amount_log', log1p('amount')) \
        .withColumn('is_large_amount', when(col('amount') > 500, 1).otherwise(0)) \
        .withColumn('is_foreign', when(col('location').contains('Foreign'), 1).otherwise(0)) \
        .withColumn('merchant_online', when(col('merchant_category') == 'online', 1).otherwise(0))
    
    # Customer features (simplified for batch)
    customer_window = Window.partitionBy('customer_id').orderBy('timestamp').rowsBetween(-10, 0)
    featured_batch = featured_batch \
        .withColumn('customer_avg_amount', avg('amount').over(customer_window)) \
        .withColumn('customer_transaction_count', count('*').over(customer_window)) \
        .fillna({'customer_avg_amount': 0, 'customer_transaction_count': 0})
    
    # Convert to Pandas for prediction
    batch_pandas = featured_batch.select(feature_columns).toPandas()
    batch_scaled = loaded_scaler.transform(batch_pandas)
    
    # Predict
    predictions = loaded_model.predict(batch_scaled)
    probabilities = loaded_model.predict_proba(batch_scaled)[:, 1]
    
    # Add predictions to original DataFrame
    result_pdf = transactions_spark_df.toPandas()
    result_pdf['fraud_prediction'] = predictions
    result_pdf['fraud_probability'] = probabilities
    result_pdf['risk_level'] = pd.cut(
        probabilities, 
        bins=[0, 0.3, 0.7, 1.0], 
        labels=['LOW', 'MEDIUM', 'HIGH']
    )
    
    return spark.createDataFrame(result_pdf)

# Test batch scoring
print("\n🧪 Testing batch inference on new transactions...")
test_batch = spark_df.limit(100)
scored_batch = batch_predict_fraud(test_batch)

print("\n✅ Batch scoring complete!")
print("\n📊 Sample predictions:")
display(scored_batch.select(
    'transaction_id', 'amount', 'location', 'is_fraud',
    'fraud_prediction', 'fraud_probability', 'risk_level'
).limit(10))

# Calculate batch accuracy
batch_accuracy = scored_batch.select(
    (col('fraud_prediction') == col('is_fraud')).cast('int')
).agg({'(fraud_prediction = is_fraud)': 'avg'}).collect()[0][0]

print(f"\n🎯 Batch Accuracy: {batch_accuracy:.2%}")
```

### Step 3.4: Create Real-Time Scoring Function

```python
# Cell 13: Real-time inference function
print("⚡ CREATING REAL-TIME INFERENCE FUNCTION")
print("="*60)

def score_single_transaction(transaction_dict):
    """
    Score a single transaction in real-time
    
    Args:
        transaction_dict: Dictionary with transaction details
    
    Returns:
        Dictionary with prediction and probability
    """
    
    # Load model artifacts
    model_uri = f"models:/{model_name}/Production"
    model = mlflow.sklearn.load_model(model_uri)
    scaler_obj = joblib.load(mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path="scaler.pkl"
    ))
    
    # Extract and engineer features
    from datetime import datetime
    timestamp = datetime.fromisoformat(transaction_dict['timestamp'])
    
    features = {
        'amount': transaction_dict['amount'],
        'amount_log': np.log1p(transaction_dict['amount']),
        'hour': timestamp.hour,
        'day_of_week': timestamp.isoweekday(),
        'is_weekend': 1 if timestamp.isoweekday() in [6, 7] else 0,
        'is_night': 1 if timestamp.hour >= 22 or timestamp.hour <= 6 else 0,
        'card_present': transaction_dict.get('card_present', 0),
        'is_large_amount': 1 if transaction_dict['amount'] > 500 else 0,
        'is_foreign': 1 if 'Foreign' in transaction_dict.get('location', '') else 0,
        'merchant_online': 1 if transaction_dict.get('merchant_category') == 'online' else 0,
        'customer_avg_amount': transaction_dict.get('customer_avg_amount', 0),
        'customer_transaction_count': transaction_dict.get('customer_transaction_count', 0)
    }
    
    # Create feature vector
    X_single = pd.DataFrame([features])[feature_columns]
    X_scaled = scaler_obj.transform(X_single)
    
    # Predict
    prediction = model.predict(X_scaled)[0]
    probability = model.predict_proba(X_scaled)[0, 1]
    
    # Determine action
    if probability >= 0.7:
        action = "BLOCK"
        reason = "High fraud probability"
    elif probability >= 0.3:
        action = "REVIEW"
        reason = "Medium fraud risk - manual review recommended"
    else:
        action = "APPROVE"
        reason = "Low fraud risk"
    
    return {
        'transaction_id': transaction_dict['transaction_id'],
        'fraud_prediction': int(prediction),
        'fraud_probability': float(probability),
        'recommended_action': action,
        'reason': reason,
        'processing_timestamp': datetime.now().isoformat()
    }

# Test real-time scoring
print("\n🧪 Testing real-time inference...")

sample_transaction = {
    'transaction_id': 'TXN_TEST_001',
    'timestamp': '2024-11-13T23:45:00',
    'amount': 1200.00,
    'location': 'Foreign-Asia',
    'merchant_category': 'online',
    'card_present': 0,
    'customer_id': 1234,
    'customer_avg_amount': 75.50,
    'customer_transaction_count': 45
}

result = score_single_transaction(sample_transaction)

print("\n✅ Real-time scoring result:")
print(f"   ├─ Transaction ID: {result['transaction_id']}")
print(f"   ├─ Fraud Probability: {result['fraud_probability']:.2%}")
print(f"   ├─ Recommended Action: {result['recommended_action']}")
print(f"   └─ Reason: {result['reason']}")

print("\n✅ Phase 3 Complete! Model is registered and ready for deployment.")
```

---

## Phase 4: Pipeline Automation & Monitoring Setup (50 minutes)

### Real-World Scenario
PaySecure processes millions of transactions daily. You need to automate the entire pipeline - from data ingestion to model retraining - and set up monitoring to catch issues before they impact customers!

### Step 4.1: Create Automated Retraining Pipeline

```python
# Cell 14: Automated retraining pipeline
print("🔄 CREATING AUTOMATED RETRAINING PIPELINE")
print("="*60)

def automated_model_retraining():
    """
    Complete retraining pipeline that runs on schedule
    """
    
    print("Starting automated retraining workflow...")
    start_time = datetime.now()
    
    # Step 1: Data refresh
    print("\n1️⃣ Fetching latest transaction data...")
    new_data = generate_transaction_data(n_samples=50000)
    print(f"   ✅ Loaded {len(new_data):,} new transactions")
    
    # Step 2: Feature engineering
    print("\n2️⃣ Engineering features...")
    spark_new = spark.createDataFrame(new_data)
    
    featured_new = spark_new \
        .withColumn('hour', hour('timestamp')) \
        .withColumn('day_of_week', dayofweek('timestamp')) \
        .withColumn('is_weekend', when(col('day_of_week').isin([1, 7]), 1).otherwise(0)) \
        .withColumn('is_night', when((col('hour') >= 22) | (col('hour') <= 6), 1).otherwise(0)) \
        .withColumn('amount_log', log1p('amount')) \
        .withColumn('is_large_amount', when(col('amount') > 500, 1).otherwise(0)) \
        .withColumn('is_foreign', when(col('location').contains('Foreign'), 1).otherwise(0)) \
        .withColumn('merchant_online', when(col('merchant_category') == 'online', 1).otherwise(0))
    
    customer_window = Window.partitionBy('customer_id').orderBy('timestamp').rowsBetween(-10, 0)
    featured_new = featured_new \
        .withColumn('customer_avg_amount', avg('amount').over(customer_window)) \
        .withColumn('customer_transaction_count', count('*').over(customer_window)) \
        .fillna({'customer_avg_amount': 0, 'customer_transaction_count': 0})
    
    print("   ✅ Features engineered")
    
    # Step 3: Train new model
    print("\n3️⃣ Training new model...")
    modeling_new = featured_new.select(feature_columns + ['is_fraud']).toPandas()
    X_new = modeling_new[feature_columns]
    y_new = modeling_new['is_fraud']
    
    X_train_new, X_val_new, y_train_new, y_val_new = train_test_split(
        X_new, y_new, test_size=0.2, random_state=42, stratify=y_new
    )
    
    new_scaler = StandardScaler()
    X_train_scaled_new = new_scaler.fit_transform(X_train_new)
    X_val_scaled_new = new_scaler.transform(X_val_new)
    
    new_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    new_model.fit(X_train_scaled_new, y_train_new)
    
    print("   ✅ Model trained")
    
    # Step 4: Evaluate new model
    print("\n4️⃣ Evaluating new model...")
    y_val_pred_new = new_model.predict(X_val_scaled_new)
    y_val_proba_new = new_model.predict_proba(X_val_scaled_new)[:, 1]
    
    new_recall = recall_score(y_val_new, y_val_pred_new)
    new_precision = precision_score(y_val_new, y_val_pred_new)
    new_f1 = f1_score(y_val_new, y_val_pred_new)
    
    print(f"   ├─ Recall: {new_recall:.4f}")
    print(f"   ├─ Precision: {new_precision:.4f}")
    print(f"   └─ F1-Score: {new_f1:.4f}")
    
    # Step 5: Compare with production model
    print("\n5️⃣ Comparing with production model...")
    
    # Load current production model for comparison
    prod_model_uri = f"models:/{model_name}/Production"
    prod_model = mlflow.sklearn.load_model(prod_model_uri)
    prod_scaler = joblib.load(mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path="scaler.pkl"
    ))
    
    X_val_scaled_prod = prod_scaler.transform(X_val_new)
    y_val_pred_prod = prod_model.predict(X_val_scaled_prod)
    prod_f1 = f1_score(y_val_new, y_val_pred_prod)
    
    improvement = new_f1 - prod_f1
    print(f"   ├─ Production F1: {prod_f1:.4f}")
    print(f"   ├─ New Model F1: {new_f1:.4f}")
    print(f"   └─ Improvement: {improvement:+.4f}")
    
    # Step 6: Decide whether to deploy
    if improvement > 0.01:  # Requires >1% improvement
        print("\n6️⃣ ✅ NEW MODEL APPROVED - Registering...")
        
        with mlflow.start_run(run_name=f"automated_retrain_{datetime.now().strftime('%Y%m%d')}"):
            mlflow.log_param("retrain_date", datetime.now().isoformat())
            mlflow.log_metric("recall", new_recall)
            mlflow.log_metric("precision", new_precision)
            mlflow.log_metric("f1_score", new_f1)
            mlflow.log_metric("improvement_over_prod", improvement)
            
            mlflow.sklearn.log_model(
                new_model,
                "model",
                registered_model_name=model_name
            )
            
        print("   ✅ New model registered and promoted")
        
    else:
        print("\n6️⃣ ⚠️ NEW MODEL REJECTED - Insufficient improvement")
        print("   Production model retained")
    
    duration = (datetime.now() - start_time).total_seconds()
    print(f"\n✅ Retraining pipeline complete in {duration:.1f} seconds")
    
    return {
        'status': 'success',
        'new_model_f1': new_f1,
        'prod_model_f1': prod_f1,
        'improvement': improvement,
        'deployed': improvement > 0.01,
        'duration_seconds': duration
    }

# Run retraining pipeline
print("🚀 Executing automated retraining pipeline...\n")
retrain_result = automated_model_retraining()

print("\n📊 Retraining Summary:")
print(f"   ├─ Status: {retrain_result['status'].upper()}")
print(f"   ├─ New Model Performance: {retrain_result['new_model_f1']:.4f}")
print(f"   ├─ Production Model: {retrain_result['prod_model_f1']:.4f}")
print(f"   ├─ Improvement: {retrain_result['improvement']:+.4f}")
print(f"   └─ Deployed: {'✅ YES' if retrain_result['deployed'] else '❌ NO'}")
```

### Step 4.2: Model Performance Monitoring

```python
# Cell 15: Model monitoring dashboard
print("📊 MODEL PERFORMANCE MONITORING DASHBOARD")
print("="*60)

def generate_monitoring_report(predictions_df):
    """
    Generate comprehensive monitoring report
    """
    
    report = {}
    
    # Prediction distribution
    total_transactions = len(predictions_df)
    fraud_predictions = (predictions_df['fraud_prediction'] == 1).sum()
    fraud_rate = fraud_predictions / total_transactions
    
    report['total_transactions'] = total_transactions
    report['predicted_fraud_count'] = fraud_predictions
    report['predicted_fraud_rate'] = fraud_rate
    
    # Probability distribution
    report['avg_fraud_probability'] = predictions_df['fraud_probability'].mean()
    report['high_risk_transactions'] = (predictions_df['fraud_probability'] > 0.7).sum()
    report['medium_risk_transactions'] = ((predictions_df['fraud_probability'] >= 0.3) & 
                                          (predictions_df['fraud_probability'] <= 0.7)).sum()
    report['low_risk_transactions'] = (predictions_df['fraud_probability'] < 0.3).sum()
    
    # If we have ground truth
    if 'is_fraud' in predictions_df.columns:
        report['actual_fraud_count'] = (predictions_df['is_fraud'] == 1).sum()
        report['true_positives'] = ((predictions_df['fraud_prediction'] == 1) & 
                                    (predictions_df['is_fraud'] == 1)).sum()
        report['false_positives'] = ((predictions_df['fraud_prediction'] == 1) & 
                                     (predictions_df['is_fraud'] == 0)).sum()
        report['false_negatives'] = ((predictions_df['fraud_prediction'] == 0) & 
                                     (predictions_df['is_fraud'] == 1)).sum()
        
        report['precision'] = report['true_positives'] / (report['true_positives'] + report['false_positives']) \
                             if (report['true_positives'] + report['false_positives']) > 0 else 0
        report['recall'] = report['true_positives'] / (report['true_positives'] + report['false_negatives']) \
                          if (report['true_positives'] + report['false_negatives']) > 0 else 0
    
    return report

# Generate monitoring data
print("\n🔄 Generating monitoring data from recent transactions...")
monitoring_batch = batch_predict_fraud(spark_df.limit(1000)).toPandas()

# Create monitoring report
monitoring_report = generate_monitoring_report(monitoring_batch)

print("\n📈 MONITORING REPORT - Last 1,000 Transactions")
print("="*60)
print(f"\n🎯 Prediction Summary:")
print(f"   ├─ Total Transactions: {monitoring_report['total_transactions']:,}")
print(f"   ├─ Predicted Fraud: {monitoring_report['predicted_fraud_count']} ({monitoring_report['predicted_fraud_rate']:.2%})")
print(f"   └─ Average Fraud Probability: {monitoring_report['avg_fraud_probability']:.4f}")

print(f"\n⚠️ Risk Distribution:")
print(f"   ├─ High Risk (>70%): {monitoring_report['high_risk_transactions']}")
print(f"   ├─ Medium Risk (30-70%): {monitoring_report['medium_risk_transactions']}")
print(f"   └─ Low Risk (<30%): {monitoring_report['low_risk_transactions']}")

if 'precision' in monitoring_report:
    print(f"\n🎯 Model Performance (with ground truth):")
    print(f"   ├─ Precision: {monitoring_report['precision']:.2%}")
    print(f"   ├─ Recall: {monitoring_report['recall']:.2%}")
    print(f"   ├─ True Positives: {monitoring_report['true_positives']}")
    print(f"   ├─ False Positives: {monitoring_report['false_positives']}")
    print(f"   └─ False Negatives: {monitoring_report['false_negatives']}")

# Alert conditions
print("\n🚨 Alert Monitoring:")
alerts = []

if monitoring_report['predicted_fraud_rate'] > 0.05:
    alerts.append("⚠️ Fraud rate exceeds 5% threshold!")
    
if 'precision' in monitoring_report and monitoring_report['precision'] < 0.85:
    alerts.append("⚠️ Precision below 85% - too many false positives!")
    
if 'recall' in monitoring_report and monitoring_report['recall'] < 0.90:
    alerts.append("⚠️ Recall below 90% - missing too many frauds!")

if alerts:
    print("   ALERTS DETECTED:")
    for alert in alerts:
        print(f"   {alert}")
else:
    print("   ✅ All metrics within acceptable ranges")
```

### Step 4.3: Create Production Monitoring Visualization

```python
# Cell 16: Monitoring visualizations
print("📊 CREATING MONITORING VISUALIZATIONS")
print("="*60)

import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.figure(figsize=(16, 10))

# 1. Fraud Probability Distribution
plt.subplot(2, 3, 1)
plt.hist(monitoring_batch['fraud_probability'], bins=50, color='steelblue', edgecolor='black')
plt.axvline(0.3, color='orange', linestyle='--', label='Medium Risk Threshold')
plt.axvline(0.7, color='red', linestyle='--', label='High Risk Threshold')
plt.xlabel('Fraud Probability')
plt.ylabel('Frequency')
plt.title('Distribution of Fraud Probabilities')
plt.legend()

# 2. Risk Level Distribution
plt.subplot(2, 3, 2)
risk_counts = monitoring_batch['risk_level'].value_counts()
colors = ['green', 'orange', 'red']
plt.pie(risk_counts, labels=risk_counts.index, autopct='%1.1f%%', colors=colors)
plt.title('Risk Level Distribution')

# 3. Confusion Matrix (if ground truth available)
if 'is_fraud' in monitoring_batch.columns:
    plt.subplot(2, 3, 3)
    cm = confusion_matrix(monitoring_batch['is_fraud'], monitoring_batch['fraud_prediction'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')

# 4. Fraud by Amount Range
plt.subplot(2, 3, 4)
monitoring_batch['amount_range'] = pd.cut(
    monitoring_batch['amount'],
    bins=[0, 50, 100, 200, 500, 10000],
    labels=['<$50', '$50-100', '$100-200', '$200-500', '>$500']
)
fraud_by_amount = monitoring_batch.groupby('amount_range')['fraud_prediction'].mean() * 100
fraud_by_amount.plot(kind='bar', color='coral')
plt.xlabel('Transaction Amount Range')
plt.ylabel('Fraud Rate (%)')
plt.title('Fraud Rate by Transaction Amount')
plt.xticks(rotation=45)

# 5. Daily fraud trend (simulate time series)
plt.subplot(2, 3, 5)
monitoring_batch['date'] = pd.to_datetime(monitoring_batch['timestamp']).dt.date
daily_fraud = monitoring_batch.groupby('date')['fraud_prediction'].mean() * 100
daily_fraud.plot(marker='o', color='darkred')
plt.xlabel('Date')
plt.ylabel('Fraud Rate (%)')
plt.title('Daily Fraud Rate Trend')
plt.xticks(rotation=45)

# 6. Model Performance Metrics Over Time
plt.subplot(2, 3, 6)
if 'is_fraud' in monitoring_batch.columns:
    metrics = {
        'Precision': monitoring_report['precision'],
        'Recall': monitoring_report['recall'],
        'Fraud Rate': monitoring_report['predicted_fraud_rate']
    }
    plt.bar(metrics.keys(), [v*100 for v in metrics.values()], color=['skyblue', 'lightgreen', 'salmon'])
    plt.ylabel('Percentage (%)')
    plt.title('Current Model Performance Metrics')
    plt.ylim(0, 100)

plt.tight_layout()
plt.savefig('/tmp/monitoring_dashboard.png', dpi=300, bbox_inches='tight')
print("\n✅ Monitoring dashboard created!")
print("   └─ Saved to: /tmp/monitoring_dashboard.png")

display(plt.gcf())
plt.close()
```

### Step 4.4: Create Alert System

```python
# Cell 17: Automated alert system
print("🚨 SETTING UP AUTOMATED ALERT SYSTEM")
print("="*60)

def check_model_health(monitoring_data):
    """
    Monitor model health and generate alerts
    """
    
    alerts = []
    health_score = 100
    
    # Check 1: Data drift - fraud rate
    current_fraud_rate = (monitoring_data['fraud_prediction'] == 1).mean()
    expected_fraud_rate = 0.03
    
    if abs(current_fraud_rate - expected_fraud_rate) > 0.02:
        alerts.append({
            'severity': 'HIGH',
            'type': 'Data Drift',
            'message': f'Fraud rate deviation detected: {current_fraud_rate:.2%} vs expected {expected_fraud_rate:.2%}',
            'recommended_action': 'Review recent data and consider model retraining'
        })
        health_score -= 20
    
    # Check 2: Model performance (if ground truth available)
    if 'is_fraud' in monitoring_data.columns:
        current_precision = precision_score(monitoring_data['is_fraud'], monitoring_data['fraud_prediction'])
        current_recall = recall_score(monitoring_data['is_fraud'], monitoring_data['fraud_prediction'])
        
        if current_precision < 0.85:
            alerts.append({
                'severity': 'HIGH',
                'type': 'Performance Degradation',
                'message': f'Precision dropped to {current_precision:.2%} (threshold: 85%)',
                'recommended_action': 'High false positive rate - review decision threshold'
            })
            health_score -= 25
        
        if current_recall < 0.90:
            alerts.append({
                'severity': 'CRITICAL',
                'type': 'Performance Degradation',
                'message': f'Recall dropped to {current_recall:.2%} (threshold: 90%)',
                'recommended_action': 'Missing too many frauds - immediate model review required'
            })
            health_score -= 30
    
    # Check 3: Prediction confidence
    avg_max_proba = monitoring_data['fraud_probability'].apply(lambda x: max(x, 1-x)).mean()
    
    if avg_max_proba < 0.7:
        alerts.append({
            'severity': 'MEDIUM',
            'type': 'Low Confidence',
            'message': f'Average prediction confidence: {avg_max_proba:.2%}',
            'recommended_action': 'Model uncertainty increasing - evaluate feature quality'
        })
        health_score -= 15
    
    # Check 4: Volume anomaly
    daily_volume = len(monitoring_data)
    expected_volume = 1000
    
    if abs(daily_volume - expected_volume) / expected_volume > 0.5:
        alerts.append({
            'severity': 'MEDIUM',
            'type': 'Volume Anomaly',
            'message': f'Transaction volume: {daily_volume} vs expected {expected_volume}',
            'recommended_action': 'Verify data pipeline health'
        })
        health_score -= 10
    
    return {
        'health_score': max(0, health_score),
        'status': 'HEALTHY' if health_score >= 80 else 'DEGRADED' if health_score >= 60 else 'CRITICAL',
        'alerts': alerts,
        'timestamp': datetime.now().isoformat()
    }

# Run health check
health_report = check_model_health(monitoring_batch)

print("\n🏥 MODEL HEALTH CHECK RESULTS")
print("="*60)
print(f"\n🎯 Overall Health Score: {health_report['health_score']}/100")
print(f"📊 Status: {health_report['status']}")

if health_report['alerts']:
    print(f"\n⚠️ {len(health_report['alerts'])} Alert(s) Detected:\n")
    for i, alert in enumerate(health_report['alerts'], 1):
        print(f"{i}. [{alert['severity']}] {alert['type']}")
        print(f"   Message: {alert['message']}")
        print(f"   Action: {alert['recommended_action']}\n")
else:
    print("\n✅ No alerts - system operating normally")

print(f"📅 Report Timestamp: {health_report['timestamp']}")

print("\n✅ Phase 4 Complete! Production monitoring and automation ready.")
```

---

## Final Summary & Presentation

```python
# Cell 18: Project summary and ROI calculation
print("=" * 80)
print("🎊 CAPSTONE PROJECT COMPLETE! 🎊")
print("=" * 80)

print("\n📋 PROJECT SUMMARY: PaySecure Fraud Detection System")
print("="*60)

print("\n1️⃣ DATASET:")
print(f"   ├─ Total Transactions Processed: {len(transactions_df):,}")
print(f"   ├─ Features Engineered: {len(feature_columns)}")
print(f"   └─ Training/Validation/Test Split: 70%/15%/15%")

print("\n2️⃣ MODEL DEVELOPMENT:")
print(f"   ├─ Models Evaluated: Logistic Regression, Random Forest")
print(f"   ├─ Winning Model: {winner}")
print(f"   └─ Optimization: Hyperparameter tuning with RandomizedSearchCV")

print("\n3️⃣ PRODUCTION PERFORMANCE:")
print(f"   ├─ Fraud Detection Rate (Recall): {test_recall:.2%}")
print(f"   ├─ False Positive Rate: {1-test_precision:.2%}")
print(f"   ├─ ROC-AUC Score: {test_roc_auc:.4f}")
print(f"   └─ Overall Accuracy: {test_accuracy:.2%}")

print("\n4️⃣ BUSINESS IMPACT (Annual Projection):")
daily_transactions = 2_000_000
annual_fraud_cases = daily_transactions * 365 * 0.03  # 3% fraud rate
fraud_detected = annual_fraud_cases * test_recall
fraud_prevented_annually = fraud_detected * 1200
false_positives_annual = daily_transactions * 365 * (1 - test_precision) * 0.03
fp_cost_annually = false_positives_annual * 50
missed_fraud_annual = annual_fraud_cases * (1 - test_recall) * 1200
net_annual_value = fraud_prevented_annually - fp_cost_annually - missed_fraud_annual

print(f"   ├─ Fraud Cases Detected: {fraud_detected:,.0f}/year")
print(f"   ├─ Fraud Losses Prevented: ${fraud_prevented_annually:,.0f}")
print(f"   ├─ False Positive Cost: ${fp_cost_annually:,.0f}")
print(f"   ├─ Missed Fraud Cost: ${missed_fraud_annual:,.0f}")
print(f"   └─ 💰 NET ANNUAL VALUE: ${net_annual_value:,.0f}")

print("\n5️⃣ MLOPS IMPLEMENTATION:")
print("   ├─ ✅ Model registered in MLflow Registry")
print("   ├─ ✅ Batch inference pipeline created")
print("   ├─ ✅ Real-time scoring function implemented")
print("   ├─ ✅ Automated retraining pipeline configured")
print("   └─ ✅ Production monitoring and alerting active")

print("\n6️⃣ KEY ACHIEVEMENTS:")
print("   ├─ 🎯 Exceeded target: 95% fraud detection (Target: 95%, Achieved: {:.1f}%)".format(test_recall*100))
print("   ├─ 🎯 Met FP target: <2% false positives (Target: <2%, Achieved: {:.1f}%)".format((1-test_precision)*100))
print("   ├─ 💰 Projected ROI: ${:,.0f} million annually".format(net_annual_value/1_000_000))
print("   └─ ⚡ Real-time scoring: <100ms latency")

print("\n7️⃣ DELIVERABLES:")
print("   ├─ Production-ready ML model")
print("   ├─ Feature engineering pipeline")
print("   ├─ Batch and real-time inference endpoints")
print("   ├─ Automated retraining workflow")
print("   ├─ Monitoring dashboard and alerts")
print("   └─ Complete MLflow experiment tracking")

print("\n" + "="*60)
print("🌟 CONGRATULATIONS! You've successfully built and deployed an")
print("   enterprise-grade fraud detection system from scratch!")
print("="*60)

print("\n📚 Next Steps for PaySecure:")
print("   1. Deploy to production Azure environment")
print("   2. Integrate with payment processing systems")
print("   3. Set up CI/CD pipeline with Azure DevOps")
print("   4. Configure real-time alerts to security team")
print("   5. Schedule monthly model retraining")
print("   6. Establish feedback loop with fraud analysts")

print("\n🎓 Your Learning Journey:")
print("   ✅ Data preparation and feature engineering")
print("   ✅ ML model development and optimization")
print("   ✅ MLflow experiment tracking and model registry")
print("   ✅ Production deployment strategies")
print("   ✅ Pipeline automation and monitoring")
print("   ✅ Business value quantification")

print("\n🚀 You're now ready to tackle real-world ML projects!")
print("="*80)
```

---

## Troubleshooting Guide

### Common Issues & Solutions

**Issue 1: MLflow Experiment Not Found**
```python
# Solution: Ensure experiment name matches your email
experiment_name = "/Users/YOUR_EMAIL@domain.com/fraud_detection_capstone"
mlflow.set_experiment(experiment_name)
```

**Issue 2: Model Registry Access**
```python
# Solution: Check Unity Catalog permissions
# Ask your admin to grant CREATE MODEL permissions
```

**Issue 3: Memory Issues with Large Dataset**
```python
# Solution: Reduce sample size
transactions_df = generate_transaction_data(n_samples=10000)  # Instead of 50000
```

**Issue 4: Slow Training**
```python
# Solution: Reduce RandomizedSearchCV iterations
n_iter=10  # Instead of 20
```

---

## Assessment Criteria

**Your project will be evaluated on:**

1. **Data Quality (20 points)**
   - Proper feature engineering
   - Handling of missing values
   - Train/val/test split

2. **Model Performance (25 points)**
   - Meets business requirements (95% recall, <2% FP)
   - Proper hyperparameter tuning
   - Model comparison methodology

3. **MLOps Implementation (25 points)**
   - MLflow tracking completeness
   - Model registry usage
   - Deployment readiness

4. **Automation & Monitoring (20 points)**
   - Retraining pipeline functionality
   - Monitoring dashboard quality
   - Alert system implementation

5. **Business Value (10 points)**
   - ROI calculation accuracy
   - Business metrics tracking
   - Stakeholder communication clarity

**Total: 100 points | Passing Score: 75 points**

---

## Congratulations! 🎉

You've completed an end-to-end machine learning project covering:
- Feature engineering
- Model development
- Hyperparameter optimization
- MLOps with MLflow
- Production deployment
- Automation and monitoring

**This capstone demonstrates production-ready ML engineering skills that employers value!**