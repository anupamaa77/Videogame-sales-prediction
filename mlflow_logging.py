import mlflow
import mlflow.sklearn
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import redis
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score

# 1. Load DataFrame from Redis
def load_from_redis(redis_key, host='localhost', port=6379, db=0):
    client = redis.Redis(host=host, port=port, db=db)
    data = client.get(redis_key)
    if data is None:
        raise ValueError(f"Key '{redis_key}' not found in Redis.")
    buffer = pa.py_buffer(data)
    reader = pa.ipc.open_stream(buffer)
    table = reader.read_all()
    return table.to_pandas()

# 2. Load Model from MLflow URI (optional utility)
def load_trained_model_from_mlflow(model_uri):
    return mlflow.sklearn.load_model(model_uri)

# 3. Load Local Model
def load_local_model(model_path):
    return joblib.load(model_path)

# 4. Log Feature Importance Plot
def log_feature_importance_plot(model, X_train, plot_path="feature_importance_plot.png"):
    feature_importances = model.feature_importances_
    plt.barh(X_train.columns, feature_importances)
    plt.xlabel('Feature Importance')
    plt.ylabel('Features')
    plt.title('Random Forest Feature Importance')
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    mlflow.log_artifact(plot_path)

# 5. Log Model and Metrics to MLflow
def log_model_to_mlflow(model, X_train, metrics: dict, log_plot=True):
    with mlflow.start_run() as run:
        mlflow.sklearn.log_model(model, "random_forest_model")

        mlflow.log_params({
            "model_type": "RandomForestRegressor",
            "n_features": X_train.shape[1],
        })

        for name, value in metrics.items():
            mlflow.log_metric(name, value)

        if log_plot:
            log_feature_importance_plot(model, X_train)

        print(f"✅ Logged to MLflow | Run ID: {run.info.run_id}")
        return run.info.run_id


def mlflow_log_pipeline(model_path, train_df, metrics_dict, drop_columns=None, log_plot=True):
    if drop_columns is None:
        drop_columns = ['Global_Sales', 'Name', 'Publisher']
    X_train = train_df.drop(columns=drop_columns, errors='ignore')
    model = load_local_model(model_path)
    run_id = log_model_to_mlflow(model, X_train, metrics_dict, log_plot=log_plot)
    return run_id


def run_mlflow_logging_task():

    train_df = load_from_redis("vgsales_cleaned")
    training_metrics = {
        "MSE": 27979.70885583681,
        "RMSE": 167.27136292813785,
        "MAE": 21.3980428578086,
        "R2": 0.9080988813949363,
        "Explained Variance": 0.9081467306810215,
    }
    mlflow_log_pipeline("random_forest_model.joblib", train_df, training_metrics)

