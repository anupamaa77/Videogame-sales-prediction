import mlflow.sklearn
import numpy as np
import pandas as pd
import pyarrow as pa
import redis
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
from sklearn.preprocessing import LabelEncoder
import joblib

platform_encoder = LabelEncoder()
genre_encoder = LabelEncoder()


def load_from_redis(redis_key):
    client = redis.Redis(host='localhost', port=6379, db=0)
    data = client.get(redis_key)
    buffer = pa.py_buffer(data)
    reader = pa.ipc.open_stream(buffer)
    table = reader.read_all()
    return table.to_pandas()

def preprocess_data(train_df, test_df):
    numerical_columns = train_df.select_dtypes(include='number').columns
    train_df[numerical_columns] = train_df[numerical_columns].fillna(train_df[numerical_columns].median())
    test_df[numerical_columns] = test_df[numerical_columns].fillna(test_df[numerical_columns].median())

    train_noise = np.random.normal(0, 10, train_df[numerical_columns].shape)
    test_noise = np.random.normal(0, 10, test_df[numerical_columns].shape)
    train_df[numerical_columns] += train_noise
    test_df[numerical_columns] += test_noise

    # Encode categorical features with shared encoders
    global platform_encoder, genre_encoder

    train_df['Platform_Encoded'] = platform_encoder.fit_transform(train_df['Platform'].astype(str))
    test_df['Platform_Encoded'] = test_df['Platform'].astype(str).apply(lambda x: x if x in platform_encoder.classes_ else 'UNKNOWN')
    platform_encoder_classes = list(platform_encoder.classes_)
    if 'UNKNOWN' not in platform_encoder_classes:
        platform_encoder_classes.append('UNKNOWN')
        platform_encoder.classes_ = np.array(platform_encoder_classes)
    test_df['Platform_Encoded'] = platform_encoder.transform(test_df['Platform_Encoded'])

    train_df['Genre_Encoded'] = genre_encoder.fit_transform(train_df['Genre'].astype(str))
    test_df['Genre_Encoded'] = test_df['Genre'].astype(str).apply(lambda x: x if x in genre_encoder.classes_ else 'UNKNOWN')
    genre_encoder_classes = list(genre_encoder.classes_)
    if 'UNKNOWN' not in genre_encoder_classes:
        genre_encoder_classes.append('UNKNOWN')
        genre_encoder.classes_ = np.array(genre_encoder_classes)
    test_df['Genre_Encoded'] = genre_encoder.transform(test_df['Genre_Encoded'])

    return train_df, test_df


def get_features_and_target(train_df, test_df):
    X_train = train_df.drop(columns=['Global_Sales', 'Name', 'Publisher'], errors='ignore')
    y_train = train_df['Global_Sales']
    X_test = test_df.drop(columns=['Global_Sales', 'Name', 'Publisher'], errors='ignore')
    y_test = test_df['Global_Sales']
    return X_train, y_train, X_test, y_test

def train_model(X_train, y_train):
    model = RandomForestRegressor(
        random_state=42,
        n_estimators=50,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=5
    )
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV score: {cv_scores.mean()}")
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    metrics = {
        "MSE": mean_squared_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "MAE": mean_absolute_error(y_test, y_pred),
        "R2": r2_score(y_test, y_pred),
        "Explained_Variance": explained_variance_score(y_test, y_pred)
    }
    for k, v in metrics.items():
        print(f"{k}: {v}")
    return metrics

def plot_feature_importance(model, feature_names_path="feature_names.txt", output_path="feature_importance_plot.png"):
    # Load feature names from file
    with open(feature_names_path, "r") as f:
        feature_names = [line.strip() for line in f.readlines()]

    importances = model.feature_importances_
    
    # Sort for better visualization (optional)
    sorted_indices = np.argsort(importances)
    feature_names = np.array(feature_names)[sorted_indices]
    importances = importances[sorted_indices]

    plt.figure(figsize=(10, 6))
    plt.barh(feature_names, importances)
    plt.xlabel('Feature Importance')
    plt.ylabel('Features')
    plt.title('Random Forest Feature Importance')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"✅ Feature importance plot saved at {output_path}")


def save_model(model, path='random_forest_model.joblib'):
    joblib.dump(model, path)
    print(f"✅ Model saved at {path}")

def run_full_training_pipeline():
    import mlflow
    import mlflow.sklearn

    # Set tracking URI
    mlflow.set_tracking_uri("http://localhost:5001")
    mlflow.set_experiment("vgsales_random_forest")

    with mlflow.start_run():
        train_df = load_from_redis("vgsales_train")
        test_df = load_from_redis("vgsales_test")
        print("✅ Data loaded from Redis")

        train_df, test_df = preprocess_data(train_df, test_df)
        X_train, y_train, X_test, y_test = get_features_and_target(train_df, test_df)

        # Save feature names to a text file
        feature_names = X_train.columns.tolist()
        with open('feature_names.txt', 'w') as f:
            for feature in feature_names:
                f.write(f"{feature}\n")

        print("✅ Feature names saved to 'feature_names.txt'")

        # Log parameter values
        mlflow.log_param("model_type", "RandomForestRegressor")
        mlflow.log_param("n_estimators", 50)
        mlflow.log_param("max_depth", 15)
        mlflow.log_param("min_samples_split", 5)
        mlflow.log_param("min_samples_leaf", 5)

        # Model training & cross-validation
        model = RandomForestRegressor(
            random_state=42,
            n_estimators=50,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=5
        )
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Mean CV score: {cv_scores.mean()}")

        mlflow.log_metric("cv_mean_score", cv_scores.mean())

        model.fit(X_train, y_train)

        # Evaluation
        metrics = evaluate_model(model, X_test, y_test)
        for k, v in metrics.items():
            mlflow.log_metric(k.lower(), v)

        # Plot and log feature importance
        plot_path = "feature_importance_plot.png"
        plot_feature_importance(model, feature_names_path="feature_names.txt", output_path=plot_path)

        mlflow.log_artifact(plot_path)

        # Save and log model
        model_path = "random_forest_model.joblib"
        joblib.dump(model, model_path)
        mlflow.sklearn.log_model(model, artifact_path="model")
        mlflow.log_artifact(model_path)
        joblib.dump(platform_encoder, "platform_encoder.joblib")
        joblib.dump(genre_encoder, "genre_encoder.joblib")
        print("✅ Encoders saved: platform_encoder.joblib, genre_encoder.joblib")

        mlflow.log_artifact("platform_encoder.joblib")
        mlflow.log_artifact("genre_encoder.joblib")


        print("✅ Training pipeline completed and logged with MLflow")

