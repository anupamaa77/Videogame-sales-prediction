import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def post_validation_preprocessing(df_validated: pd.DataFrame) -> pd.DataFrame:
    """
    Perform post-validation preprocessing steps including feature engineering,
    encoding, and scaling, without splitting the dataset.
    
    Parameters:
        df_validated (pd.DataFrame): Cleaned and validated DataFrame.
    
    Returns:
        pd.DataFrame: Fully preprocessed DataFrame ready for modeling or storage (e.g., Redis).
    """
    df = df_validated.copy()

    # -----------------------------------------
    # 1. FEATURE ENGINEERING
    # -----------------------------------------
    df["Decade"] = (df["Year"] // 10) * 10

    # -----------------------------------------
    # 2. ENCODING CATEGORICAL VARIABLES
    # -----------------------------------------
    # Label encode 'Platform'
    le_platform = LabelEncoder()
    df["Platform_Encoded"] = le_platform.fit_transform(df["Platform"])

    # -----------------------------------------
    # 3. SCALING NUMERICAL VARIABLES
    # -----------------------------------------
    scaler = StandardScaler()
    df["Global_Sales_Scaled"] = scaler.fit_transform(df[["Global_Sales"]])

    print("✅ Post-validation preprocessing completed.")
    return df

def load_and_preprocess_data(cleaned_df_path: str, output_path: str) -> None:
    """
    Load cleaned data from the CSV, perform post-validation preprocessing, 
    and save the preprocessed data.
    
    Parameters:
        cleaned_df_path (str): Path to the cleaned DataFrame (CSV).
        output_path (str): Path to save the preprocessed data.
    """
    # Load the cleaned DataFrame from the specified CSV path
    cleaned_df = pd.read_csv(cleaned_df_path)
    print(f"✅ Cleaned data loaded from {cleaned_df_path}")
    
    # Perform post-validation preprocessing
    df_ready_for_redis = post_validation_preprocessing(cleaned_df)
    
    # Save the preprocessed data locally
    df_ready_for_redis.to_csv(output_path, index=False)
    print(f"✅ Preprocessed data saved to {output_path}")

# Example usage
cleaned_df_path = "/home/anupamarai24128432/vgsales_project/data/validated_vgsales.csv"  # Adjust this path as needed
output_path = "/home/anupamarai24128432/vgsales_project/data/final_preprocessed_vgsales.csv"  # Adjust this path as needed
load_and_preprocess_data(cleaned_df_path, output_path)
