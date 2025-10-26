import pandas as pd
import numpy as np

# -------------------------------
# FUNCTION: Clean vgsales dataset
# -------------------------------
def clean_vgsales_data(file_path, save_cleaned_data=True, preview_data=True):
    """
    Cleans the vgsales dataset to align with Great Expectations validation rules and optionally saves and previews the cleaned data.
    """
    # Load the dataset
    df_raw = pd.read_csv(file_path)
    
    df_cleaned = df_raw.copy()

    # -------------------------------
    # CLEAN 'Name' COLUMN
    # -------------------------------
    df_cleaned['Name'] = df_cleaned['Name'].fillna('').astype(str).str.strip()
    df_cleaned['Name'] = df_cleaned['Name'].str.replace(r"[^A-Za-z0-9\s\:\-\&']", '', regex=True)
    df_cleaned['Name'] = df_cleaned['Name'].apply(lambda x: np.nan if x.strip() == '' else x)
    df_cleaned = df_cleaned[df_cleaned['Name'].notna()].copy()

    # -------------------------------
    # CLEAN 'Platform' COLUMN
    # -------------------------------
    valid_platforms = [
        'Wii', 'NES', 'PS4', 'PS3', 'X360', 'GB', 'DS', 'PS2', 'SNES',
        'GBA', '3DS', 'N64', 'XB', 'PC', 'PS', 'XOne'
    ]
    platform_corrections = {
        'XBOX 360': 'X360', 'XBOX ONE': 'XOne', 'XBOX': 'XB',
        'PLAYSTATION': 'PS', 'PLAYSTATION 2': 'PS2', 'PLAYSTATION 3': 'PS3',
        'PLAYSTATION 4': 'PS4', 'GAMEBOY': 'GB', 'GAME BOY': 'GB',
        'NINTENDO 64': 'N64', 'PSP': 'PS'
    }

    df_cleaned['Platform'] = df_cleaned['Platform'].astype(str).str.upper().str.strip()
    df_cleaned['Platform'] = df_cleaned['Platform'].replace(platform_corrections)
    df_cleaned = df_cleaned[df_cleaned['Platform'].isin(valid_platforms)].copy()
    df_cleaned = df_cleaned[df_cleaned['Platform'].notna()].copy()

    # -------------------------------
    # CLEAN 'Year' COLUMN
    # -------------------------------
    df_cleaned['Year'] = pd.to_numeric(df_cleaned['Year'], errors='coerce')
    df_cleaned = df_cleaned[df_cleaned['Year'].between(1980, 2025)].copy()
    df_cleaned = df_cleaned[df_cleaned['Year'].notna()].copy()

    # -------------------------------
    # CLEAN 'Global_Sales' COLUMN
    # -------------------------------
    df_cleaned['Global_Sales'] = pd.to_numeric(df_cleaned['Global_Sales'], errors='coerce')
    df_cleaned = df_cleaned[df_cleaned['Global_Sales'].notna() & (df_cleaned['Global_Sales'] >= 0)].copy()

    # -------------------------------
    # OPTIONAL: CLEAN OTHER SALES COLUMNS
    # -------------------------------
    other_sales = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']
    for col in other_sales:
        df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce').fillna(0)

    # Reset index
    df_cleaned.reset_index(drop=True, inplace=True)

    # -------------------------------
    # Save cleaned data (Optional)
    # -------------------------------
    if save_cleaned_data:
        cleaned_file_path = file_path.replace(".csv", "_cleaned.csv")
        df_cleaned.to_csv(cleaned_file_path, index=False)
        print(f"✅ Cleaned data saved to '{cleaned_file_path}'")
    
    # -------------------------------
    # Preview cleaned data (Optional)
    # -------------------------------
    if preview_data:
        print("Preview of cleaned data:")
        print(df_cleaned.head())

    return df_cleaned

# -------------------------------
# EXECUTION: Call the function
# -------------------------------
file_path = "/home/anupamarai24128432/vgsales_project/data/vgsales.csv"  
df_cleaned = clean_vgsales_data(file_path)
