import pandas as pd
import pyarrow as pa
import redis
import sqlalchemy
from sklearn.model_selection import train_test_split

# === Load and Prepare Data ===
def load_and_prepare_data(file_path: str, drop_cols=None) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    if drop_cols:
        df = df.drop(columns=[col for col in drop_cols if col in df.columns])
    return df

# === Save to Local CSV ===
def save_locally(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False)
    print(f"✅ Cleaned data saved locally at {path}")

# === Split into Train and Test ===
def split_data(df: pd.DataFrame, test_size=0.3, random_state=42):
    return train_test_split(df, test_size=test_size, random_state=random_state)

# === Save to Redis ===
def save_to_redis(df: pd.DataFrame, redis_key: str, host='localhost', port=6379, db=0):
    redis_client = redis.Redis(host=host, port=port, db=db)
    table = pa.Table.from_pandas(df)
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    buffer = sink.getvalue()
    redis_client.set(redis_key, buffer.to_pybytes())
    print(f"✅ Data saved to Redis key: {redis_key}")

# === Upload to MariaDB ===
def upload_to_mariadb(df: pd.DataFrame, table_name: str,
                      db_user='anupamarai', db_password='anupamarai-123',
                      db_host='localhost', db_port=3308, db_name='vgsales'):
    connection_str = f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    engine = sqlalchemy.create_engine(connection_str)
    df.to_sql(name=table_name, con=engine, if_exists='replace', index=False)
    print(f"✅ Data uploaded to MariaDB table: {table_name}")

# === Master function to run the pipeline ===
def run_full_saving_pipeline():
    # Step 1: Load and Clean
    input_path = "/home/anupamarai24128432/vgsales_project/data/final_preprocessed_vgsales.csv"
    df = load_and_prepare_data(input_path, drop_cols=["Name", "Rank"])

    # Step 2: Save cleaned full set
    save_locally(df, "/home/anupamarai24128432/vgsales_project/data/Cleaned_data_After_Splitting.csv")

    # Step 3: Split
    train_df, test_df = split_data(df)

    # Step 4: Save to Redis
    save_to_redis(df, 'vgsales_cleaned')
    save_to_redis(train_df, 'vgsales_train')
    save_to_redis(test_df, 'vgsales_test')

    # Step 5: Save to MariaDB
    upload_to_mariadb(df, 'cleaned_vgsales')
    upload_to_mariadb(train_df, 'train_vgsales')
    upload_to_mariadb(test_df, 'test_vgsales')

    print("🎉 All steps completed successfully.")

# if __name__ == "__main__":
#     run_full_saving_pipeline()
