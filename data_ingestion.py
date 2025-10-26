#------data_ingestion------

import pandas as pd
from sqlalchemy import create_engine

def ingest_data(
    db_user="anupamarai",
    db_password="anupamarai-123",
    db_host="127.0.0.1",
    db_port="3308",
    db_name="vgsales",
    table_name="dim_game",
    csv_path="/home/anupamarai24128432/Downloads/vgsales.csv"
):
    print("📥 Reading CSV file...")
    df = pd.read_csv(csv_path)
    print("✅ CSV read successfully.")

    engine = create_engine(
        f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    )

    print(f"🗃 Ingesting data into `{table_name}` table...")
    df.to_sql(table_name, con=engine, if_exists='replace', index=False)
    print("✅ Data ingestion completed.")
    return df

# To run this section:
if __name__ == "__main__":
    raw_df = ingest_data()
