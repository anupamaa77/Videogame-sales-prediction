# ------Great Expectation (raw dataset)-----

import great_expectations as gx
import pandas as pd
from great_expectations.core.batch import RuntimeBatchRequest

def validate_raw_data(csv_path="/home/anupamarai24128432/vgsales_project/data/vgsales.csv"):
    print("📄 Loading raw dataset...")
    df = pd.read_csv(csv_path)
    print(f"✅ Raw data loaded: {df.shape[0]} rows")

    print("🛠 Setting up Great Expectations context...")
    context = gx.get_context()

    context.add_datasource(
        name="vg_pandas_datasource",
        class_name="Datasource",
        execution_engine={"class_name": "PandasExecutionEngine"},
        data_connectors={
            "runtime_connector": {
                "class_name": "RuntimeDataConnector",
                "batch_identifiers": ["default_identifier"]
            }
        }
    )

    suite_name = "vg_sales_expectation_suite"
    context.add_or_update_expectation_suite(expectation_suite_name=suite_name)

    batch_request = RuntimeBatchRequest(
        datasource_name="vg_pandas_datasource",
        data_connector_name="runtime_connector",
        data_asset_name="vgsales_data",
        runtime_parameters={"batch_data": df},
        batch_identifiers={"default_identifier": "default"}
    )

    validator = context.get_validator(
        batch_request=batch_request,
        expectation_suite_name=suite_name
    )

    # Define expectations
    validator.expect_table_row_count_to_be_between(min_value=1, max_value=1_000_000)
    validator.expect_column_to_exist("Name")
    validator.expect_column_to_exist("Platform")
    validator.expect_column_to_exist("Year")
    validator.expect_column_to_exist("Global_Sales")

    validator.expect_column_values_to_not_be_null("Name")
    validator.expect_column_values_to_not_be_null("Platform")
    validator.expect_column_values_to_not_be_null("Global_Sales")

    validator.expect_column_values_to_be_between("Year", min_value=1980, max_value=2025)
    validator.expect_column_values_to_be_between("Global_Sales", min_value=0)

    validator.expect_column_values_to_be_in_set("Platform", [
        "Wii", "NES", "PS4", "PS3", "X360", "GB", "DS", "PS2", "SNES",
        "GBA", "3DS", "N64", "XB", "PC", "PS", "XOne"
    ])
    validator.expect_column_values_to_match_regex("Name", r"^[A-Za-z0-9\s\:\-\&\']+$")

    validator.save_expectation_suite(discard_failed_expectations=False)

    print("📊 Validating raw dataset...")
    result = validator.validate()

    print("\n✅ Validation success:", result.success)
    print("📊 Stats:", result.statistics)

    print("\n📋 Expectation Results:\n")
    for idx, res in enumerate(result.results, 1):
        print(f"🔹 Expectation {idx}: {res.expectation_config.expectation_type}")
        print(f"   ➤ Success: {res.success}")
        if not res.success:
            if "unexpected_list" in res.result:
                print("   ⚠️ Unexpected values:", res.result["unexpected_list"])
            elif "unexpected_percent" in res.result:
                print("   ⚠️ Unexpected %:", res.result["unexpected_percent"])
        print("-" * 80)

# To run this section:
if __name__ == "__main__":
    validate_raw_data()
