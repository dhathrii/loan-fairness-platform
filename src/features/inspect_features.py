from src.data.load_data import load_data
from src.features.rename_columns import rename_columns
from src.utils.config import RAW_DATA_DIR


def main():

    df = load_data(
        RAW_DATA_DIR / "german_credit.csv"
    )

    df = rename_columns(df)

    print("\nColumns:")
    print(df.columns)

    print("\nData Types:")
    print(df.dtypes)

    print("\nTarget Distribution:")
    print(df["target"].value_counts())


if __name__ == "__main__":
    main()