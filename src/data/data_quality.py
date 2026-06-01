from src.data.load_data import load_data
from src.features.rename_columns import rename_columns
from src.utils.config import RAW_DATA_DIR


def main():

    df = load_data(
        RAW_DATA_DIR / "german_credit.csv"
    )

    df = rename_columns(df)

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nDuplicate Rows:")
    print(df.duplicated().sum())

    print("\nTarget Distribution:")
    print(df["target"].value_counts())

    print("\nSummary Statistics:")
    print(df.describe())


if __name__ == "__main__":
    main()