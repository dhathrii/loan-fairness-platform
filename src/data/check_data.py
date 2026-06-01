from src.data.load_data import load_data
from src.utils.config import RAW_DATA_DIR


def main():

    data_path = RAW_DATA_DIR / "german_credit.csv"

    df = load_data(data_path)

    print("\nShape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns)

    print("\nFirst Five Rows:")
    print(df.head())


if __name__ == "__main__":
    main()