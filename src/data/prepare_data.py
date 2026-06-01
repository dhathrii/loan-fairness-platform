from src.data.load_data import load_data
from src.data.split_data import split_data
from src.features.rename_columns import rename_columns
from src.utils.config import RAW_DATA_DIR, PROCESSED_DATA_DIR


def main():
    df = load_data(RAW_DATA_DIR / "german_credit.csv")

    # rename raw columns to English before splitting
    df = rename_columns(df)

    target_col = "target"

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    ) = split_data(df, target_col=target_col)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    X_train.to_csv(PROCESSED_DATA_DIR / "X_train.csv", index=False)
    X_val.to_csv(PROCESSED_DATA_DIR / "X_val.csv", index=False)
    X_test.to_csv(PROCESSED_DATA_DIR / "X_test.csv", index=False)

    y_train.to_csv(PROCESSED_DATA_DIR / "y_train.csv", index=False)
    y_val.to_csv(PROCESSED_DATA_DIR / "y_val.csv", index=False)
    y_test.to_csv(PROCESSED_DATA_DIR / "y_test.csv", index=False)

    print("Data split completed.")


if __name__ == "__main__":
    main()