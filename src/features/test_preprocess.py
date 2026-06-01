import pandas as pd

from src.features.preprocess import (
    get_preprocessor,
)

from src.utils.config import (
    PROCESSED_DATA_DIR,
)


def main():

    X_train = pd.read_csv(
        PROCESSED_DATA_DIR / "X_train.csv"
    )

    preprocessor = get_preprocessor()

    X_processed = preprocessor.fit_transform(
        X_train
    )

    print(
        "Original Shape:",
        X_train.shape,
    )

    print(
        "Processed Shape:",
        X_processed.shape,
    )


if __name__ == "__main__":
    main()