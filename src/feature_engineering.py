import pandas as pd

CLEANED_DATA_PATH = "data/cars_cleaned.csv"
FEATURES_DATA_PATH = "data/cars_cleaned_with_features.csv"


def _create_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Kreira osnovne vremenske i transformisane karakteristike."""
    df = df.copy()

    # 1. Car_age - Trenutna godina je 2026
    current_year = 2026
    df["car_age"] = current_year - df["year"]

    # Zastita od deljenja sa nulom: ako je auto nov (0 godina)
    adjusted_age = df["car_age"].replace(0, 1)

    # 2. Mileage_per_year - Kilometraža po godini 
    df["mileage_per_year"] = df["mileage_km"] / adjusted_age

    # 3. Engine_volume_liters - Zapremina motora u litrima 
    if "volume_cm3" in df.columns:
        df["engine_volume_liters"] = df["volume_cm3"] / 1000.0

    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:

    df_features = (
        df
        .pipe(_create_basic_features)
        .reset_index(drop=True)
    )

    return df_features


def main() -> None:
    """Load cleaned data, build features, and save the feature-engineered dataset."""
    print("Loading cleaned dataset...")

    df_cleaned = pd.read_csv(CLEANED_DATA_PATH)

    print("Building features...")

    df_features = build_features(df_cleaned)
    print("Saving feature-engineered dataset...")

    df_features.to_csv(FEATURES_DATA_PATH, index=False)

    print(f"Feature-engineered dataset saved to: {FEATURES_DATA_PATH}")
    
if __name__ == "__main__":
    main()
