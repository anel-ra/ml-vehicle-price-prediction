import re
import pandas as pd

RAW_DATA_PATH = "data/cars.csv"
CLEANED_DATA_PATH = "data/cars_cleaned.csv"


def _standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()
    
    rename_dict = {
        "mileage(kilometers)": "mileage_km",
        "volume(cm3)": "volume_cm3",
        "priceUSD": "price_usd",
    }
    df.rename(columns=rename_dict, inplace=True)

    # Dodatno čišćenje za ostale kolone ako ima skrivenih razmaka
    df.columns = df.columns.str.strip().str.lower()
    
    return df


def _strip_string_values(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    text_columns = df.select_dtypes(include=["str"]).columns

    for col in text_columns:
        df[col] = df[col].astype(str).str.strip()

    return df


MISSING_LIKE_VALUES = {
    "",
    " ",
    "nan",
    "NaN",
    "NAN",
    "null",
    "Null",
    "NULL",
    "none",
    "None",
    "NONE",
 }

def _replace_missing_like_values(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    df = df.replace(list(MISSING_LIKE_VALUES), pd.NA)

    return df



def _convert_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()
    numeric_columns = ["price_usd", "year", "mileage_km", "volume_cm3"]
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def _filter_outliers_and_invalids(df: pd.DataFrame) -> pd.DataFrame:
    """Uklanja ili ispravlja ekstremne cjene, kilometraže i nevalidne godine."""

    df = df.copy()

    # 1. Nevalidne godine proizvodnje (npr. automobili ne mogu biti iz budućnosti niti stariji od 1900)
    current_year = 2026
    df = df[(df["year"] >= 1900) & (df["year"] <= current_year)]

    # 2. Ekstremno male ili velike cjene 
    df = df[(df["price_usd"] >= 100) & (df["price_usd"] <= 1000000)]

    # 3. Ekstremne kilometraže (npr. kilometraža ne može biti negativna, a sve preko 1,000,000 km je sumnjivo)
    df = df[(df["mileage_km"] >= 0) & (df["mileage_km"] <= 1000000)]

    return df



def _handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Rješava nedostajuće vrijednosti za specifične kolone."""
    df = df.copy()
    # Kategorijalne kolone: mijenjamo sa 'Unknown' ili najčešćom vrednosti (mode)
    for col in ["drive_unit", "segment"]:
        if col in df.columns:   
            df[col] = df[col].fillna("Unknown")
            
    # Numeričke kolone: volume_cm3 popunjavamo medijanom grupe 
    if "volume_cm3" in df.columns:
        df["volume_cm3"] = df["volume_cm3"].fillna(df["volume_cm3"].median())
        

    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:

    df_clean = (
        df
        .pipe(_standardize_column_names)
        .pipe(_strip_string_values)
        .pipe(_replace_missing_like_values)
        .pipe(_convert_numeric_columns)
        .pipe(_filter_outliers_and_invalids)
        .pipe(_handle_missing_values)
        .reset_index(drop=True)
    )

    return df_clean 


def main() -> None:
    """Load raw data, clean it, and save the cleaned dataset."""
    print("Loading raw dataset...")

    df_raw = pd.read_csv(RAW_DATA_PATH)

    print("Cleaning dataset...")

    df_cleaned = clean(df_raw)

    print("Saving cleaned dataset...")

    df_cleaned.to_csv(CLEANED_DATA_PATH, index=False)

    print(f"Cleaned dataset saved to: {CLEANED_DATA_PATH}")

if __name__ == "__main__":
    main()