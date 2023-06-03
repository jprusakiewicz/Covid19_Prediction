from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler

import pandas as pd
from sklearn.model_selection import train_test_split


def preprocess_data(df: pd.DataFrame):
    df = df[df["countryterritoryCode"] == "POL"]
    df['dateRep'] = pd.to_datetime(df['dateRep'], format="%d/%m/%Y")
    df = df.sort_values("dateRep")
    cleaned_data = df.drop(
        ["countryterritoryCode", "continentExp", "geoId", "countriesAndTerritories", "dateRep", "day", "month", "year",
         "popData2020", "deaths"], axis=1)
    cleaned_data['cases_next_day'] = df['cases'].shift(-1)
    x = cleaned_data.drop(["cases_next_day"], axis=1).values[:-1]
    y = cleaned_data["cases_next_day"].values[:-1]
    return train_test_split(x, y, test_size=0.3, shuffle=False)


def get_scaler(config):
    return MinMaxScaler()
    # return StandardScaler() # todo add as option


def get_preprocessor(config):
    numeric_transformer = Pipeline(
        # todo if list will be empty, (nothing in config) it will fail
        steps=[
            ("scaler", get_scaler(config.scaler))  # todo choose in config
        ]
    )

    return numeric_transformer
