import pandas as pd
import json


def read_data(path: str = "data/covid_data.json") -> pd.DataFrame:
    with open(path) as f:
        data = json.load(f)
    return pd.DataFrame(data["records"])
