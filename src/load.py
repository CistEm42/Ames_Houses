import numpy as np
import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / 'data'
OUTPUT_PATH = OUTPUT_DIR / 'AmesHousingUpdated.csv'


def load_data():
    data = pd.read_csv(OUTPUT_PATH)
    print(f"data loaded with {data.shape}")
    print(f"data loaded head {data.head()}")


    return data


if __name__ == '__main__':
    data = load_data()

