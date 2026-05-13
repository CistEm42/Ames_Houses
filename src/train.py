from src.load import load_data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import  StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from src.config import TARGET, TEST_SIZE, RANDOM_STATE



def train_data(data, TARGET, TEST_SIZE, RANDOM_STATE):
    X = data.drop(columns=[TARGET])
    y = data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    NUMERICAL_FEATURES = X_train.select_dtypes(include=['int64', 'float64']).columns
    CATEGORICAL_FEATURES = X_train.select_dtypes(include=['object']).columns


    numerical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num",numerical_transformer, NUMERICAL_FEATURES),
        ("cat",categorical_transformer, CATEGORICAL_FEATURES)
    ])

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", LinearRegression())
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)

    print(f"MAE: {mae:.5f}")
    print(f"MSE: {mse:.5f}")
    print(f"RMSE: {rmse:.5f}")

    return pipeline

if __name__ == '__main__':
     import pandas as pd
     from src.config import DATA_DIR, TARGET, TEST_SIZE, RANDOM_STATE

     df = pd.read_csv(DATA_DIR)

     pipeline = train_data(df, TARGET, TEST_SIZE, RANDOM_STATE)