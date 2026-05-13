from src.load import load_data
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import  StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from src.config import TARGET, TEST_SIZE, RANDOM_STATE, MODEL_PATH
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate_data(data):
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


    models = {
        "Linear Regression": LinearRegression(),
        "Lasso": Lasso(alpha=0.001, max_iter=10000),
        "Ridge": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, random_state=RANDOM_STATE)
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    results = []

    for name, model in models.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])
        print(f" Evaluating: {name}")
        cv_scores = cross_val_score(pipeline, X_train, y_train,scoring="neg_root_mean_squared_error", cv=cv)

        rmse = -cv_scores.mean()
        results.append((name, rmse))

        print(f"{name} CV RMSE: {rmse:.5f}")

    leaderboard = sorted(results, key=lambda x: x[1])

    print(" Leaderboard:")
    for rank, (name, rmse) in enumerate(leaderboard, 1):
        print(f"{rank}. {name} → RMSE: {rmse:.5f}")

    # Train best model on full train data
    best_model_name = leaderboard[0][0]
    best_model = models[best_model_name]

    final_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", best_model)
    ])

    final_pipeline.fit(X_train, y_train)

    # Evaluate on test set (REAL performance)
    predictions = final_pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    print(f" Best Model: {best_model_name}")
    print(f"Test MAE: {mae:.5f}")
    print(f"Test RMSE: {rmse:.5f}")

    return final_pipeline


if __name__ == "__main__":
    import joblib
    from src.load import load_data
    from src.config import MODEL_PATH

    data = load_data()

    final_pipeline, leaderboard = evaluate_data(data)

    joblib.dump(final_pipeline, MODEL_PATH)

    print(" Model saved successfully!")




        