# Ames Housing Price Prediction Pipeline

A comprehensive machine learning pipeline for predicting housing prices using the Ames Housing Dataset. The pipeline includes data loading, preprocessing, training multiple regression models, and evaluating their performance.

## Project Structure
Ames_Houses/
├── data/
│ └── AmesHousingUpdated.csv
├── src/
│ ├── init.py
│ ├── load.py
│ ├── train.py
│ ├── evaluate.py
│ └── config.py
├── artifacts/
│ └── (saved models)
├── README.md
└── requirements.txt

Model Performance
The evaluation script produces:

Cross-validation RMSE for each model

Leaderboard ranking models by performance

Test set metrics (MAE, RMSE) for the best model

License
This project is for educational purposes.

Acknowledgments
Ames Housing Dataset originally compiled by Dean De Cock

Scikit-learn documentation for pipeline examples

Author
John Emmanuel Durosimi Terry
