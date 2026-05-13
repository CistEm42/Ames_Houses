from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data" / "AmesHousingUpdated.csv"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
# ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / 'model.pkl'
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET = 'SalePrice'
RANDOM_STATE = 42
TEST_SIZE = 0.2
