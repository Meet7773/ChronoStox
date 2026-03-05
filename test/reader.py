import xgboost as xgb
from keras.models import load_model

model = xgb.Booster()
# model.load_model("sector_model_v7_UNIVERSAL_20251113_063532.joblib")

df = load_model('final_lstm_20251113_124137.keras')
df.summary()