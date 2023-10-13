import joblib
from Training import clf
model_filename = 'your_model_filename.pkl'
joblib.dump(clf, model_filename)
loaded_model = joblib.load(model_filename)