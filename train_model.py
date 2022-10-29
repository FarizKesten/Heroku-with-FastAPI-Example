# Script to train machine learning model.

#%%
# Add the necessary imports for the starter code.
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics
from ml.pipeline import compute_model_performance_on_slices
from helper import read_config
import pandas as pd
import joblib


# Add code to load in the data.
#%%
data = pd.read_csv("data/census_clean.csv",skipinitialspace=True)

#%%
# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = read_config('config.yml')['data']['cat_features']
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features,
    label="salary", training=True
)

# Process the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features,
    label="salary", training=False, encoder=encoder, lb=lb
)
#%%
# Train and save a model.
model = train_model(X_train, y_train)
joblib.dump(lb, "model/lb.joblib")
joblib.dump(encoder, "model/encoder.joblib")
joblib.dump(model, "model/model.joblib")

# %%
#print out the general result of the model
precision, recall, fbeta = compute_model_metrics(y_test, inference(model, X_test))
print(f"General: Precision: {precision} Recall {recall} FBeta {fbeta}")

# print out the result based on slicing
results = compute_model_performance_on_slices(test, model, encoder, lb)
print(results)
# %%
