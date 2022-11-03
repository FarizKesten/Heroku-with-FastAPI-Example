#%%
# Script to train machine learning model.

# Add the necessary imports for the starter code.
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics
from ml.pipeline import compute_model_performance_on_slices
from helper import read_config
import pandas as pd
import joblib
import pathlib


main_path = pathlib.Path(__file__).parent.resolve()
data_path = main_path / "data"
model_path = main_path / "ml" / "model"
# Add code to load in the data.
#%%
data = pd.read_csv(data_path / "census_clean.csv",skipinitialspace=True, index_col=0)

#%%
# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)
test.to_csv(data_path / "test.csv")
cat_features = read_config(main_path / 'config.yml')['data']['cat_features']
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
joblib.dump(lb, model_path / "lb.joblib")
joblib.dump(encoder,model_path / "encoder.joblib")
joblib.dump(model, model_path / "model.joblib")

# %%
#print out the general result of the model
precision, recall, fbeta = compute_model_metrics(y_test, inference(model, X_test))
print(f"General: Precision: {precision} Recall {recall} FBeta {fbeta}")

#%%
# print out the result based on slicing
results = compute_model_performance_on_slices(test, model, encoder, lb)
print(results)

# save results to slice_output.txt
with open(model_path / "slice_output.txt", "w") as f:
    f.write(str(results))




# %%
