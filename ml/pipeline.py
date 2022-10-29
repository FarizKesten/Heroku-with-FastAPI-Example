from .data import process_data
from .model import inference


def run_pipeline(data, model, lb, encoder):
    """ Run preprocessing + model inference

    Inputs
    ------
    data: raw unprocessed data
    model : RandomForest model
        Trained machine learning model.
    lb : label binarizer
    encoder : one-hote-encoder
    Returns
    -------
    (y, preds) : np.array
        True data & Predictions from the model.
    """
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    X, y, encoder, lb = process_data(data, categorical_features=cat_features,
        label="salary", training=False, encoder=encoder, lb=lb)
    preds = inference(model, X)
    return (y, preds)