from .data import process_data
from .model import inference, compute_model_metrics


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



def compute_model_performance_on_slices(data, model, encoder, lb):
    """compute model performance in slices based from categorical features

    Args:
        model (RandomForestClassifier): trained model
        encoder (OneHotEncoder): trained encoder
        lb (LabelBinarizer): trained label-binarizer
        df (pandas data format): input values

    Returns:
        performance: performance values of the model based on each categories
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

    results = ""
    for category in cat_features:
        for unique in data[category].unique():
            data_temp = data[data[category] == unique]
            y, preds = run_pipeline(data_temp, model, lb, encoder)
            precision, recall, fbeta = compute_model_metrics(y, preds)
            results += f"Slice: {category}-{unique} Precision: {precision} Recall {recall} FBeta {fbeta}\n"
    return results


