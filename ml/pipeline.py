from .data import process_data
from .model import inference, compute_model_metrics
from helper import read_config
import joblib
import pathlib


main_path = pathlib.Path(__file__).parent.parent.resolve()
model_path = pathlib.Path(__file__).parent.resolve() / "model"



def run_pipeline(data,
                 label='salary',
                 model=joblib.load( model_path / "model.joblib"),
                 lb=joblib.load(model_path / "lb.joblib"),
                 encoder=joblib.load(model_path / "encoder.joblib"),
                 class_name=False):
    """ Run preprocessing + model inference

    Inputs
    ------
    data: raw unprocessed data
    model : RandomForest model
        Trained machine learning model.
    lb : label binarizer
    encoder : one-hot-encoder
    class_name : bool (default False)
    If True, return class name instead of class index.
    Returns
    -------
    (y, preds) : np.array
        True data & Predictions from the model.
    """
    cat_features = read_config(main_path / 'config.yml')['data']['cat_features']
    X, y, encoder, lb = process_data(data, categorical_features=cat_features,
        label=label, training=False, encoder=encoder, lb=lb)
    preds = inference(model, X)
    if class_name:
        preds = lb.inverse_transform(preds)
        y = lb.inverse_transform(y)

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
    cat_features = read_config(main_path / 'config.yml')['data']['cat_features']
    results = ""
    for category in cat_features:
        for unique in data[category].unique():
            data_temp = data[data[category] == unique]
            y, preds = run_pipeline(data_temp, model=model, lb=lb, encoder=encoder)
            precision, recall, fbeta = compute_model_metrics(y, preds)
            results += f"Slice: {category}-{unique} Precision: {precision} Recall {recall} FBeta {fbeta}\n"
    return results


