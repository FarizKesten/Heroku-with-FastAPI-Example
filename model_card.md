# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
RandomForestClassifier is used as a model from sklearn with mostly default settings.

## Intended Use
The model is used to predict an individual salary based on other attributes

## Training Data
Data are based from here: https://archive.ics.uci.edu/ml/datasets/census+income. 80% of the
data is reserved for the training data.

## Evaluation Data

20% out of the data here are used: https://archive.ics.uci.edu/ml/datasets/census+income. 20%
of the data is reserved for the evaluation data.

## Metrics
Precision, Recall & FBeta score are used. Current model performance:
| precision [%] | Recall [%] | FBeta |
|--|--|--|
|64.9%|51.4%|0.574|

## Ethical Considerations
Metrics are calculated in slices (`ml/model/slice_output.txt`)
Further analysis should be done on the metrics to detect potential bias that could result to any discriminations.

## Caveats and Recommendations
Recommendation: Try using Aequitas-Package(https://github.com/dssg/aequitas) to further find bias & fairness of the used data.