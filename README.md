# heart-disease-prediction
[Note: this is a re-upload since the original repository is private]

This project compares four classification models to identify heart disease using a Kaggle dataset. Specifically, the models were logistic regression, support vector machines, neural network and k-nearest neighbours. These models were trained and tested using Python with their associated libraries. Each model was evaluted on the following metrics: ROC AUC, precision, recall, and F1 score. Neural network performed the best in terms of accuracy, while the kNN model displayed the highest recall value.

Results Summary:
| Metric | kNN | Logistic Regression | Neural Network | SVM (linear kernel) |
|---|---|---|---|---|
| ROC AUC | 0.907 | 0.900 | 0.910 | 0.840 |
| Precision | 0.833 | 0.830 | 0.867 | 0.850 |
| Recall | 0.899 | 0.890 | 0.886 | 0.890 |
| F1 | 0.860 | 0.860   | 0.875 | 0.870 |

The full accompanying report PDF can also be found on this repository.
