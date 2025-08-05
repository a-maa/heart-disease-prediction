import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

def load_dataset(type="default", path="./datasets/heart.csv") -> pd.DataFrame:
    data = pd.read_csv(path)

    match type:
        case "str":
            return data.astype(str)
        case "default":
            return data
        case _:
            raise ValueError("Invalid type")

def transform_columns(data: pd.DataFrame) -> pd.DataFrame:
    data = data.drop('FastingBS', axis=1)
    data['RestingBP'] = data['RestingBP'].replace(0, data['RestingBP'].median())
    data['Cholesterol'] = data['Cholesterol'].replace(0, data['Cholesterol'].median())
    data['Oldpeak'] = data['Oldpeak'].abs()

    sex_encoded = pd.get_dummies(data['Sex'], prefix='Sex', drop_first=True)
    data = pd.concat([data, sex_encoded], axis=1).drop('Sex', axis=1)
    
    exercise_angina_encoded = pd.get_dummies(data['ExerciseAngina'], prefix='ExerciseAngina', drop_first=True)
    data = pd.concat([data, exercise_angina_encoded], axis=1).drop('ExerciseAngina', axis=1)
    
    chest_pain_type_encoded = pd.get_dummies(data['ChestPainType'], prefix='ChestPainType')
    data = pd.concat([data, chest_pain_type_encoded], axis=1).drop('ChestPainType', axis=1)
    
    resting_ecg_encoded = pd.get_dummies(data['RestingECG'], prefix='RestingECG')
    data = pd.concat([data, resting_ecg_encoded], axis=1).drop('RestingECG', axis=1)
    
    st_slope_encoded = pd.get_dummies(data['ST_Slope'], prefix='ST_Slope')
    data = pd.concat([data, st_slope_encoded], axis=1).drop('ST_Slope', axis=1)

    return data

def plot_correlation_matrix(data: pd.DataFrame) -> None:
    correlation_matrix = data.corr()

    fig = px.imshow(
        correlation_matrix,
        color_continuous_scale=px.colors.diverging.RdBu,
        title="Correlation Matrix"
    )

    fig.show()

def train_test_split_stratified_kfold_cross_validation(data: pd.DataFrame, n_splits=6):
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=1)

    X = data.drop('HeartDisease', axis=1)
    y = data['HeartDisease']

    for train_index, test_index in skf.split(X, y):
        train_data, test_data = data.iloc[train_index], data.iloc[test_index]
        yield train_data, test_data

def train_model(train_data) -> LogisticRegression:
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(train_data.drop('HeartDisease', axis=1))
    y_train = train_data['HeartDisease']

    clsf = LogisticRegression()
    clsf.fit(x_train, y_train)

    return clsf

def evaluate_model(model, test_data) -> float:
    scaler = MinMaxScaler()
    X_test = scaler.fit_transform(test_data.drop('HeartDisease', axis=1))
    y_test = test_data['HeartDisease']

    y_prediction = model.predict(X_test)

    accurancy = roc_auc_score(y_test, y_prediction)

    return accurancy

def run_training_and_testing(data, n_splits=6) -> None:
    fold_accuracies = []

    for fold, (train_data, test_data) in enumerate(train_test_split_stratified_kfold_cross_validation(data, n_splits=n_splits)):
        # print(f"Fold: {fold + 1}")

        clsf = train_model(train_data)
        linear_accuracy = evaluate_model(clsf, test_data)
        fold_accuracies.append(linear_accuracy)

        # print(f"Accuracy: {linear_accuracy}")

    return fold_accuracies

def main():
    data = load_dataset()
    transformed_data = transform_columns(data)
    # plot_correlation_matrix(transformed_data)
    fold_accuracies = run_training_and_testing(transformed_data, n_splits=6)

    print(f"Mean roc_auc logisctic regression: {np.mean(fold_accuracies)}")

if __name__ == "__main__":
    main() 