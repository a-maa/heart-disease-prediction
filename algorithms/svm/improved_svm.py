import pandas as pd
import plotly.express as px
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC

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

def train_model(train_data, kernel='linear', C=1.0, gamma='scale') -> SVC:
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(train_data.drop('HeartDisease', axis=1))
    y_train = train_data['HeartDisease']

    svc_classifier = SVC(kernel=kernel, C=C, gamma=gamma)
    svc_classifier.fit(X_train, y_train)

    return svc_classifier

def run_training_and_testing(data, n_splits=6) -> None:
    for fold, (train_data, test_data) in enumerate(train_test_split_stratified_kfold_cross_validation(data, n_splits=n_splits)):
        print(f"Fold: {fold + 1}")

        X_train = train_data.drop('HeartDisease', axis=1)
        y_train = train_data['HeartDisease']
        X_test = test_data.drop('HeartDisease', axis=1)
        y_test = test_data['HeartDisease']

        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        for kernel in ['linear', 'sigmoid', 'poly', 'rbf']:
            model = SVC(kernel=kernel, C=1.0, gamma='scale')  # Or other hyperparameters as needed
            model.fit(X_train_scaled, y_train)
            y_prediction = model.predict(X_test_scaled)
            accuracy = roc_auc_score(y_test, y_prediction)
            print(f"{kernel.capitalize()} Kernel Accuracy: {accuracy}")

def main():
    data = load_dataset()
    transformed_data = transform_columns(data)
    # plot_correlation_matrix(transformed_data)
    run_training_and_testing(transformed_data, n_splits=6)

if __name__ == "__main__":
    main()