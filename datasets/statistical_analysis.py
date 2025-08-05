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
    # Since FastingBS contains a lot of 0 values we can drop it:
    fasting_bs_zeros_count = (data['FastingBS'] == 0).sum()
    print(fasting_bs_zeros_count)
    data = data.drop('FastingBS', axis=1)

    # We need to replace 0 values in RestingBP and Cholesterol with median values:
    data['RestingBP'] = data['RestingBP'].replace(0, data['RestingBP'].median())
    data['Cholesterol'] = data['Cholesterol'].replace(0, data['Cholesterol'].median())

    # We need to change negative values of oldpeak to positive since it cannot be negative:
    oldpeak_negative_values_count = (data['Oldpeak'] < 0).sum()
    print(f'Oldpeak negative values: {oldpeak_negative_values_count}')
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

def descriptive_statistics(data: pd.DataFrame) -> pd.DataFrame:
    for column in data.columns:
        if 'ChestPainType' in column or 'RestingECG' in column or 'ST_Slope' in column:
            data[column] = data[column].astype(int)

    print(data.describe().iloc[:, :10])
    print(data.describe().iloc[:, 10:])


    return data.describe(include='all')


def main():
    data = load_dataset()
    transformed_data = transform_columns(data)
    # plot_correlation_matrix(transformed_data)
    descriptive_statistics(transformed_data)

if __name__ == "__main__":
    main()