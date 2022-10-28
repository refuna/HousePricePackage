import joblib
import pandas as pd
import os
from .preprocess import preprocessing_pipe

def make_predictions(df: pd.DataFrame) -> int:
    if not os.path.exists('models/LGBM_model.joblib'):
        raise Exception('Merci de créer un dossier models contenant LGBM_model.joblib, OneHotEncoder.joblib, sc.joblib'
                        'Vous pouvez les crééer en lancant HousePricePackage.train ')
    preprocessed_df = preprocessing_pipe(df)
    model = joblib.load('models/LGBM_model.joblib')
    result = model.predict(preprocessed_df)
    return result

if __name__ == '__main__':
    df_test = pd.read_csv('data/test.csv')
    columns = ["SalePrice", "OverallQual", "GrLivArea", "GarageArea", "TotalBsmtSF", "Street", "LotShape"]
    feature_columns = columns[1:]
    result = make_predictions(df_test[feature_columns])
    print(result)
