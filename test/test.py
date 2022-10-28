# Create some tests for the housePircePackage
import unittest
import pandas as pd
import HousePricePackage as Hpp


class TestHousePrice(unittest.TestCase):
    def test_housePrice(self):
        input_data = pd.DataFrame(
                            [{"OverallQual": 5, "GrLivArea": 1000,
                             "GarageArea": 500, "TotalBsmtSF": 500,
                             "Street": "Pave", "LotShape": "Reg"}]
                            )
        house_price = Hpp.make_predictions(input_data)[0]
        self.assertEqual(int(house_price), 61164)

    def test_trainModel(self):
        df = pd.read_csv('data/train.csv')
        columns = ["SalePrice", "OverallQual", "GrLivArea", "GarageArea", "TotalBsmtSF", "Street", "LotShape"]
        score = Hpp.build_model(df[columns])


