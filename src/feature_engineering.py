import pandas as pd

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df['Age'] = 2025 - df['YearBuilt']
    df['TotalSquareFoot'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    df['SinceRemodel'] = df['YrSold'] - df['YearRemodAdd']
    df['TotalBathrooms'] = (
        df['FullBath'] + df['HalfBath'] * 0.5 +
        df['BsmtFullBath'] + df['BsmtHalfBath'] * 0.5
    )
    df['TotalPorchSF'] = (
        df['OpenPorchSF'] + df['EnclosedPorch'] +
        df['3SsnPorch'] + df['ScreenPorch']
    )
    df['TotalRooms'] = df['TotRmsAbvGrd'] + df['BedroomAbvGr']
    df['HouseAgeAtSale'] = df['YrSold'] - df['YearBuilt']

    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    qual_map = {'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'None':0}
    bsmt_exp_map = {'None':0,'No':1,'Mn':2,'Av':3,'Gd':4}
    bsmt_fin_map = {'None':0,'Unf':1,'LwQ':2,'Rec':3,'BLQ':4,'ALQ':5,'GLQ':6}
    functional_map = {'Sev':1,'Maj2':2,'Maj1':3,'Mod':4,'Min2':5,'Min1':6,'Typ':7}
    land_slope_map = {'Gtl':1,'Mod':2,'Sev':3}
    paved_drive_map = {'N':0,'P':1,'Y':2}
    central_air_map = {'N':0,'Y':1}
    utilities_map = {'ELO':1,'NoSeWa':2,'NoSewr':3,'AllPub':4}

    ordinal_cols = {
        'ExterQual': qual_map, 'ExterCond': qual_map,
        'BsmtQual': qual_map, 'BsmtCond': qual_map,
        'KitchenQual': qual_map, 'HeatingQC': qual_map,
        'FireplaceQu': qual_map, 'GarageQual': qual_map,
        'GarageCond': qual_map, 'PoolQC': qual_map,
        'BsmtExposure': bsmt_exp_map,
        'BsmtFinType1': bsmt_fin_map,
        'BsmtFinType2': bsmt_fin_map,
        'Functional': functional_map,
        'LandSlope': land_slope_map,
        'PavedDrive': paved_drive_map,
        'CentralAir': central_air_map,
        'Utilities': utilities_map
    }

    for col, mapping in ordinal_cols.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)

    categorical_cols = df.select_dtypes(include='object').columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    return df
