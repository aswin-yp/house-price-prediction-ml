import pandas as pd

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df['MasVnrArea'] = df['MasVnrArea'].fillna(0)

    df.loc[df['GarageType'].isna(), 'GarageYrBlt'] = 0
    df.loc[(df['GarageType'].notna()) & (df['GarageYrBlt'].isna()), 'GarageYrBlt'] = df['YearBuilt']

    grp = df.groupby('Neighborhood')['LotFrontage'].median()
    for k, v in grp.items():
        df.loc[(df['Neighborhood'] == k) & (df['LotFrontage'].isna()), 'LotFrontage'] = v

    df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])

    fill_none_cols = ['Alley','MasVnrType','FireplaceQu','PoolQC','Fence','MiscFeature']
    df[fill_none_cols] = df[fill_none_cols].fillna('None')

    garage_cols = ['GarageType','GarageFinish','GarageQual','GarageCond']
    df.loc[df['GarageType'].isna(), garage_cols] = 'None'

    bsmt_cols = ['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2']
    df.loc[df['BsmtQual'].isna(), bsmt_cols] = 'None'

    df['BsmtExposure'] = df['BsmtExposure'].fillna(df['BsmtExposure'].mode()[0])
    df['BsmtFinType2'] = df['BsmtFinType2'].fillna(df['BsmtFinType2'].mode()[0])

    df.loc[954, 'KitchenAbvGr'] = 1

    return df
