import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder, LabelBinarizer

def process_data(df: pd.DataFrame, lb_cols: list=[], le_cols:list =[]) -> pd.DataFrame:
    ''' Args
    df - the data frame to process. Should be a pd.DataFrame
    lb_cols - columns to apply label binarizing to. Uses sklearn's LabelBinarizer. 
    le_cols - columns to apply label encoding to. Uses sklearn's LabelEncoder. 
    
    Note: le_cols is good for ordinal data while lb_cols is used similar to one-hot
    '''
    final_df = df.copy()
    lb = LabelBinarizer()
    for lb_col in lb_cols:
        lb.fit(df[lb_col])
        to_merge_df = pd.DataFrame(lb.transform(df[lb_col]), columns=lb.classes_)
        final_df = pd.merge(final_df, to_merge_df, left_index=True, right_index=True)
    
    final_df= final_df.drop(lb_cols, axis=1)
    
    le = LabelEncoder()
    for le_col in le_cols:
        le.fit(df[lb_col])
        final_df[le_col] = le.fit_transform(final_df[le_col])
        
    return final_df

def lag_housing_df(resale_approval_full: pd.DataFrame):
    import time
    start = time.time()
    lagged_cols = np.append(resale_approval_full.columns.values, ["last_known_price", "months_since_last"])
    unique_flats = resale_approval_full[["block", "street_name", "storey_range"]].drop_duplicates()
    to_concat = []
    for flat in unique_flats.itertuples():
        subset = resale_approval_full[(resale_approval_full["block"] == flat[1]) & (resale_approval_full["street_name"] == flat[2]) & (resale_approval_full["storey_range"] == flat[3])]
        if subset.shape[0] > 1:
            test_subset = subset
            test_subset["last_known_price"] = test_subset["resale_price"].shift(1)
            test_subset["months_since_last"] = test_subset["month_index_since_1990"][1:] - test_subset["month_index_since_1990"].shift(1).dropna()
            to_concat.append(test_subset)

    lagged_data = pd.concat(to_concat, ignore_index=True)
    lagged_data.dropna(inplace=True)
    lagged_data.reset_index(drop=True, inplace=True)
    lagged_data.to_csv("data/fixed_resale_with_bus.csv")
    print(f"time taken: {time.time() - start}")    

def get_processed_gdp_data():
    gdp_data = pd.read_csv("data/gross-domestic-product-at-current-prices-annual.csv")
    gdp_data["perc_change"] = gdp_data["value"].pct_change() * 100
    gdp_data.drop("level_1", axis=1, inplace=True)
    gdp_data.columns=["year", "GDP_value", "GDP_perc_change"]
    return gdp_data
