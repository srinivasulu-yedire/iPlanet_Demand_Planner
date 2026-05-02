import pandas as pd
from utils.logger import log

def get_dropdown_options(file_path):
    """Fetches unique Region, Store, and Product names for the UI."""
    df = pd.read_excel(file_path, engine='openpyxl')
    regions = sorted(df['RegionName'].unique().tolist())
    stores = sorted(df['StoreName'].unique().tolist())
    products = sorted(df['ProductName'].unique().tolist())
    return regions, stores, products

def prepare_inference_data(file_path, region, store, product):
    """Filters the master data for a specific selection and aggregates it."""
    df = pd.read_excel(file_path, engine='openpyxl')
    
    # Filter
    mask = (df['RegionName'] == region) & \
           (df['StoreName'] == store) & \
           (df['ProductName'] == product)
    filtered_df = df[mask].copy()
    
    if filtered_df.empty:
        return None

    # Standardize Date
    filtered_df['BillDate'] = pd.to_datetime(filtered_df['BillDate'])

    filtered_df['True_Demand'] = pd.to_numeric(filtered_df.get('True Demand', 0), errors='coerce').fillna(0)
    if 'Bank Funded NoCost EMI Applied With BankName' in filtered_df.columns:
        filtered_df['Bank_Emi_Flag'] = filtered_df['Bank Funded NoCost EMI Applied With BankName'].astype(str).apply(
            lambda x: 0 if x.strip().lower() in ['', 'nan', 'none', 'no', 'n/a'] else 1
        )
    else:
        filtered_df['Bank_Emi_Flag'] = 0

    if 'Bank Funded Cashback Coverted To Percentage With BankName' in filtered_df.columns:
        filtered_df['Bank_Cashback_Pct'] = pd.to_numeric(
            filtered_df['Bank Funded Cashback Coverted To Percentage With BankName'].astype(str)
                .str.extract(r'([+-]?\d+(?:\.\d+)?)')[0],
            errors='coerce'
        ).fillna(0)
    else:
        filtered_df['Bank_Cashback_Pct'] = 0

    if 'Additional Apple Or Distributor Funded Scheme In Percentage' in filtered_df.columns:
        filtered_df['Additional_Funded_Scheme_Pct'] = pd.to_numeric(
            filtered_df['Additional Apple Or Distributor Funded Scheme In Percentage'].astype(str)
                .str.extract(r'([+-]?\d+(?:\.\d+)?)')[0],
            errors='coerce'
        ).fillna(0)
    else:
        filtered_df['Additional_Funded_Scheme_Pct'] = 0

    # Aggregate to Weekly (To fix the 240 vs 9 unit error)
    weekly_df = filtered_df.set_index('BillDate').resample('W').agg({
        'Quantity': 'sum',
        'MRP': 'mean',
        'ProductLevelDiscAmount': 'mean',
        'True_Demand': 'sum',
        'Bank_Emi_Flag': 'mean',
        'Bank_Cashback_Pct': 'mean',
        'Additional_Funded_Scheme_Pct': 'mean'
    }).reset_index()
    
    return weekly_df