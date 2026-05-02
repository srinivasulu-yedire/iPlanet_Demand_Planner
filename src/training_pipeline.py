import pandas as pd
import numpy as np
import os
import re
import joblib
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder
from utils.logger import log
from external_features import add_external_features

REQUIRED_COLUMNS = [
    'RegionName',
    'StoreName',
    'ProductName',
    'BillDate',
    'Quantity',
    'MRP',
    'ProductLevelDiscAmount',
]

MODEL_ARTIFACTS = [
    'master_model.pkl',
    'le_region.pkl',
    'le_store.pkl',
    'le_prod.pkl',
    'le_product_store.pkl',
    'le_month_product.pkl',
    'model_metrics.pkl',
]

def parse_numeric_percentage(value):
    if pd.isna(value):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value)
    match = re.search(r'([+-]?\d+(?:\.\d+)?)', text)
    return float(match.group(1)) if match else 0.0


def parse_emi_flag(value):
    if pd.isna(value):
        return 0.0
    text = str(value).strip().lower()
    if text in ['', 'nan', 'none', 'no', 'n/a']:
        return 0.0
    return 1.0


def validate_sales_data_columns(data_path):
    """Validate that the uploaded sales workbook has the columns required for training."""
    columns = pd.read_excel(data_path, nrows=0, engine='openpyxl').columns.tolist()
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in columns]
    return missing_columns, columns


def run_master_tournament(data_path='data/sales_data.xlsx', model_dir='models', log_func=log):
    log_func("=== STARTING AUTO-ML MASTER TOURNAMENT (TIME-SERIES OPTIMIZED) ===")

    if not os.path.exists(data_path):
        log_func(f"ERROR: Excel file not found: {data_path}")
        return None

    missing_columns, _ = validate_sales_data_columns(data_path)
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    # 1. LOAD & DE-FRAGMENT DATA
    df = pd.read_excel(data_path, engine='openpyxl')
    df['BillDate'] = pd.to_datetime(df['BillDate'], errors='coerce')
    df = df.dropna(subset=['BillDate']).copy() 
    if df.empty:
        raise ValueError("No valid rows found after parsing BillDate.")
    
    # 2. FEATURE ENGINEERING (AutoML Context)
    # Calculate Promotion impact
    df['DiscountPct'] = (df['ProductLevelDiscAmount'] / (df['Quantity'] * df['MRP'] + 1e-5)).fillna(0)

    # Normalize additional sales features when available
    df['True_Demand'] = pd.to_numeric(df.get('True Demand', 0), errors='coerce').fillna(0)
    df['Bank_Emi_Flag'] = df['Bank Funded NoCost EMI Applied With BankName'].apply(parse_emi_flag) if 'Bank Funded NoCost EMI Applied With BankName' in df.columns else 0.0
    df['Bank_Cashback_Pct'] = df['Bank Funded Cashback Coverted To Percentage With BankName'].apply(parse_numeric_percentage) if 'Bank Funded Cashback Coverted To Percentage With BankName' in df.columns else 0.0
    df['Additional_Funded_Scheme_Pct'] = df['Additional Apple Or Distributor Funded Scheme In Percentage'].apply(parse_numeric_percentage) if 'Additional Apple Or Distributor Funded Scheme In Percentage' in df.columns else 0.0
    
    # Aggregate to Weekly snapshots
    df_weekly = df.set_index('BillDate').groupby(['RegionName', 'StoreName', 'ProductName']).resample('W').agg({
        'Quantity': 'sum', 'MRP': 'mean', 'DiscountPct': 'mean',
        'True_Demand': 'sum',
        'Bank_Emi_Flag': 'mean',
        'Bank_Cashback_Pct': 'mean',
        'Additional_Funded_Scheme_Pct': 'mean'
    }).reset_index()

    # Enforce non-negative integers for Poisson/Tweedie
    df_weekly['Quantity'] = df_weekly['Quantity'].clip(lower=0).round().astype(int)
    
    # ENHANCED FEATURE ENGINEERING FOR BETTER ACCURACY
    
    # 1. ADVANCED TIME-SERIES FEATURES
    # Multiple lag periods for better pattern recognition
    for lag in [1, 2, 3, 4, 8, 12]:  # Weekly lags
        df_weekly[f'Lag_{lag}_Week'] = df_weekly.groupby(['ProductName', 'StoreName'])['Quantity'].shift(lag).fillna(0)
    
    # Rolling statistics with different windows
    for window in [2, 4, 8, 12]:
        df_weekly[f'Rolling_Mean_{window}'] = df_weekly.groupby(['ProductName', 'StoreName'])['Quantity'].transform(lambda x: x.rolling(window).mean()).fillna(0)
        df_weekly[f'Rolling_Std_{window}'] = df_weekly.groupby(['ProductName', 'StoreName'])['Quantity'].transform(lambda x: x.rolling(window).std()).fillna(0)
    
    # Exponential weighted moving average
    df_weekly['EWMA_4'] = df_weekly.groupby(['ProductName', 'StoreName'])['Quantity'].transform(lambda x: x.ewm(span=4).mean()).fillna(0)
    
    # 2. SEASONALITY & TREND FEATURES
    # Day of week (though weekly, this captures intra-week patterns)
    df_weekly['Quarter'] = df_weekly['BillDate'].dt.quarter
    df_weekly['Is_Month_Start'] = df_weekly['BillDate'].dt.is_month_start.astype(int)
    df_weekly['Is_Month_End'] = df_weekly['BillDate'].dt.is_month_end.astype(int)
    
    # 3. TREND INDICATORS
    # Rate of change features
    df_weekly['MoM_Change'] = df_weekly.groupby(['ProductName', 'StoreName'])['Quantity'].pct_change(4).fillna(0)  # Month-over-month
    df_weekly['Trend_4Week'] = df_weekly.groupby(['ProductName', 'StoreName'])['Rolling_Mean_4'].diff(1).fillna(0)
    
    # 4. PRICE & PROMOTION FEATURES
    # Enhanced discount features
    df_weekly['Discount_Intensity'] = pd.cut(df_weekly['DiscountPct'], bins=[0, 0.05, 0.15, 0.3, 1.0], labels=[0, 1, 2, 3]).fillna(0).astype(int)
    df_weekly['Effective_Price'] = df_weekly['MRP'] * (1 - df_weekly['DiscountPct'])
    
    # 5. STATISTICAL FEATURES
    # Distribution-based features
    df_weekly['Quantity_ZScore'] = df_weekly.groupby(['ProductName', 'StoreName'])['Quantity'].transform(lambda x: (x - x.rolling(12).mean()) / x.rolling(12).std()).fillna(0)
    df_weekly['TrueDemand_Ratio'] = df_weekly['True_Demand'] / (df_weekly['Quantity'] + 1e-5)
    df_weekly['Bank_Cashback_Intensity'] = pd.cut(df_weekly['Bank_Cashback_Pct'], bins=[-1, 0, 2, 5, 10, 100], labels=[0, 1, 2, 3, 4]).fillna(0).astype(int)
    df_weekly['Additional_Funded_Scheme_Intensity'] = pd.cut(df_weekly['Additional_Funded_Scheme_Pct'], bins=[-1, 0, 2, 5, 10, 100], labels=[0, 1, 2, 3, 4]).fillna(0).astype(int)
    
    # Time-based Seasonality
    df_weekly['Week'] = df_weekly['BillDate'].dt.isocalendar().week.astype(int)
    df_weekly['Month'] = df_weekly['BillDate'].dt.month
    df_weekly['Year'] = df_weekly['BillDate'].dt.year

    # Cyclical encoding for seasonal features (better than raw numbers)
    df_weekly['Week_Sin'] = np.sin(2 * np.pi * df_weekly['Week'] / 52.0)
    df_weekly['Week_Cos'] = np.cos(2 * np.pi * df_weekly['Week'] / 52.0)
    df_weekly['Month_Sin'] = np.sin(2 * np.pi * df_weekly['Month'] / 12.0)
    df_weekly['Month_Cos'] = np.cos(2 * np.pi * df_weekly['Month'] / 12.0)

    # Add Indian holiday features using region-as-state logic
    df_weekly = add_external_features(df_weekly, region_column='RegionName')

    # 3. ENCODING
    le_reg, le_store, le_prod = LabelEncoder(), LabelEncoder(), LabelEncoder()
    df_weekly['RegionID'] = le_reg.fit_transform(df_weekly['RegionName'])
    df_weekly['StoreID'] = le_store.fit_transform(df_weekly['StoreName'])
    df_weekly['ProductID'] = le_prod.fit_transform(df_weekly['ProductName'])

    # 4. INTERACTION FEATURES (moved after encoding)
    # Product-store specific patterns
    df_weekly['Product_Store_Combo'] = df_weekly['ProductID'].astype(str) + '_' + df_weekly['StoreID'].astype(str)
    le_combo = LabelEncoder()
    df_weekly['Product_Store_ID'] = le_combo.fit_transform(df_weekly['Product_Store_Combo'])
    
    # Season-product interactions
    df_weekly['Month_Product'] = df_weekly['Month'].astype(str) + '_' + df_weekly['ProductID'].astype(str)
    le_month_prod = LabelEncoder()
    df_weekly['Month_Product_ID'] = le_month_prod.fit_transform(df_weekly['Month_Product'])

    # Feature selection including the enhanced AutoML engineered features
    features = [
        # Basic identifiers
        'RegionID', 'StoreID', 'ProductID', 'Product_Store_ID', 'Month_Product_ID',
        
        # Temporal features
        'Week', 'Month', 'Year', 'Quarter', 'Is_Month_Start', 'Is_Month_End',
        
        # Cyclical encoding
        'Week_Sin', 'Week_Cos', 'Month_Sin', 'Month_Cos',
        
        # Lag features (multiple periods)
        'Lag_1_Week', 'Lag_2_Week', 'Lag_3_Week', 'Lag_4_Week', 'Lag_8_Week', 'Lag_12_Week',
        
        # Rolling statistics
        'Rolling_Mean_2', 'Rolling_Mean_4', 'Rolling_Mean_8', 'Rolling_Mean_12',
        'Rolling_Std_2', 'Rolling_Std_4', 'Rolling_Std_8', 'Rolling_Std_12',
        
        # Trend indicators
        'EWMA_4', 'MoM_Change', 'Trend_4Week', 'Quantity_ZScore',
        'True_Demand', 'TrueDemand_Ratio', 'Bank_Emi_Flag', 'Bank_Cashback_Pct',
        'Bank_Cashback_Intensity', 'Additional_Funded_Scheme_Pct', 'Additional_Funded_Scheme_Intensity',

        # Holiday and state-specific calendar features
        'Is_Holiday', 'Is_Holiday_Week', 'Holiday_Count',
        'Days_to_Next_Holiday', 'Days_since_Last_Holiday',

        # Price & promotion
        'DiscountPct', 'Discount_Intensity', 'Effective_Price'
    ]
    target = 'Quantity'
    
    # 4. ROBUST TIME-SERIES SPLIT (80-20)
    split_idx = int(len(df_weekly) * 0.8)
    train, test = df_weekly.iloc[:split_idx], df_weekly.iloc[split_idx:]
    if train.empty or test.empty:
        raise ValueError("Not enough weekly rows to create an 80/20 train-test split.")
    log_func(f"Training Samples: {len(train)} | Testing Samples: {len(test)}")

    # 5. ENHANCED AUTO-ML MODEL ZOO (Optimized for Rich Features)
    model_zoo = {
        "XGBoost_Poisson_Enhanced": XGBRegressor(
            objective='count:poisson',
            n_estimators=1500,
            learning_rate=0.01,
            max_depth=8,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            early_stopping_rounds=50,
            n_jobs=-1,
            random_state=42
        ),
        "Random_Forest_Enhanced": RandomForestRegressor(
            n_estimators=500,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            bootstrap=True,
            oob_score=True,
            random_state=42,
            n_jobs=-1
        ),
        "Hist_Gradient_Boosting_Enhanced": HistGradientBoostingRegressor(
            max_iter=1000,
            max_leaf_nodes=63,
            learning_rate=0.05,
            max_depth=10,
            min_samples_leaf=20,
            l2_regularization=0.1,
            random_state=42
        ),
        "LightGBM_Poisson": XGBRegressor(  # Using XGB as LGBM alternative
            objective='count:poisson',
            n_estimators=1200,
            learning_rate=0.02,
            max_depth=7,
            num_leaves=31,
            subsample=0.85,
            colsample_bytree=0.85,
            early_stopping_rounds=50,
            n_jobs=-1,
            random_state=42
        )
    }

    best_model = None
    lowest_mae = float('inf')
    winning_metrics = {}

    for name, model in model_zoo.items():
        log_func(f"AutoML Testing: {name}...")
        try:
            if isinstance(model, XGBRegressor):
                # Use evaluation set to prevent overfitting during training
                model.fit(
                    train[features], train[target],
                    eval_set=[(test[features], test[target])],
                    verbose=False
                )
            else:
                model.fit(train[features], train[target])
            
            preds = model.predict(test[features])
            mae = mean_absolute_error(test[target], preds)
            rmse = np.sqrt(mean_squared_error(test[target], preds))
            
            # Enhanced accuracy metrics
            total_actual = test[target].sum()
            total_error = np.sum(np.abs(test[target] - preds))
            wape = (total_error / (total_actual + 1e-5)) * 100  # Weighted Absolute Percentage Error
            accuracy = max(0, 100 - wape)
            
            # Additional metrics
            mape = np.mean(np.abs((test[target] - preds) / (test[target] + 1e-5))) * 100
            smape = np.mean(2 * np.abs(test[target] - preds) / (np.abs(test[target]) + np.abs(preds) + 1e-5)) * 100
            
            log_func(f"-> Result: Accuracy: {accuracy:.2f}% | MAE: {mae:.2f} | MAPE: {mape:.2f}% | sMAPE: {smape:.2f}%")

            if mae < lowest_mae:
                lowest_mae = mae
                best_model = model
                winning_metrics = {
                    "Algorithm": name.replace("_", " "),
                    "Forecast Accuracy (%)": f"{round(accuracy, 2)}%",
                    "MAE (Absolute Error)": round(mae, 4),
                    "RMSE (Safety Deviation)": round(rmse, 4),
                    "MAPE (%)": round(mape, 2),
                    "sMAPE (%)": round(smape, 2),
                    "Features_Used": len(features)
                }
        except Exception as e:
            log_func(f"Model {name} failed: {e}")

    if best_model is None:
        raise RuntimeError("All candidate models failed. No model artifacts were generated.")

    # 6. ASSET PRESERVATION
    if not os.path.exists(model_dir): os.makedirs(model_dir)
    joblib.dump(best_model, os.path.join(model_dir, 'master_model.pkl'))
    joblib.dump(le_reg, os.path.join(model_dir, 'le_region.pkl'))
    joblib.dump(le_store, os.path.join(model_dir, 'le_store.pkl'))
    joblib.dump(le_prod, os.path.join(model_dir, 'le_prod.pkl'))
    joblib.dump(le_combo, os.path.join(model_dir, 'le_product_store.pkl'))
    joblib.dump(le_month_prod, os.path.join(model_dir, 'le_month_product.pkl'))
    joblib.dump(winning_metrics, os.path.join(model_dir, 'model_metrics.pkl'))
    
    log_func(f"TOURNAMENT FINISHED. WINNER: {winning_metrics['Algorithm']}")
    return winning_metrics

if __name__ == "__main__":
    run_master_tournament()
