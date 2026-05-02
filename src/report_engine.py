import pandas as pd
import numpy as np
import datetime
import io
import re

# Import GenAI layer for enhanced reporting
try:
    from genai_layer import create_executive_summary, initialize_genai
    genai_available = True
except ImportError:
    genai_available = False

try:
    from external_features import add_external_features
    external_features_available = True
except ImportError:
    external_features_available = False

def extract_first_number(value):
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


def get_last_known_metrics(df, store_name, product_name, cutoff_date=None):
    """Fetches real historical sales to anchor the AI forecast."""
    # Filter history for this specific product/store
    history = df[(df['StoreName'] == store_name) & (df['ProductName'] == product_name)].copy()
    
    if history.empty:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    
    # Ensure date format is correct for calculation
    history['BillDate'] = pd.to_datetime(history['BillDate'])
    
    # Filter data up to cutoff date if provided
    if cutoff_date is not None:
        cutoff_date = pd.to_datetime(cutoff_date)
        history = history[history['BillDate'] <= cutoff_date]
    
    if history.empty:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    if 'True Demand' not in history.columns:
        history['True Demand'] = 0.0
    if 'Bank Funded NoCost EMI Applied With BankName' in history.columns:
        history['Bank_Emi_Flag'] = history['Bank Funded NoCost EMI Applied With BankName'].apply(parse_emi_flag)
    else:
        history['Bank_Emi_Flag'] = 0.0
    if 'Bank Funded Cashback Coverted To Percentage With BankName' in history.columns:
        history['Bank_Cashback_Pct'] = history['Bank Funded Cashback Coverted To Percentage With BankName'].apply(extract_first_number)
    else:
        history['Bank_Cashback_Pct'] = 0.0
    if 'Additional Apple Or Distributor Funded Scheme In Percentage' in history.columns:
        history['Additional_Funded_Scheme_Pct'] = history['Additional Apple Or Distributor Funded Scheme In Percentage'].apply(extract_first_number)
    else:
        history['Additional_Funded_Scheme_Pct'] = 0.0
    
    # Aggregate to weekly to find the most recent 'trend'
    weekly = history.set_index('BillDate').resample('W').agg({
        'Quantity': 'sum',
        'True Demand': 'sum',
        'Bank_Emi_Flag': 'mean',
        'Bank_Cashback_Pct': 'mean',
        'Additional_Funded_Scheme_Pct': 'mean'
    }).tail(4)
    
    last_week_qty = float(weekly['Quantity'].iloc[-1]) if len(weekly) >= 1 else 0.0
    rolling_4_avg = float(weekly['Quantity'].mean()) if len(weekly) >= 1 else 0.0
    last_true_demand = float(weekly['True Demand'].iloc[-1]) if len(weekly) >= 1 else 0.0
    last_emi_ratio = float(weekly['Bank_Emi_Flag'].iloc[-1]) if len(weekly) >= 1 else 0.0
    last_cashback_pct = float(weekly['Bank_Cashback_Pct'].iloc[-1]) if len(weekly) >= 1 else 0.0
    last_additional_pct = float(weekly['Additional_Funded_Scheme_Pct'].iloc[-1]) if len(weekly) >= 1 else 0.0
    
    return last_week_qty, rolling_4_avg, last_true_demand, last_emi_ratio, last_cashback_pct, last_additional_pct

def generate_single_forecast_csv(region, store, product_name, labels, predictions, start_date=None):
    """Generates detailed individual report with Region, Store, and Week windows."""
    if start_date is None:
        start_date = datetime.date.today()
    else:
        start_date = pd.to_datetime(start_date).date()
        
    rows = []
    for i, pred in enumerate(predictions):
        week_start = start_date + datetime.timedelta(weeks=i)
        week_end = week_start + datetime.timedelta(days=6)
        rows.append({
            'Region': region, 'Store': store, 'Product': product_name,
            'Forecast_Start_Date': start_date.strftime('%Y-%m-%d'),
            'Week_Starting': week_start.strftime('%Y-%m-%d'),
            'Week_Ending': week_end.strftime('%Y-%m-%d'),
            'Forecasted_Units': int(round(pred))
        })
    return pd.DataFrame(rows).to_csv(index=False).encode('utf-8')

def generate_bulk_report(df_meta, model, encoders, selected_region, selected_store, cutoff_date=None):
    """Generates a master store report with 30 and 60 day horizons."""
    le_reg, le_store, le_prod = encoders[:3]
    le_combo = encoders[3] if len(encoders) > 3 else None
    le_month_prod = encoders[4] if len(encoders) > 4 else None
    bulk_results = []
    all_products = df_meta[df_meta['StoreName'] == selected_store]['ProductName'].unique()
    start_date = pd.to_datetime(cutoff_date).date() if cutoff_date is not None else datetime.date.today()
    
    for product in all_products:
        try:
            # Anchor the bulk predictions with real data
            last_q, roll_q, last_true_demand, last_emi_ratio, last_cashback_pct, last_add_pct = get_last_known_metrics(
                df_meta, selected_store, product, cutoff_date
            )
            
            p_id = le_prod.transform([product])[0]
            s_id = le_store.transform([selected_store])[0]
            r_id = le_reg.transform([selected_region])[0]
            
            # Predict 9 weeks with the same enhanced feature set as training
            num_weeks = 9
            future_df = pd.DataFrame({
                'RegionID': [r_id] * num_weeks,
                'StoreID': [s_id] * num_weeks,
                'ProductID': [p_id] * num_weeks,
                'RegionName': [selected_region] * num_weeks,
                'BillDate': [(start_date + datetime.timedelta(weeks=i)) for i in range(num_weeks)],
                'Week': [(start_date + datetime.timedelta(weeks=i)).isocalendar()[1] for i in range(num_weeks)],
                'Month': [(start_date + datetime.timedelta(weeks=i)).month for i in range(num_weeks)],
                'Year': [(start_date + datetime.timedelta(weeks=i)).year for i in range(num_weeks)],
                'DiscountPct': [0.05] * num_weeks,
                'Lag_1_Week': [last_q] * num_weeks,
                'Lag_2_Week': [last_q] * num_weeks,
                'Lag_3_Week': [last_q] * num_weeks,
                'Lag_4_Week': [last_q] * num_weeks,
                'Lag_8_Week': [last_q] * num_weeks,
                'Lag_12_Week': [last_q] * num_weeks,
                'Rolling_Mean_2': [roll_q] * num_weeks,
                'Rolling_Mean_4': [roll_q] * num_weeks,
                'Rolling_Mean_8': [roll_q] * num_weeks,
                'Rolling_Mean_12': [roll_q] * num_weeks,
                'Rolling_Std_2': [roll_q * 0.15] * num_weeks,
                'Rolling_Std_4': [roll_q * 0.15] * num_weeks,
                'Rolling_Std_8': [roll_q * 0.15] * num_weeks,
                'Rolling_Std_12': [roll_q * 0.15] * num_weeks,
                'True_Demand': [last_true_demand] * num_weeks,
                'TrueDemand_Ratio': [last_true_demand / (last_q + 1e-5)] * num_weeks,
                'Bank_Emi_Flag': [last_emi_ratio] * num_weeks,
                'Bank_Cashback_Pct': [last_cashback_pct] * num_weeks,
                'Additional_Funded_Scheme_Pct': [last_add_pct] * num_weeks,
            })
            future_df['BillDate'] = pd.to_datetime(future_df['BillDate'])
            
            future_df['Quarter'] = future_df['Month'].apply(lambda x: (x - 1) // 3 + 1)
            future_df['Is_Month_Start'] = [(start_date + datetime.timedelta(weeks=i)).day <= 7 for i in range(num_weeks)]
            future_df['Is_Month_End'] = [(start_date + datetime.timedelta(weeks=i)).day >= 25 for i in range(num_weeks)]
            future_df['Week_Sin'] = np.sin(2 * np.pi * future_df['Week'] / 52.0)
            future_df['Week_Cos'] = np.cos(2 * np.pi * future_df['Week'] / 52.0)
            future_df['Month_Sin'] = np.sin(2 * np.pi * future_df['Month'] / 12.0)
            future_df['Month_Cos'] = np.cos(2 * np.pi * future_df['Month'] / 12.0)

            if external_features_available:
                try:
                    future_df = add_external_features(future_df, region_column='RegionName')
                except Exception:
                    pass

            future_df['EWMA_4'] = [roll_q] * num_weeks
            future_df['MoM_Change'] = [0.0] * num_weeks
            future_df['Trend_4Week'] = [0.0] * num_weeks
            future_df['Quantity_ZScore'] = [0.0] * num_weeks
            future_df['Discount_Intensity'] = pd.cut(future_df['DiscountPct'], bins=[0, 0.05, 0.15, 0.3, 1.0], labels=[0, 1, 2, 3]).fillna(0).astype(int)
            future_df['Effective_Price'] = [100.0] * num_weeks
            future_df['Bank_Cashback_Intensity'] = pd.cut(future_df['Bank_Cashback_Pct'], bins=[-1, 0, 2, 5, 10, 100], labels=[0, 1, 2, 3, 4]).fillna(0).astype(int)
            future_df['Additional_Funded_Scheme_Intensity'] = pd.cut(future_df['Additional_Funded_Scheme_Pct'], bins=[-1, 0, 2, 5, 10, 100], labels=[0, 1, 2, 3, 4]).fillna(0).astype(int)
            future_df['Product_Store_Combo'] = future_df['ProductID'].astype(str) + '_' + future_df['StoreID'].astype(str)
            if le_combo is not None:
                try:
                    future_df['Product_Store_ID'] = le_combo.transform(future_df['Product_Store_Combo'])
                except ValueError:
                    future_df['Product_Store_ID'] = [0] * num_weeks
            else:
                future_df['Product_Store_ID'] = [0] * num_weeks
            future_df['Month_Product'] = future_df['Month'].astype(str) + '_' + future_df['ProductID'].astype(str)
            if le_month_prod is not None:
                try:
                    future_df['Month_Product_ID'] = le_month_prod.transform(future_df['Month_Product'])
                except ValueError:
                    future_df['Month_Product_ID'] = [0] * num_weeks
            else:
                future_df['Month_Product_ID'] = [0] * num_weeks
            
            if hasattr(model, 'feature_names_in_'):
                model_features = list(model.feature_names_in_)
            else:
                expected_features = [
                    'RegionID', 'StoreID', 'ProductID', 'Product_Store_ID', 'Month_Product_ID',
                    'Week', 'Month', 'Year', 'Quarter', 'Is_Month_Start', 'Is_Month_End',
                    'Week_Sin', 'Week_Cos', 'Month_Sin', 'Month_Cos',
                    'Lag_1_Week', 'Lag_2_Week', 'Lag_3_Week', 'Lag_4_Week', 'Lag_8_Week', 'Lag_12_Week',
                    'Rolling_Mean_2', 'Rolling_Mean_4', 'Rolling_Mean_8', 'Rolling_Mean_12',
                    'Rolling_Std_2', 'Rolling_Std_4', 'Rolling_Std_8', 'Rolling_Std_12',
                    'EWMA_4', 'MoM_Change', 'Trend_4Week', 'Quantity_ZScore',
                    'True_Demand', 'TrueDemand_Ratio', 'Bank_Emi_Flag', 'Bank_Cashback_Pct',
                    'Bank_Cashback_Intensity', 'Additional_Funded_Scheme_Pct', 'Additional_Funded_Scheme_Intensity',
                    'DiscountPct', 'Discount_Intensity', 'Effective_Price'
                ]
                model_features = [f for f in expected_features if f in future_df.columns]

            for feature in [f for f in model_features if f not in future_df.columns]:
                future_df[feature] = 0

            future_df = future_df[model_features]

            preds = [max(0, p) for p in model.predict(future_df)]
            bulk_results.append({
                'Store': selected_store, 'Product': product,
                'Forecast_Start_Date': start_date.strftime('%Y-%m-%d'),
                '30_Day_Forecast': int(round(sum(preds[:4]))),
                '60_Day_Forecast': int(round(sum(preds[:9]))),
                'Current_Weekly_Velocity': round(last_q, 2)
            })
        except Exception as e:
            bulk_results.append({
                'Store': selected_store, 'Product': product,
                'Forecast_Start_Date': start_date.strftime('%Y-%m-%d'),
                '30_Day_Forecast': 0,
                '60_Day_Forecast': 0,
                'Current_Weekly_Velocity': round(last_q, 2) if 'last_q' in locals() else 0,
                'Status': f'Failed: {e}'
            })
    return pd.DataFrame(bulk_results)

def generate_executive_summary_report(bulk_df, model_metrics, business_context=""):
    """
    Generate an AI-powered executive summary for bulk forecast reports.

    Args:
        bulk_df: DataFrame from generate_bulk_report
        model_metrics: Dictionary with model performance metrics
        business_context: Additional business context

    Returns:
        String containing executive summary
    """
    if not genai_available:
        return "GenAI not available for executive summary generation."

    try:
        # Initialize GenAI if not already done
        initialize_genai(provider="openai")

        # Convert bulk data to format expected by GenAI function
        predictions_data = []
        for _, row in bulk_df.iterrows():
            # Create mock predictions based on available data
            weekly_forecast = row['30_Day_Forecast'] / 4.0  # Approximate weekly
            predictions_data.append({
                'product_name': row['Product'],
                'predictions': [weekly_forecast] * 4,  # 4 weeks of forecast
                'historical_avg': row['Current_Weekly_Velocity']
            })

        return create_executive_summary(predictions_data, model_metrics, business_context)

    except Exception as e:
        return f"Error generating executive summary: {str(e)}"
