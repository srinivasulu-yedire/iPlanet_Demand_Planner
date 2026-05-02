import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import holidays
import os
import logging
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration: Enable/disable features
FEATURE_CONFIG = {
    'holidays': True,
    'seasonal': True,
    'economic': True,  # Will check for economic_indicators.xlsx
    'weather': True,   # Will check for weather_data.xlsx
    'competitor': True, # Will check for competitor_data.xlsx
    'marketing': True, # Will check for digital_marketing.xlsx
    'supply_chain': True, # Will check for supply_chain.xlsx
    'custom_bank_offers': True,  # Custom feature
    'custom_discounts': True,    # Custom feature
    'custom_promotions': True,   # Custom feature
}

STATE_CODE_MAP = {
    'AndhraPradesh': 'AP', 'ArunachalPradesh': 'AR', 'Assam': 'AS', 'Bihar': 'BR',
    'Chhattisgarh': 'CG', 'Goa': 'GA', 'Gujarat': 'GJ', 'Haryana': 'HR',
    'HimachalPradesh': 'HP', 'JammuAndKashmir': 'JK', 'Jharkhand': 'JH',
    'Karnataka': 'KA', 'Kerala': 'KL', 'MadhyaPradesh': 'MP', 'Maharashtra': 'MH',
    'Manipur': 'MN', 'Meghalaya': 'ML', 'Mizoram': 'MZ', 'Nagaland': 'NL',
    'Odisha': 'OR', 'Punjab': 'PB', 'Rajasthan': 'RJ', 'Sikkim': 'SK',
    'TamilNadu': 'TN', 'Telangana': 'TG', 'Tripura': 'TR', 'UttarPradesh': 'UP',
    'Uttarakhand': 'UK', 'WestBengal': 'WB', 'AndamanAndNicobarIslands': 'AN',
    'Chandigarh': 'CH', 'DadraAndNagarHaveliAndDamanAndDiu': 'DN',
    'Delhi': 'DL', 'Lakshadweep': 'LD', 'Puducherry': 'PY'
}


def normalize_state_value(state_value):
    if pd.isna(state_value):
        return None
    raw = str(state_value).strip()
    if len(raw) == 2 and raw.isalpha():
        return raw.upper()
    cleaned = re.sub(r'[^A-Za-z]', '', raw).title()
    return STATE_CODE_MAP.get(cleaned, cleaned)


def get_india_holiday_calendar(years, state_code=None):
    years = sorted(set(int(y) for y in years if pd.notna(y)))
    if not years:
        years = [datetime.now().year]
    kwargs = {'years': years}
    if state_code:
        for arg in ('prov', 'state', 'subdiv'):
            try:
                return holidays.India(**{**kwargs, arg: state_code})
            except TypeError:
                continue
            except Exception:
                continue
    return holidays.India(**kwargs)


def get_state_holiday_list(state_value, years=None):
    state_code = normalize_state_value(state_value)
    if years is None:
        years = [datetime.now().year]
    holiday_calendar = get_india_holiday_calendar(years, state_code)
    holiday_items = [
        {'Date': d, 'Holiday': holiday_calendar.get(d)}
        for d in sorted(holiday_calendar)
    ]
    return pd.DataFrame(holiday_items)


def add_external_features(df_weekly, region_column='RegionName'):
    """
    Add external features that can improve forecast accuracy.
    Features are optional and will be skipped if data files are missing.
    """

    logger.info("Starting external feature engineering...")

    # 1. HOLIDAY FEATURES (Always available - no external file needed)
    if FEATURE_CONFIG.get('holidays', True):
        try:
            if region_column not in df_weekly.columns:
                raise ValueError(f"Region column '{region_column}' not found in data")

            df_weekly['Holiday_State_Code'] = df_weekly[region_column].apply(normalize_state_value)
            df_weekly['Is_Holiday'] = 0
            df_weekly['Is_Holiday_Week'] = 0
            df_weekly['Holiday_Count'] = 0
            df_weekly['Days_to_Next_Holiday'] = 0
            df_weekly['Days_since_Last_Holiday'] = 0

            calendars = {}
            for state_code in df_weekly['Holiday_State_Code'].fillna('').unique():
                calendars[state_code] = get_india_holiday_calendar(df_weekly['Year'].unique(), state_code or None)

            for idx, row in df_weekly.iterrows():
                date = row['BillDate'].date()
                state_code = row['Holiday_State_Code'] or None
                holiday_calendar = calendars.get(state_code)
                holiday_dates = sorted(list(holiday_calendar)) if holiday_calendar is not None else []

                week_end = date
                week_start = week_end - timedelta(days=6)
                holidays_in_week = [d for d in holiday_dates if week_start <= d <= week_end]

                df_weekly.at[idx, 'Is_Holiday'] = int(week_end in holiday_dates)
                df_weekly.at[idx, 'Is_Holiday_Week'] = int(len(holidays_in_week) > 0)
                df_weekly.at[idx, 'Holiday_Count'] = len(holidays_in_week)

                next_holiday = min([d for d in holiday_dates if d > date], default=date + timedelta(days=365))
                last_holiday = max([d for d in holiday_dates if d < date], default=date - timedelta(days=365))

                df_weekly.at[idx, 'Days_to_Next_Holiday'] = (next_holiday - date).days
                df_weekly.at[idx, 'Days_since_Last_Holiday'] = (date - last_holiday).days

            logger.info("✓ Holiday features added")
        except Exception as e:
            logger.warning(f"✗ Holiday features failed: {e}")

    # 2. SEASONAL EVENTS (Always available)
    if FEATURE_CONFIG.get('seasonal', True):
        try:
            df_weekly['Is_Festival_Season'] = df_weekly['Month'].isin([10, 11, 12]).astype(int)
            df_weekly['Is_Summer_Season'] = df_weekly['Month'].isin([3, 4, 5]).astype(int)
            df_weekly['Is_Monsoon_Season'] = df_weekly['Month'].isin([6, 7, 8, 9]).astype(int)
            logger.info("✓ Seasonal features added")
        except Exception as e:
            logger.warning(f"✗ Seasonal features failed: {e}")

    # 3. ECONOMIC INDICATORS (Optional - requires economic_indicators.xlsx)
    if FEATURE_CONFIG.get('economic', True):
        economic_file = 'data/economic_indicators.xlsx'
        if os.path.exists(economic_file):
            try:
                economic_df = pd.read_excel(economic_file, engine='openpyxl')
                economic_df['Date'] = pd.to_datetime(economic_df['Date'])

                # Merge on Year and Month
                df_weekly = df_weekly.merge(
                    economic_df[['Year', 'Month', 'GDP_Growth_Rate', 'Consumer_Confidence_Index',
                               'Inflation_Rate', 'Unemployment_Rate', 'Interest_Rate']],
                    on=['Year', 'Month'],
                    how='left'
                )

                # Fill missing values with column means
                for col in ['GDP_Growth_Rate', 'Consumer_Confidence_Index', 'Inflation_Rate',
                           'Unemployment_Rate', 'Interest_Rate']:
                    if col in df_weekly.columns:
                        df_weekly[col] = df_weekly[col].fillna(df_weekly[col].mean())

                logger.info("✓ Economic indicators added")
            except Exception as e:
                logger.warning(f"✗ Economic indicators failed: {e}")
        else:
            logger.info("○ Economic indicators skipped (file not found)")

    # 4. WEATHER IMPACT (Optional - requires weather_data.xlsx)
    if FEATURE_CONFIG.get('weather', True):
        weather_file = 'data/weather_data.xlsx'
        if os.path.exists(weather_file):
            try:
                weather_df = pd.read_excel(weather_file, engine='openpyxl')
                weather_df['Date'] = pd.to_datetime(weather_df['Date'])

                df_weekly = df_weekly.merge(
                    weather_df[['Year', 'Month', 'Avg_Temperature_C', 'Total_Rainfall_mm',
                               'Humidity_Percent', 'Weather_Index']],
                    on=['Year', 'Month'],
                    how='left'
                )

                # Fill missing values
                for col in ['Avg_Temperature_C', 'Total_Rainfall_mm', 'Humidity_Percent', 'Weather_Index']:
                    if col in df_weekly.columns:
                        df_weekly[col] = df_weekly[col].fillna(df_weekly[col].mean())

                logger.info("✓ Weather features added")
            except Exception as e:
                logger.warning(f"✗ Weather features failed: {e}")
        else:
            logger.info("○ Weather features skipped (file not found)")

    # 5. COMPETITOR ACTIVITY (Optional - requires competitor_data.xlsx)
    if FEATURE_CONFIG.get('competitor', True):
        competitor_file = 'data/competitor_data.xlsx'
        if os.path.exists(competitor_file):
            try:
                competitor_df = pd.read_excel(competitor_file, engine='openpyxl')
                competitor_df['Date'] = pd.to_datetime(competitor_df['Date'])

                # Aggregate competitor activity by week
                competitor_weekly = competitor_df.groupby(['Year', 'Week']).agg({
                    'Promotion_Intensity': 'mean',
                    'Price_Undercut_Percent': 'mean',
                    'New_Store_Opening': 'max'
                }).reset_index()

                competitor_weekly = competitor_weekly.rename(columns={
                    'Promotion_Intensity': 'Competitor_Promotion_Intensity',
                    'Price_Undercut_Percent': 'Competitor_Price_Undercut',
                    'New_Store_Opening': 'Competitor_New_Store_Opening'
                })

                df_weekly = df_weekly.merge(competitor_weekly, on=['Year', 'Week'], how='left')

                # Fill missing values
                for col in ['Competitor_Promotion_Intensity', 'Competitor_Price_Undercut', 'Competitor_New_Store_Opening']:
                    if col in df_weekly.columns:
                        df_weekly[col] = df_weekly[col].fillna(0)

                logger.info("✓ Competitor features added")
            except Exception as e:
                logger.warning(f"✗ Competitor features failed: {e}")
        else:
            logger.info("○ Competitor features skipped (file not found)")

    # 6. DIGITAL MARKETING IMPACT (Optional - requires digital_marketing.xlsx)
    if FEATURE_CONFIG.get('marketing', True):
        marketing_file = 'data/digital_marketing.xlsx'
        if os.path.exists(marketing_file):
            try:
                marketing_df = pd.read_excel(marketing_file, engine='openpyxl')
                marketing_df['Date'] = pd.to_datetime(marketing_df['Date'])

                # Aggregate marketing activity by week
                marketing_weekly = marketing_df.groupby(['Year', 'Week']).agg({
                    'Ad_Spend_INR': 'sum',
                    'Impressions': 'sum',
                    'Clicks': 'sum',
                    'Conversions': 'sum',
                    'Campaign_Intensity': 'mean'
                }).reset_index()

                df_weekly = df_weekly.merge(marketing_weekly, on=['Year', 'Week'], how='left')

                # Fill missing values
                for col in ['Ad_Spend_INR', 'Impressions', 'Clicks', 'Conversions', 'Campaign_Intensity']:
                    if col in df_weekly.columns:
                        df_weekly[col] = df_weekly[col].fillna(0)

                logger.info("✓ Digital marketing features added")
            except Exception as e:
                logger.warning(f"✗ Digital marketing features failed: {e}")
        else:
            logger.info("○ Digital marketing features skipped (file not found)")

    # 7. SUPPLY CHAIN FACTORS (Optional - requires supply_chain.xlsx)
    if FEATURE_CONFIG.get('supply_chain', True):
        supply_file = 'data/supply_chain.xlsx'
        if os.path.exists(supply_file):
            try:
                supply_df = pd.read_excel(supply_file, engine='openpyxl')
                supply_df['Date'] = pd.to_datetime(supply_df['Date'])

                # Aggregate supply chain metrics by week and product
                supply_weekly = supply_df.groupby(['Year', 'Week', 'Product_ID']).agg({
                    'Lead_Time_Days': 'mean',
                    'On_Time_Delivery_Rate': 'mean',
                    'Quality_Score': 'mean',
                    'Supplier_Reliability_Index': 'mean',
                    'Stockout_Events': 'sum'
                }).reset_index()

                # For now, take overall averages (can be enhanced to product-specific)
                supply_overall = supply_weekly.groupby(['Year', 'Week']).agg({
                    'Lead_Time_Days': 'mean',
                    'On_Time_Delivery_Rate': 'mean',
                    'Quality_Score': 'mean',
                    'Supplier_Reliability_Index': 'mean',
                    'Stockout_Events': 'mean'
                }).reset_index()

                df_weekly = df_weekly.merge(supply_overall, on=['Year', 'Week'], how='left')

                # Fill missing values
                for col in ['Lead_Time_Days', 'On_Time_Delivery_Rate', 'Quality_Score',
                           'Supplier_Reliability_Index', 'Stockout_Events']:
                    if col in df_weekly.columns:
                        df_weekly[col] = df_weekly[col].fillna(df_weekly[col].mean())

                logger.info("✓ Supply chain features added")
            except Exception as e:
                logger.warning(f"✗ Supply chain features failed: {e}")
        else:
            logger.info("○ Supply chain features skipped (file not found)")

    # ===== CUSTOM FEATURES SECTION =====
    # Add your custom features here

    # 8. BANK OFFERS (Custom - requires bank_offers.xlsx)
    if FEATURE_CONFIG.get('custom_bank_offers', True):
        bank_file = 'data/bank_offers.xlsx'
        if os.path.exists(bank_file):
            try:
                bank_df = pd.read_excel(bank_file, engine='openpyxl')
                bank_df['Date'] = pd.to_datetime(bank_df['Date'])

                # Example: Merge bank offer data
                # Customize this based on your bank_offers.xlsx structure
                df_weekly = df_weekly.merge(
                    bank_df[['Year', 'Week', 'Bank_Offer_Intensity', 'Card_Promotion_Level']],
                    on=['Year', 'Week'],
                    how='left'
                )

                # Fill missing values
                for col in ['Bank_Offer_Intensity', 'Card_Promotion_Level']:
                    if col in df_weekly.columns:
                        df_weekly[col] = df_weekly[col].fillna(0)

                logger.info("✓ Custom bank offer features added")
            except Exception as e:
                logger.warning(f"✗ Custom bank offer features failed: {e}")
        else:
            logger.info("○ Custom bank offer features skipped (file not found)")

    # 9. CUSTOM DISCOUNTS (Custom - requires custom_discounts.xlsx)
    if FEATURE_CONFIG.get('custom_discounts', True):
        discount_file = 'data/custom_discounts.xlsx'
        if os.path.exists(discount_file):
            try:
                discount_df = pd.read_excel(discount_file, engine='openpyxl')
                discount_df['Date'] = pd.to_datetime(discount_df['Date'])

                # Example: Merge custom discount data
                df_weekly = df_weekly.merge(
                    discount_df[['Year', 'Week', 'Store_Discount_Level', 'Loyalty_Discount_Available']],
                    on=['Year', 'Week'],
                    how='left'
                )

                # Fill missing values
                for col in ['Store_Discount_Level', 'Loyalty_Discount_Available']:
                    if col in df_weekly.columns:
                        df_weekly[col] = df_weekly[col].fillna(0)

                logger.info("✓ Custom discount features added")
            except Exception as e:
                logger.warning(f"✗ Custom discount features failed: {e}")
        else:
            logger.info("○ Custom discount features skipped (file not found)")

    # 10. CUSTOM PROMOTIONS (Custom - requires custom_promotions.xlsx)
    if FEATURE_CONFIG.get('custom_promotions', True):
        promo_file = 'data/custom_promotions.xlsx'
        if os.path.exists(promo_file):
            try:
                promo_df = pd.read_excel(promo_file, engine='openpyxl')
                promo_df['Date'] = pd.to_datetime(promo_df['Date'])

                # Example: Merge custom promotion data
                df_weekly = df_weekly.merge(
                    promo_df[['Year', 'Week', 'Promotion_Type_Code', 'Promotion_Discount_Pct']],
                    on=['Year', 'Week'],
                    how='left'
                )

                # Fill missing values
                for col in ['Promotion_Type_Code', 'Promotion_Discount_Pct']:
                    if col in df_weekly.columns:
                        df_weekly[col] = df_weekly[col].fillna(0)

                logger.info("✓ Custom promotion features added")
            except Exception as e:
                logger.warning(f"✗ Custom promotion features failed: {e}")
        else:
            logger.info("○ Custom promotion features skipped (file not found)")

    # ===== ADD YOUR OWN CUSTOM FEATURES BELOW =====
    # Copy the pattern above to add new feature types

    logger.info(f"External feature engineering completed. Final shape: {df_weekly.shape}")
    return df_weekly

def add_product_category_features(df_weekly):
    """
    Add product-specific features based on categories.
    This would require product categorization data.
    """

    # Placeholder for product category features
    # In real implementation, you'd have a product master with categories

    # Product lifecycle stage (introduction, growth, maturity, decline)
    df_weekly['Product_Lifecycle_Stage'] = np.random.choice(['Introduction', 'Growth', 'Maturity', 'Decline'],
                                                           size=len(df_weekly), p=[0.1, 0.3, 0.5, 0.1])

    # Encode lifecycle stages
    lifecycle_map = {'Introduction': 0, 'Growth': 1, 'Maturity': 2, 'Decline': 3}
    df_weekly['Lifecycle_Code'] = df_weekly['Product_Lifecycle_Stage'].map(lifecycle_map)

    # Product seasonality (some products sell better in certain seasons)
    df_weekly['Product_Seasonality_Index'] = np.random.uniform(0.5, 1.5, len(df_weekly))

    return df_weekly

def add_store_characteristics(df_weekly):
    """
    Add store-specific features that affect demand.
    """

    # Store size/traffic proxy
    df_weekly['Store_Size_Category'] = np.random.choice(['Small', 'Medium', 'Large'], size=len(df_weekly))

    # Store location characteristics
    df_weekly['Store_Location_Type'] = np.random.choice(['Downtown', 'Mall', 'Street', 'Online'], size=len(df_weekly))

    # Foot traffic proxy
    df_weekly['Foot_Traffic_Index'] = np.random.uniform(0.3, 1.0, len(df_weekly))

    # Store performance history
    df_weekly['Store_Performance_Rating'] = np.random.uniform(0.6, 1.0, len(df_weekly))

    return df_weekly

# Usage example:
# df_enhanced = add_external_features(df_weekly)
# df_enhanced = add_product_category_features(df_enhanced)
# df_enhanced = add_store_characteristics(df_enhanced)