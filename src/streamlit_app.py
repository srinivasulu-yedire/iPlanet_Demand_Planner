import streamlit as st
import pandas as pd
import joblib
import os
import datetime
import shutil
import numpy as np
import io
import altair as alt
import json
import hashlib
import time

try:
    from training_pipeline import MODEL_ARTIFACTS, validate_sales_data_columns, run_master_tournament
    training_pipeline_available = True
except ImportError as e:
    training_pipeline_available = False
    training_pipeline_error = e

# Import the dedicated report module
try:
    import report_engine as re
except ImportError:
    st.error("Missing 'report_engine.py' in the src folder.")

# Import holiday feature helper
try:
    from external_features import add_external_features, get_state_holiday_list
    holiday_features_available = True
except ImportError:
    holiday_features_available = False
    st.warning("Holiday feature module not available. Predictions will use default holiday behavior.")

# Import GenAI layer
try:
    from genai_layer import (
        initialize_genai, explain_forecast_accuracy, analyze_forecast_trends,
        generate_inventory_recommendations, create_executive_summary,
        detect_anomalies, get_forecast_insights, format_business_alert
    )
    genai_available = True
except ImportError:
    genai_available = False
    genai_import_error = "GenAI layer not available. AI insights will be limited."

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="iPlanet Demand Planner | Bangalore",
    layout="wide",
    page_icon="🍎",
    initial_sidebar_state="expanded"
)

# --- APP SHELL STYLING ---
st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem;
    }
    [data-testid="stSidebarCollapseButton"] {
        display: none;
    }
    [data-testid="stToolbar"], [data-testid="stDecoration"], #MainMenu, footer {
        visibility: hidden;
        height: 0;
    }
    .app-brand {
        border: 1px solid rgba(49, 51, 63, 0.16);
        border-radius: 10px;
        padding: 0.9rem 0.95rem;
        margin-bottom: 0.9rem;
        background: linear-gradient(135deg, #f8fafc 0%, #eef5f2 100%);
    }
    .app-brand-title {
        font-size: 1.02rem;
        font-weight: 750;
        color: #1f2937;
        line-height: 1.25;
    }
    .app-brand-subtitle {
        font-size: 0.78rem;
        color: #64748b;
        margin-top: 0.25rem;
    }
    .workspace-help {
        border-left: 3px solid #2f855a;
        padding: 0.45rem 0.65rem;
        margin: 0.55rem 0 1rem 0;
        background: #f7fbf8;
        color: #334155;
        font-size: 0.82rem;
    }
    .section-kicker {
        color: #64748b;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.02em;
        text-transform: uppercase;
        margin-bottom: 0.2rem;
    }
    .top-workspace {
        color: #64748b;
        font-size: 0.86rem;
        line-height: 2.4rem;
        white-space: nowrap;
    }
    .top-workspace-label {
        color: #64748b;
        font-size: 1rem;
        font-weight: 700;
        line-height: 2.4rem;
        text-align: center;
        white-space: nowrap;
    }
    .signin-shell {
        min-height: calc(100vh - 7rem);
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 2rem 0;
    }
    .signin-card {
        width: min(380px, 92vw);
        padding: 1.5rem 1.6rem;
        border: 1px solid rgba(255, 255, 255, 0.52);
        border-radius: 14px;
        background: rgba(255, 255, 255, 0.88);
        box-shadow: 0 22px 70px rgba(15, 23, 42, 0.24);
        backdrop-filter: blur(10px);
    }
    .signin-bg {
        position: fixed;
        inset: 0;
        z-index: -1;
        background:
            linear-gradient(90deg, rgba(15, 23, 42, 0.48), rgba(15, 23, 42, 0.10)),
            linear-gradient(180deg, rgba(255,255,255,0.10), rgba(255,255,255,0.18)),
            radial-gradient(circle at 78% 24%, rgba(255, 255, 255, 0.62) 0 3%, transparent 4%),
            linear-gradient(90deg, transparent 0 10%, rgba(255,255,255,0.62) 10% 12%, transparent 12% 22%, rgba(255,255,255,0.52) 22% 24%, transparent 24% 34%, rgba(255,255,255,0.58) 34% 36%, transparent 36% 100%),
            linear-gradient(180deg, #243447 0 18%, #eef2f7 18% 21%, #9fb4c7 21% 56%, #e5e7eb 56% 59%, #1f2937 59% 100%);
        background-size: cover;
    }
    .loading-card {
        width: min(430px, 92vw);
        margin: 28vh auto 0;
        padding: 1.4rem 1.6rem;
        border: 1px solid rgba(49, 51, 63, 0.12);
        border-radius: 12px;
        background: rgba(255, 255, 255, 0.92);
        box-shadow: 0 18px 58px rgba(15, 23, 42, 0.18);
        text-align: center;
    }
    .loading-spinner {
        width: 34px;
        height: 34px;
        margin: 0 auto 0.85rem;
        border: 4px solid rgba(47, 133, 90, 0.18);
        border-top-color: #2f855a;
        border-radius: 50%;
        animation: loading-spin 0.9s linear infinite;
    }
    @keyframes loading-spin {
        to { transform: rotate(360deg); }
    }
    .user-row-header {
        color: #64748b;
        font-size: 0.82rem;
        font-weight: 700;
        padding: 0.45rem 0.25rem;
    }
    .user-table {
        border: 1px solid rgba(49, 51, 63, 0.14);
        border-radius: 8px;
        overflow: hidden;
        margin-top: 0.5rem;
    }
    .user-table-header {
        background: #f8fafc;
        border-bottom: 1px solid rgba(49, 51, 63, 0.14);
    }
    .user-table-row {
        border-bottom: 1px solid rgba(49, 51, 63, 0.10);
        padding: 0.2rem 0;
    }
    .user-table-row:last-child {
        border-bottom: none;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- TRAINING WORKSPACE ---
DATA_PATH = 'data/sales_data.xlsx'
MODEL_DIR = 'models'
TRAINING_UPLOAD_DIR = os.path.join('data', 'training_uploads')
STAGED_MODEL_DIR = os.path.join(MODEL_DIR, '_training_latest')
USER_STORE_PATH = os.path.join('data', 'users.json')
SALES_REQUIRED_COLUMNS = [
    'RegionName', 'StoreName', 'ProductName', 'BillDate',
    'Quantity', 'MRP', 'ProductLevelDiscAmount'
]
ROLES = ["Admin", "Store Operator"]
DEFAULT_ADMIN_USERNAME = "admin"
DEFAULT_ADMIN_PASSWORD = "admin123"


def _hash_password(password):
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def _default_users():
    return {
        DEFAULT_ADMIN_USERNAME: {
            "username": DEFAULT_ADMIN_USERNAME,
            "display_name": "Administrator",
            "role": "Admin",
            "password_hash": _hash_password(DEFAULT_ADMIN_PASSWORD),
            "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        }
    }


def load_users():
    os.makedirs(os.path.dirname(USER_STORE_PATH), exist_ok=True)
    if not os.path.exists(USER_STORE_PATH):
        users = _default_users()
        save_users(users)
        return users

    try:
        with open(USER_STORE_PATH, "r", encoding="utf-8") as user_file:
            users = json.load(user_file)
    except (json.JSONDecodeError, OSError):
        users = {}

    if DEFAULT_ADMIN_USERNAME not in users:
        users.update(_default_users())
        save_users(users)
    return users


def save_users(users):
    os.makedirs(os.path.dirname(USER_STORE_PATH), exist_ok=True)
    with open(USER_STORE_PATH, "w", encoding="utf-8") as user_file:
        json.dump(users, user_file, indent=2)


def authenticate_user(username, password):
    users = load_users()
    user = users.get(username.strip().lower())
    if not user:
        return None
    if user.get("password_hash") != _hash_password(password):
        return None
    return user


def render_login_transition():
    st.markdown('<div class="signin-bg"></div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="loading-card">
            <div class="loading-spinner"></div>
            <div style="font-size: 1.15rem; font-weight: 750; color: #1f2937;">Taking you to iPlanet software hub...</div>
            <div style="font-size: 0.86rem; color: #64748b; margin-top: 0.35rem;">Please wait while your workspace loads.</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    time.sleep(0.8)
    st.session_state.login_transition = False
    st.rerun()


def require_login():
    if "authenticated_user" in st.session_state:
        if st.session_state.get("login_transition"):
            render_login_transition()
        return st.session_state.authenticated_user

    login_placeholder = st.empty()
    with login_placeholder.container():
        st.markdown('<div class="signin-bg"></div>', unsafe_allow_html=True)
        st.markdown('<div style="height: 12vh;"></div>', unsafe_allow_html=True)
        _, login_col, _ = st.columns([1, 0.72, 1])
        with login_col:
            st.title("iPlanet Demand Planner")
            st.caption("Sign in to access forecasting and operations workspaces.")

            with st.form("login_form"):
                username = st.text_input("User", width=280)
                password = st.text_input("Password", type="password", width=280)
                submitted = st.form_submit_button("Sign In", type="primary")

            if submitted:
                user = authenticate_user(username, password)
                if user:
                    st.session_state.authenticated_user = user
                    st.session_state.report_generated = False
                    st.session_state.login_transition = True
                    st.rerun()
                st.error("Invalid username or password.")

            st.info("Default admin login: username `admin`, password `admin123`.")
    st.stop()


@st.dialog("Change Password")
def render_change_password_dialog(selected_username):
    users = load_users()
    user = users.get(selected_username)
    if not user:
        st.error("User was not found.")
        return

    st.caption(f"User: {user.get('display_name', selected_username)} ({selected_username})")
    with st.form(f"change_password_form_{selected_username}", clear_on_submit=True):
        new_password = st.text_input("New Password", type="password", width=280)
        confirm_password = st.text_input("Confirm Password", type="password", width=280)
        password_submitted = st.form_submit_button("Submit", type="primary", width=120)

    if password_submitted:
        if not new_password:
            st.error("New password is required.")
        elif new_password != confirm_password:
            st.error("Password confirmation does not match.")
        else:
            users[selected_username]["password_hash"] = _hash_password(new_password)
            users[selected_username]["updated_at"] = datetime.datetime.now().isoformat(timespec="seconds")
            users[selected_username]["password_changed_at"] = users[selected_username]["updated_at"]
            save_users(users)
            if st.session_state.authenticated_user.get("username") == selected_username:
                st.session_state.authenticated_user = users[selected_username]
            st.success("Password changed.")
            st.rerun()


def render_user_management():
    st.title("User Management")
    st.caption("Create users, assign access roles, and reset passwords.")

    users = load_users()
    with st.form("create_user_form", clear_on_submit=True):
        col_a, col_b, col_c, col_d = st.columns([0.22, 0.22, 0.22, 0.34])
        with col_a:
            username = st.text_input("User", width=220)
        with col_b:
            display_name = st.text_input("Display Name", width=220)
        with col_c:
            role = st.selectbox("Role", ROLES, width=220)
        with col_d:
            password = st.text_input("Temporary Password", type="password", width=220)
            submitted = st.form_submit_button("Save User", type="primary")

    if submitted:
        normalized_username = username.strip().lower()
        if not normalized_username or not password:
            st.error("Username and password are required.")
        elif normalized_username in users:
            st.error(f"User `{normalized_username}` already exists. Choose a unique username.")
        else:
            now = datetime.datetime.now().isoformat(timespec="seconds")
            users[normalized_username] = {
                "username": normalized_username,
                "display_name": display_name.strip() or normalized_username,
                "role": role,
                "password_hash": _hash_password(password),
                "created_at": now,
                "updated_at": now,
                "password_changed_at": now,
            }
            save_users(users)
            st.success(f"Saved user `{normalized_username}` as {role}.")

    rows = [
        {
            "Username": user.get("username", username),
            "Display Name": user.get("display_name", ""),
            "Role": user.get("role", "Store Operator"),
            "Created": user.get("created_at", ""),
            "Last Password Change": user.get("password_changed_at", user.get("updated_at", "")),
        }
        for username, user in sorted(load_users().items(), key=lambda item: item[0])
    ]
    st.subheader("Users")
    if not rows:
        st.info("No users found.")
        return

    st.markdown('<div class="user-table">', unsafe_allow_html=True)
    st.markdown('<div class="user-table-header">', unsafe_allow_html=True)
    header_cols = st.columns([1.1, 1.5, 1.1, 1.5, 1.7, 0.55])
    for col, label in zip(header_cols, ["User", "Display Name", "Role", "Created", "Last Password Change", ""]):
        col.markdown(f'<div class="user-row-header">{label}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    for row in rows:
        st.markdown('<div class="user-table-row">', unsafe_allow_html=True)
        row_cols = st.columns([1.1, 1.5, 1.1, 1.5, 1.7, 0.55], vertical_alignment="center")
        row_cols[0].write(row["Username"])
        row_cols[1].write(row["Display Name"])
        row_cols[2].write(row["Role"])
        row_cols[3].write(row["Created"])
        row_cols[4].write(row["Last Password Change"])
        if row_cols[5].button(
            "",
            key=f"password_{row['Username']}",
            help=f"Change password for {row['Username']}",
            icon=":material/lock_reset:",
            width=44
        ):
            render_change_password_dialog(row["Username"])
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


def _format_date(value):
    if pd.isna(value):
        return "Not available"
    return pd.to_datetime(value).strftime("%d %b %Y")


def build_upload_summary(uploaded_df):
    summary_df = uploaded_df.copy()
    original_rows = len(summary_df)
    duplicate_rows = int(summary_df.duplicated().sum())
    summary_df = summary_df.drop_duplicates().copy()

    summary_df['BillDate'] = pd.to_datetime(summary_df['BillDate'], errors='coerce')
    valid_dates = summary_df['BillDate'].dropna()
    date_min = valid_dates.min() if not valid_dates.empty else pd.NaT
    date_max = valid_dates.max() if not valid_dates.empty else pd.NaT

    uploaded_compare = summary_df[SALES_REQUIRED_COLUMNS].copy()
    uploaded_compare['BillDate'] = pd.to_datetime(uploaded_compare['BillDate'], errors='coerce')
    uploaded_compare = uploaded_compare.drop_duplicates()
    rows_added = len(uploaded_compare)
    if os.path.exists(DATA_PATH):
        try:
            active_compare = pd.read_excel(DATA_PATH, usecols=SALES_REQUIRED_COLUMNS, engine='openpyxl')
            active_compare['BillDate'] = pd.to_datetime(active_compare['BillDate'], errors='coerce')
            active_compare = active_compare.drop_duplicates()
            rows_added = len(
                uploaded_compare.merge(
                    active_compare,
                    how='left',
                    indicator=True
                ).query("_merge == 'left_only'")
            )
        except Exception:
            rows_added = len(uploaded_compare)

    rows_after_dedupe = len(summary_df)

    missing_dates = []
    anomaly_rows = []
    if not valid_dates.empty:
        valid_sales = summary_df.dropna(subset=['BillDate']).copy()
        valid_sales['Quantity'] = pd.to_numeric(valid_sales['Quantity'], errors='coerce').fillna(0)
        daily_quantity = (
            valid_sales
            .groupby(valid_sales['BillDate'].dt.normalize())['Quantity']
            .sum()
            .sort_index()
        )
        all_dates = pd.date_range(date_min.normalize(), date_max.normalize(), freq='D')
        missing_dates = [date for date in all_dates.difference(daily_quantity.index)]

        daily_change = daily_quantity.diff()
        previous_quantity = daily_quantity.shift(1)
        pct_change = daily_quantity.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        change_threshold = max(daily_change.abs().quantile(0.90), daily_quantity.mean() * 0.50, 1)
        anomaly_mask = (daily_change.abs() >= change_threshold) & (pct_change.abs() >= 0.50)
        anomaly_rows = [
            {
                "Date": index.strftime("%d %b %Y"),
                "Type": "Spike" if daily_change.loc[index] > 0 else "Drop",
                "Quantity": int(daily_quantity.loc[index]),
                "Previous": int(previous_quantity.loc[index]) if not pd.isna(previous_quantity.loc[index]) else 0,
                "Change": int(daily_change.loc[index]) if not pd.isna(daily_change.loc[index]) else 0,
            }
            for index in daily_quantity.index[anomaly_mask]
        ]

    return {
        "data": summary_df,
        "original_rows": original_rows,
        "rows_after_dedupe": rows_after_dedupe,
        "rows_added": rows_added,
        "duplicates_removed": duplicate_rows,
        "date_min": date_min,
        "date_max": date_max,
        "missing_dates": missing_dates,
        "anomalies": anomaly_rows,
    }


def apply_retraining_strategy(upload_summary, strategy):
    training_df = upload_summary["data"].copy()
    cutoff_date = None
    if strategy == "Rolling window (last 2-3 years)" and not pd.isna(upload_summary["date_max"]):
        cutoff_date = upload_summary["date_max"] - pd.DateOffset(years=3)
        training_df = training_df[training_df['BillDate'] >= cutoff_date].copy()
    return training_df, cutoff_date


def calculate_safety_stock(history_df, forecast_values, horizon_days, lead_time_days, service_factor=1.65):
    avg_daily_forecast = sum(forecast_values) / horizon_days if horizon_days else 0
    fallback_sigma = avg_daily_forecast * 0.15

    if history_df.empty:
        return round(service_factor * fallback_sigma * np.sqrt(lead_time_days)), fallback_sigma

    history = history_df.copy()
    history['BillDate'] = pd.to_datetime(history['BillDate'], errors='coerce')
    history['Quantity'] = pd.to_numeric(history['Quantity'], errors='coerce').fillna(0)
    history = history.dropna(subset=['BillDate']).sort_values('BillDate')
    if history.empty:
        return round(service_factor * fallback_sigma * np.sqrt(lead_time_days)), fallback_sigma

    max_history_date = history['BillDate'].max()
    recent_start = max_history_date - pd.Timedelta(days=180)
    recent_history = history[history['BillDate'] >= recent_start]
    daily_quantity = recent_history.set_index('BillDate').resample('D')['Quantity'].sum()
    historical_sigma = float(daily_quantity.std(ddof=0)) if len(daily_quantity) > 1 else 0.0
    sigma = max(historical_sigma, fallback_sigma)

    return round(service_factor * sigma * np.sqrt(lead_time_days)), sigma


def render_workspace_selector(current_user):
    st.sidebar.markdown(
        """
        <div class="app-brand">
            <div class="app-brand-title">iPlanet Demand Planner</div>
            <div class="app-brand-subtitle">Demand planning</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    shell_cols = st.columns([0.34, 0.10, 0.30, 0.17, 0.09], vertical_alignment="center")
    shell_cols[3].markdown(
        f"<div class=\"top-workspace\">Signed in as {current_user.get('display_name', current_user['username'])} | {current_user['role']}</div>",
        unsafe_allow_html=True
    )
    if shell_cols[4].button("Sign Out", width=96):
        st.session_state.pop("authenticated_user", None)
        st.session_state.pop("login_transition", None)
        st.session_state.report_generated = False
        st.rerun()

    options = ["Prediction Dashboard"]
    if current_user["role"] == "Admin":
        options.extend(["Model Training", "User Management"])
    else:
        return "Prediction Dashboard"

    labels = {
        "Prediction Dashboard": "Prediction",
        "Model Training": "Training",
        "User Management": "Settings"
    }
    if hasattr(st.sidebar, "segmented_control"):
        shell_cols[1].markdown('<div class="top-workspace-label">Workspace</div>', unsafe_allow_html=True)
        selected_workspace = shell_cols[2].segmented_control(
            "Workspace",
            options,
            default="Prediction Dashboard",
            format_func=lambda option: labels[option],
            label_visibility="collapsed",
            key="workspace_selector"
        )
    else:
        shell_cols[1].markdown('<div class="top-workspace-label">Workspace</div>', unsafe_allow_html=True)
        selected_workspace = shell_cols[2].radio(
            "Workspace",
            options,
            format_func=lambda option: labels[option],
            label_visibility="collapsed",
            horizontal=True,
            key="workspace_selector"
        )

    return selected_workspace


def render_training_workspace():
    st.title("Model Training")
    st.caption("Upload fresh sales data, run the model tournament, and promote the generated artifacts only after training succeeds.")

    if not training_pipeline_available:
        st.error(f"Training pipeline could not be imported: {training_pipeline_error}")
        return

    step_a, step_b, step_c = st.columns(3)
    with step_a:
        with st.container(border=True):
            st.markdown("**1. Upload**")
            st.write("Choose the latest `sales_data.xlsx` workbook.")
    with step_b:
        with st.container(border=True):
            st.markdown("**2. Validate**")
            st.write("Required sales columns are checked before training.")
    with step_c:
        with st.container(border=True):
            st.markdown("**3. Promote**")
            st.write("New `.pkl` files become active only after success.")

    current_metrics_path = os.path.join(MODEL_DIR, 'model_metrics.pkl')
    model_col, upload_col = st.columns([1, 1.25], vertical_alignment="top")
    with model_col:
        with st.container(border=True):
            st.subheader("Current Active Model")
            if os.path.exists(current_metrics_path):
                try:
                    current_metrics = pd.DataFrame(joblib.load(current_metrics_path).items(), columns=["Metric", "Value"])
                    current_metrics['Value'] = current_metrics['Value'].astype(str)
                    st.dataframe(current_metrics, use_container_width=True, hide_index=True)
                except Exception as e:
                    st.warning(f"Could not read current model metrics: {e}")
            else:
                st.info("No active model metrics found yet.")

    with upload_col:
        upload_panel = st.container(border=True)
        upload_panel.subheader("Train From Uploaded Sales Data")
        uploaded_file = upload_panel.file_uploader(
            "Upload input file named sales_data.xlsx",
            type=["xlsx"],
            help="The workbook must include the required sales columns used by the training pipeline."
        )

    if uploaded_file is None:
        upload_panel.info("Choose a sales workbook to enable training.")
        return

    if uploaded_file.name.lower() != "sales_data.xlsx":
        upload_panel.warning("Recommended filename is `sales_data.xlsx`. The uploaded file will still be validated before training.")

    try:
        uploaded_sales_df = pd.read_excel(uploaded_file, engine='openpyxl')
        uploaded_file.seek(0)
        preview_columns = uploaded_sales_df.columns.tolist()
        missing_columns = [column for column in SALES_REQUIRED_COLUMNS if column not in preview_columns]
        if missing_columns:
            upload_panel.error(f"Cannot train yet. Missing required columns: {', '.join(missing_columns)}")
            return
        upload_panel.success(f"Validated workbook columns. Found {len(preview_columns)} columns.")
    except Exception as e:
        upload_panel.error(f"Could not read uploaded workbook: {e}")
        return

    upload_summary = build_upload_summary(uploaded_sales_df)
    upload_panel.subheader("Upload Summary")
    metric_a, metric_b, metric_c = upload_panel.columns(3)
    metric_a.metric("Rows added", f"{upload_summary['rows_added']:,}")
    metric_b.metric("Duplicates removed", f"{upload_summary['duplicates_removed']:,}")
    metric_c.metric("Rows for review", f"{upload_summary['rows_after_dedupe']:,}")
    upload_panel.caption(
        f"Date range: {_format_date(upload_summary['date_min'])} to {_format_date(upload_summary['date_max'])}"
    )

    gap_count = len(upload_summary["missing_dates"])
    anomaly_count = len(upload_summary["anomalies"])
    if gap_count == 0:
        upload_panel.success("Detect gaps: no missing calendar dates found.")
    else:
        upload_panel.warning(f"Detect gaps: {gap_count:,} missing calendar dates found.")
        with upload_panel.expander("Missing dates"):
            missing_date_df = pd.DataFrame({
                "Missing Date": [date.strftime("%d %b %Y") for date in upload_summary["missing_dates"]]
            })
            st.dataframe(missing_date_df, use_container_width=True, hide_index=True)

    if anomaly_count == 0:
        upload_panel.success("Sudden drops/spikes: none detected in daily quantity totals.")
    else:
        upload_panel.warning(f"Sudden drops/spikes: {anomaly_count:,} daily movement(s) detected.")
        with upload_panel.expander("Sudden drops/spikes"):
            st.dataframe(pd.DataFrame(upload_summary["anomalies"]), use_container_width=True, hide_index=True)

    retraining_strategy = upload_panel.radio(
        "Retraining strategy",
        ["Full retrain (simple)", "Rolling window (last 2-3 years)"],
        horizontal=True,
        help="Full retrain uses the cleaned uploaded file. Rolling window trains only on the latest three years from the upload date range."
    )
    training_df, cutoff_date = apply_retraining_strategy(upload_summary, retraining_strategy)
    if cutoff_date is not None:
        upload_panel.caption(
            f"Rolling window selected: training will use {len(training_df):,} rows from {_format_date(cutoff_date)} onward."
        )
    else:
        upload_panel.caption(f"Full retrain selected: training will use {len(training_df):,} cleaned rows.")

    if upload_panel.button("Train Model", type="primary"):
        os.makedirs(TRAINING_UPLOAD_DIR, exist_ok=True)
        os.makedirs(STAGED_MODEL_DIR, exist_ok=True)
        for artifact in MODEL_ARTIFACTS:
            staged_artifact_path = os.path.join(STAGED_MODEL_DIR, artifact)
            if os.path.exists(staged_artifact_path):
                os.remove(staged_artifact_path)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        upload_path = os.path.join(TRAINING_UPLOAD_DIR, f"sales_data_{timestamp}.xlsx")

        training_df.to_excel(upload_path, index=False, engine='openpyxl')

        training_logs = []

        def ui_log(message):
            training_logs.append(message)

        try:
            with upload_panel.status("Training model tournament. This can take a few minutes...", expanded=True) as training_status:
                missing_columns, _ = validate_sales_data_columns(upload_path)
                if missing_columns:
                    raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

                metrics = run_master_tournament(
                    data_path=upload_path,
                    model_dir=STAGED_MODEL_DIR,
                    log_func=ui_log
                )
                if metrics is None:
                    raise RuntimeError("Training did not return model metrics.")

                missing_artifacts = [
                    artifact for artifact in MODEL_ARTIFACTS
                    if not os.path.exists(os.path.join(STAGED_MODEL_DIR, artifact))
                ]
                if missing_artifacts:
                    raise RuntimeError(f"Training completed but artifacts are missing: {', '.join(missing_artifacts)}")

                training_status.update(label="Training finished. Promoting model artifacts...", state="running")
                os.makedirs(MODEL_DIR, exist_ok=True)
                for artifact in MODEL_ARTIFACTS:
                    shutil.copyfile(
                        os.path.join(STAGED_MODEL_DIR, artifact),
                        os.path.join(MODEL_DIR, artifact)
                    )
                shutil.copyfile(upload_path, DATA_PATH)

                st.cache_data.clear()
                st.cache_resource.clear()
                training_status.update(label="Training completed. New model artifacts are active.", state="complete", expanded=False)

            upload_panel.success("Training completed. New model artifacts are now active for prediction.")
            upload_panel.subheader("New Model Metrics")
            upload_panel.table(pd.DataFrame(metrics.items(), columns=["Metric", "Value"]))
            with upload_panel.expander("Training log"):
                st.text("\n".join(training_logs))
        except Exception as e:
            upload_panel.error(f"Training failed: {e}")
            if training_logs:
                with upload_panel.expander("Training log"):
                    st.text("\n".join(training_logs))


current_user = require_login()
workspace = render_workspace_selector(current_user)
if not genai_available:
    st.sidebar.caption("AI insights are disabled until the GenAI layer is implemented.")

if workspace == "Model Training":
    if current_user["role"] != "Admin":
        st.error("Only Admin users can access model training.")
        st.stop()
    render_training_workspace()
    st.stop()

if workspace == "User Management":
    if current_user["role"] != "Admin":
        st.error("Only Admin users can access user management.")
        st.stop()
    render_user_management()
    st.stop()

# --- ASSET LOADING ---
@st.cache_resource
def load_assets():
    """Loads the AI brain and the technical scorecard generated during training."""
    try:
        model = joblib.load('models/master_model.pkl')
        le_reg = joblib.load('models/le_region.pkl')
        le_store = joblib.load('models/le_store.pkl')
        le_prod = joblib.load('models/le_prod.pkl')
        
        # Try to load enhanced encoders, use defaults if not available
        try:
            le_combo = joblib.load('models/le_product_store.pkl')
        except:
            le_combo = None
            
        try:
            le_month_prod = joblib.load('models/le_month_product.pkl')
        except:
            le_month_prod = None
            
        metrics = joblib.load('models/model_metrics.pkl')
        return model, le_reg, le_store, le_prod, le_combo, le_month_prod, metrics
    except:
        return None, None, None, None, None, None, None

# --- SESSION STATE INITIALIZATION ---
# This prevents the app from refreshing to the landing page after every click.
if 'report_generated' not in st.session_state:
    st.session_state.report_generated = False

# --- GENAI INITIALIZATION ---
if genai_available:
    try:
        # Try to initialize GenAI (will work if OPENAI_API_KEY is set)
        initialize_genai(provider="openai")
        genai_initialized = True
    except Exception as e:
        st.sidebar.warning(f"GenAI not initialized: {e}")
        genai_initialized = False
else:
    genai_initialized = False

# --- SIDEBAR: PREDICTION CONTROLS ---
st.sidebar.markdown('<div class="section-kicker">Prediction Controls</div>', unsafe_allow_html=True)
DATA_PATH = 'data/sales_data.xlsx'

if os.path.exists(DATA_PATH):
    @st.cache_data(show_spinner=False)
    def get_meta(path):
        """Loads necessary columns for historical trending and AutoML grounding."""
        base_cols = ['RegionName', 'StoreName', 'ProductName', 'BillDate', 'Quantity']
        optional_cols = [
            'True Demand',
            'Bank Funded NoCost EMI Applied With BankName',
            'Bank Funded Cashback Coverted To Percentage With BankName',
            'Additional Apple Or Distributor Funded Scheme In Percentage'
        ]
        available_columns = pd.read_excel(path, nrows=0, engine='openpyxl').columns.tolist()
        usecols = base_cols + [c for c in optional_cols if c in available_columns]
        df = pd.read_excel(path, usecols=usecols, engine='openpyxl')
        return df, [c for c in optional_cols if c in df.columns]
    
    df_meta, available_extra_features = get_meta(DATA_PATH)
    sel_reg = st.sidebar.selectbox("Region", sorted(df_meta['RegionName'].unique()))
    sel_store = st.sidebar.selectbox("Store", sorted(df_meta[df_meta['RegionName'] == sel_reg]['StoreName'].unique()))
    sel_prod = st.sidebar.selectbox("Product", sorted(df_meta[df_meta['StoreName'] == sel_store]['ProductName'].unique()))
    
    st.sidebar.divider()
    # Add historical forecast date selector
    max_date = pd.to_datetime(df_meta['BillDate']).max().date()
    min_date = pd.to_datetime(df_meta['BillDate']).min().date()
    forecast_start_date = st.sidebar.date_input(
        "Forecast Start Date (Historical)",
        value=max_date,
        min_value=min_date,
        max_value=max_date,
        help="Select a historical date to forecast from. Data up to this date will be used for training."
    )
    st.sidebar.divider()
    horizon = st.sidebar.select_slider("Planning Horizon (Days)", options=[7, 30, 60], value=30)
    lead_time = st.sidebar.select_slider("Supplier Lead Time (Days)", options=[7, 30, 60], value=7)
    growth_margin = st.sidebar.slider("Growth Margin (%)", 0, 50, 15, 5)
else:
    st.error("Critical: 'data/sales_data.xlsx' not found.")
    st.stop()

# --- TRIGGER REPORT GENERATION ---
if st.sidebar.button("🚀 Run Intelligence Report"):
    st.session_state.report_generated = True

# --- MAIN DASHBOARD AREA ---
if st.session_state.report_generated:
    model, le_reg, le_store, le_prod, le_combo, le_month_prod, metrics = load_assets()
    
    if model and metrics:
        try:
            # 1. FETCH HISTORICAL STARTING POINTS (using cutoff date)
            last_qty, rolling_qty, last_true_demand, last_emi_ratio, last_cashback_pct, last_add_pct = re.get_last_known_metrics(
                df_meta, sel_store, sel_prod, forecast_start_date
            )
            num_weeks = (horizon // 7) + (1 if horizon % 7 > 0 else 0)
            
            # 2. CONSTRUCT ENHANCED FEATURE DATAFRAME (matching training pipeline)
            # Create base features
            base_features = {
                'RegionID': [le_reg.transform([sel_reg])[0]]*num_weeks,
                'StoreID': [le_store.transform([sel_store])[0]]*num_weeks,
                'ProductID': [le_prod.transform([sel_prod])[0]]*num_weeks,
                'RegionName': [sel_reg] * num_weeks,
                'BillDate': [(forecast_start_date + datetime.timedelta(weeks=i)) for i in range(num_weeks)],
                'Week': [(forecast_start_date + datetime.timedelta(weeks=i)).isocalendar()[1] for i in range(num_weeks)],
                'Month': [(forecast_start_date + datetime.timedelta(weeks=i)).month for i in range(num_weeks)],
                'Year': [(forecast_start_date + datetime.timedelta(weeks=i)).year for i in range(num_weeks)],
                'DiscountPct': [0.05]*num_weeks,
                'Lag_1_Week': [last_qty]*num_weeks,
                'Lag_2_Week': [last_qty]*num_weeks,
                'Lag_3_Week': [last_qty]*num_weeks,
                'Lag_4_Week': [last_qty]*num_weeks,
                'Lag_8_Week': [last_qty]*num_weeks,
                'Lag_12_Week': [last_qty]*num_weeks,
                'Rolling_Mean_2': [rolling_qty]*num_weeks,
                'Rolling_Mean_4': [rolling_qty]*num_weeks,
                'Rolling_Mean_8': [rolling_qty]*num_weeks,
                'Rolling_Mean_12': [rolling_qty]*num_weeks,
                'Rolling_Std_2': [rolling_qty * 0.15]*num_weeks,
                'Rolling_Std_4': [rolling_qty * 0.15]*num_weeks,
                'Rolling_Std_8': [rolling_qty * 0.15]*num_weeks,
                'Rolling_Std_12': [rolling_qty * 0.15]*num_weeks,
                'True_Demand': [last_true_demand]*num_weeks,
                'TrueDemand_Ratio': [last_true_demand / (last_qty + 1e-5)]*num_weeks,
                'Bank_Emi_Flag': [last_emi_ratio]*num_weeks,
                'Bank_Cashback_Pct': [last_cashback_pct]*num_weeks,
                'Additional_Funded_Scheme_Pct': [last_add_pct]*num_weeks
            }
            
            # Create future_df with base features
            future_df = pd.DataFrame(base_features)
            future_df['BillDate'] = pd.to_datetime(future_df['BillDate'])
            
            # Add enhanced features to match training pipeline
            
            # Temporal features
            future_df['Quarter'] = future_df['Month'].apply(lambda x: (x-1)//3 + 1)
            future_df['Is_Month_Start'] = [(forecast_start_date + datetime.timedelta(weeks=i)).day <= 7 for i in range(num_weeks)]
            future_df['Is_Month_End'] = [(forecast_start_date + datetime.timedelta(weeks=i)).day >= 25 for i in range(num_weeks)]
            future_df['Week_Sin'] = np.sin(2 * np.pi * future_df['Week'] / 52.0)
            future_df['Week_Cos'] = np.cos(2 * np.pi * future_df['Week'] / 52.0)
            future_df['Month_Sin'] = np.sin(2 * np.pi * future_df['Month'] / 12.0)
            future_df['Month_Cos'] = np.cos(2 * np.pi * future_df['Month'] / 12.0)

            # Add holiday features from external feature module
            if holiday_features_available:
                try:
                    future_df = add_external_features(future_df, region_column='RegionName')
                except Exception as e:
                    st.warning(f"Holiday feature generation failed: {e}")
            else:
                st.warning("Holiday features are unavailable because the module could not be imported.")
            
            # Lag features (use available historical data or defaults)
            for lag in [2, 3, 4, 8, 12]:
                future_df[f'Lag_{lag}_Week'] = [last_qty] * num_weeks  # Default to last known value
            
            # Rolling statistics (use available data or defaults)
            for window in [2, 4, 8, 12]:
                future_df[f'Rolling_Mean_{window}'] = [rolling_qty] * num_weeks
                future_df[f'Rolling_Std_{window}'] = [rolling_qty * 0.15] * num_weeks  # Estimate std as 15% of mean
            
            # Trend indicators
            future_df['EWMA_4'] = [rolling_qty] * num_weeks
            future_df['MoM_Change'] = [0.0] * num_weeks  # Default to no change
            future_df['Trend_4Week'] = [0.0] * num_weeks  # Default to no trend
            future_df['Quantity_ZScore'] = [0.0] * num_weeks  # Default to mean
            
            # Price & promotion features
            future_df['Discount_Intensity'] = pd.cut(future_df['DiscountPct'], bins=[0, 0.05, 0.15, 0.3, 1.0], labels=[0, 1, 2, 3]).fillna(0).astype(int)
            future_df['Effective_Price'] = [100.0] * num_weeks  # Default MRP
            future_df['Bank_Cashback_Intensity'] = pd.cut(future_df['Bank_Cashback_Pct'], bins=[-1, 0, 2, 5, 10, 100], labels=[0, 1, 2, 3, 4]).fillna(0).astype(int)
            future_df['Additional_Funded_Scheme_Intensity'] = pd.cut(future_df['Additional_Funded_Scheme_Pct'], bins=[-1, 0, 2, 5, 10, 100], labels=[0, 1, 2, 3, 4]).fillna(0).astype(int)
            
            # Interaction features (using proper encoders if available)
            future_df['Product_Store_Combo'] = future_df['ProductID'].astype(str) + '_' + future_df['StoreID'].astype(str)
            if le_combo is not None:
                try:
                    future_df['Product_Store_ID'] = le_combo.transform(future_df['Product_Store_Combo'])
                except ValueError:
                    # Handle unseen labels by using a default value
                    future_df['Product_Store_ID'] = [0] * num_weeks  # Default for unseen combinations
                    st.warning("Some product-store combinations not seen during training, using defaults")
            else:
                future_df['Product_Store_ID'] = [0] * num_weeks  # Fallback for older models
            
            future_df['Month_Product'] = future_df['Month'].astype(str) + '_' + future_df['ProductID'].astype(str)
            if le_month_prod is not None:
                try:
                    future_df['Month_Product_ID'] = le_month_prod.transform(future_df['Month_Product'])
                except ValueError:
                    # Handle unseen labels by using a default value
                    future_df['Month_Product_ID'] = [0] * num_weeks  # Default for unseen combinations
                    st.warning("Some month-product combinations not seen during training, using defaults")
            else:
                future_df['Month_Product_ID'] = [0] * num_weeks  # Fallback for older models
            
            # NOW detect which features the model was trained with (after all features are created)
            try:
                # Try to get feature names from the model if available (for sklearn models)
                if hasattr(model, 'feature_names_in_'):
                    model_features = list(model.feature_names_in_)
                    st.info(f"Model expects {len(model_features)} features")
                else:
                    # Fallback: use all available features that match expected pattern
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
                    st.info(f"Using {len(model_features)} available features")
            except Exception as e:
                # Ultimate fallback to basic features that should always exist
                model_features = ['RegionID', 'StoreID', 'ProductID', 'Week', 'Month', 'Year', 'DiscountPct', 'Lag_1_Week', 'Rolling_Mean_4']
                st.warning(f"Feature detection failed ({e}), using basic features")
            
            # Reorder columns to match model expectations
            missing_features = [f for f in model_features if f not in future_df.columns]
            if missing_features:
                for feature in missing_features:
                    future_df[feature] = 0
                st.warning(f"Filled missing model features with defaults: {missing_features}")

            future_df = future_df[model_features]
            
            # Debug: Check for missing features
            missing_features = [f for f in model_features if f not in future_df.columns]
            if missing_features:
                st.error(f"Missing features after reordering: {missing_features}")
                st.stop()

            # 3. EXECUTE PREDICTION
            preds = [max(0, p) for p in model.predict(future_df)]

            # Holiday list for selected state and forecast window
            holiday_df = pd.DataFrame()
            if holiday_features_available:
                try:
                    years = sorted({forecast_start_date.year, (forecast_start_date + datetime.timedelta(days=horizon-1)).year})
                    holiday_df = get_state_holiday_list(sel_reg, years=years)
                    end_date = forecast_start_date + datetime.timedelta(days=horizon-1)
                    holiday_df['Date'] = pd.to_datetime(holiday_df['Date'])
                    holiday_df = holiday_df[(holiday_df['Date'] >= pd.Timestamp(forecast_start_date)) & (holiday_df['Date'] <= pd.Timestamp(end_date))]
                    holiday_df['Date'] = holiday_df['Date'].dt.date
                except Exception as e:
                    st.warning(f"Unable to build holiday list: {e}")
            total_qty = int(round(sum(preds)))
            labels = [(forecast_start_date + datetime.timedelta(weeks=i)).strftime('%b %d') for i in range(num_weeks)]

            # --- TABBED LAYOUT ---
            st.title(f"📊 {sel_prod} Analysis: {sel_store}")
            st.caption(f"Forecast starts from {forecast_start_date.strftime('%B %d, %Y')}.")
            
            tab1, tab2, tab3, tab4 = st.tabs([
                "📈 Historical Trend", 
                "🔮 Demand Forecast & Strategy", 
                "📋 Reports", 
                "🤖 AI Insights"
            ])

            # TAB 1: HISTORICAL TREND
            with tab1:
                trend_panel = st.container(border=True)
                trend_panel.subheader("3-Year Historical Sales Trend")
                full_hist = df_meta[(df_meta['StoreName'] == sel_store) & (df_meta['ProductName'] == sel_prod)].copy()
                if not full_hist.empty:
                    full_hist['BillDate'] = pd.to_datetime(full_hist['BillDate'])
                    # resample uses 'ME' for modern Pandas compatibility
                    trend_data = full_hist.set_index('BillDate').sort_index().resample('ME')['Quantity'].sum()
                    trend_panel.line_chart(trend_data, color="#3498db")
                    trend_panel.write(f"*Actual sales recorded from {trend_data.index.min().date()} to {trend_data.index.max().date()}.*")
                else:
                    trend_panel.warning("No historical data available for this selection.")

            # TAB 2: FORECAST & STRATEGY
            with tab2:
                avg_daily = total_qty / horizon
                safety_stock, sigma = calculate_safety_stock(full_hist, preds, horizon, lead_time)
                reorder_point = round((avg_daily * lead_time) + safety_stock)
                final_proc = int(round(total_qty * (1 + (growth_margin/100))))

                with st.container(border=True):
                    k1, k2, k3, k4 = st.columns(4)
                    k1.metric(f"{horizon}-Day Demand", f"{total_qty} Units")
                    k2.metric("Safety Stock", f"{safety_stock} Units")
                    k3.metric("Reorder Point", f"{reorder_point}")
                    k4.metric("Daily Velocity", f"{round(avg_daily, 2)} / Day")
                
                l_col, r_col = st.columns([1.55, 1], vertical_alignment="top")
                with l_col:
                    forecast_panel = st.container(border=True)
                    forecast_panel.subheader("AI Demand Projection")
                    forecast_chart = pd.DataFrame({'Week': labels, 'Units': preds})
                    forecast_chart['Week'] = forecast_chart['Week'].astype(str)
                    chart = alt.Chart(forecast_chart).encode(
                        x=alt.X('Week:N', title='Week', sort=None),
                        y=alt.Y('Units:Q', title='Forecast Units'),
                        tooltip=[
                            alt.Tooltip('Week:N', title='Week'),
                            alt.Tooltip('Units:Q', title='Forecast Units', format=',.0f')
                        ]
                    )
                    forecast_viz = (
                        chart.mark_area(color="#2ecc71", opacity=0.14)
                        + chart.mark_line(color="#2ecc71", size=3, interpolate='monotone')
                        + chart.mark_point(color="#117a65", size=120, filled=True)
                    ).configure_axis(labelAngle=-45)
                    forecast_panel.altair_chart(forecast_viz, use_container_width=True)
                    forecast_panel.caption(f"Trend projection from {labels[0]} to {labels[-1]}.")
                with r_col:
                    strategy_panel = st.container(border=True)
                    strategy_panel.subheader("Inventory Strategy")
                    strategy_panel.info(f"Planning horizon velocity: **{round(np.mean(preds), 1)} units/week**.")
                    strategy_panel.info(f"Lead time buffer: **{safety_stock} units**.")
                    strategy_panel.info(f"Restock trigger: **{reorder_point} units**.")
                    strategy_panel.success(f"Procure **{final_proc} units** with {growth_margin}% growth margin.")
                    # Add AI insights if available
                    if genai_initialized:
                        strategy_panel.markdown("---")
                        strategy_panel.subheader("AI-Powered Insights")
                        with st.spinner("Generating AI insights..."):
                            quick_insights = get_forecast_insights(preds, sel_prod)
                            strategy_panel.info(quick_insights)

                score_col, holiday_col = st.columns(2, vertical_alignment="top")
                with score_col:
                    score_panel = st.container(border=True)
                    score_panel.subheader(f"Technical Scorecard: {metrics['Algorithm']}")
                    m_df = pd.DataFrame(metrics.items(), columns=["Parameter", "Value"])
                    m_df['Value'] = m_df['Value'].astype(str) # Prevents Arrow TypeError
                    score_panel.dataframe(m_df, use_container_width=True, hide_index=True)

                with holiday_col:
                    holiday_panel = st.container(border=True)
                    holiday_panel.subheader("Holiday Calendar")
                    if not holiday_df.empty:
                        holiday_panel.write(f"Showing holidays for **{sel_reg}** from **{forecast_start_date}** to **{end_date}**.")
                        holiday_panel.dataframe(holiday_df.reset_index(drop=True), use_container_width=True, hide_index=True)

                        total_holidays = len(holiday_df)
                        upcoming_holiday = holiday_df.iloc[0] if total_holidays > 0 else None
                        if upcoming_holiday is not None:
                            holiday_name = upcoming_holiday['Holiday'] or 'Unnamed holiday'
                            holiday_date = upcoming_holiday['Date']
                            holiday_panel.markdown("**Holiday Impact Summary**")
                            holiday_panel.write(
                                f"- **{total_holidays} holiday(s)** occur during the forecast window.\n"
                                f"- The earliest upcoming holiday is **{holiday_name}** on **{holiday_date}**.\n"
                                f"- Holiday weeks often shift buying behavior, so inventory and promotions should be checked for these dates."
                            )
                        else:
                            holiday_panel.markdown("**Holiday Impact Summary**")
                            holiday_panel.write("- Holiday dates were identified but the list could not be summarized.")
                    elif holiday_features_available:
                        holiday_panel.info("No holidays found in the forecast window for this region.")
                    else:
                        holiday_panel.info("Holiday support is unavailable for this session.")

            # TAB 3: REPORTS
            with tab3:
                st.subheader("Export Center")
                col_a, col_b = st.columns(2)
                with col_a:
                    single_report_panel = st.container(border=True)
                    single_report_panel.subheader("Individual Product Report")
                    csv_bytes = re.generate_single_forecast_csv(sel_reg, sel_store, sel_prod, labels, preds, forecast_start_date)
                    single_report_panel.download_button("Download Detailed CSV", csv_bytes, f"Forecast_{sel_prod}.csv", "text/csv")
                
                with col_b:
                    master_report_panel = st.container(border=True)
                    master_report_panel.subheader("Master Store Report")
                    if master_report_panel.button("Generate Bulk Analysis"):
                        with st.spinner("Processing store-wide intelligence..."):
                            bulk_df = re.generate_bulk_report(df_meta, model, (le_reg, le_store, le_prod, le_combo, le_month_prod), sel_reg, sel_store, forecast_start_date)
                            master_report_panel.download_button("Download Master Bulk CSV", bulk_df.to_csv(index=False).encode('utf-8'), f"Bulk_{sel_store}.csv", "text/csv")

                            # Add AI-powered executive summary if GenAI is available
                            if genai_initialized and not bulk_df.empty:
                                master_report_panel.subheader("AI Executive Summary")
                                with st.spinner("Generating executive summary..."):
                                    executive_summary = re.generate_executive_summary_report(
                                        bulk_df, metrics,
                                        business_context=f"Store-wide forecast analysis for {sel_store} in {sel_reg}"
                                    )
                                    master_report_panel.write(executive_summary)

            # TAB 4: AI INSIGHTS
            with tab4:
                st.subheader("🤖 GenAI Strategic Briefing")

                if not genai_initialized:
                    st.warning("⚠️ GenAI not available. Please set OPENAI_API_KEY environment variable to enable AI insights.")
                    st.info(f"The {metrics['Algorithm']} model currently suggests a stability in demand for {sel_prod} based on historical patterns in {sel_reg}.")
                else:
                    # Get historical data for analysis
                    hist_data = df_meta[(df_meta['StoreName'] == sel_store) & (df_meta['ProductName'] == sel_prod)].copy()
                    hist_avg = hist_data['Quantity'].mean() if not hist_data.empty else 0

                    # AI-Powered Analysis Tabs
                    insight_tab1, insight_tab2, insight_tab3, insight_tab4 = st.tabs([
                        "📊 Model Performance",
                        "🔍 Trend Analysis",
                        "📦 Inventory Strategy",
                        "🚨 Alerts & Anomalies"
                    ])

                    with insight_tab1:
                        st.subheader("Model Accuracy Explanation")
                        with st.spinner("Analyzing model performance..."):
                            accuracy_explanation = explain_forecast_accuracy(
                                metrics,
                                business_context=f"Retail demand forecasting for {sel_prod} in {sel_store}, {sel_reg}"
                            )
                            st.write(accuracy_explanation)

                    with insight_tab2:
                        st.subheader("Forecast Trend Analysis")
                        with st.spinner("Analyzing forecast trends..."):
                            # Get external factors (placeholder - can be enhanced later)
                            external_factors = {
                                "season": "Based on historical patterns",
                                "promotion": f"Assumed {future_df['DiscountPct'].iloc[0]*100}% discount",
                                "competitor_activity": "Not available",
                                "economic_indicators": "Not available"
                            }

                            trend_analysis = analyze_forecast_trends(
                                preds, sel_prod, hist_avg, external_factors
                            )
                            st.write(trend_analysis)

                    with insight_tab3:
                        st.subheader("Smart Inventory Recommendations")
                        current_stock = st.number_input(
                            "Current Stock Level",
                            min_value=0,
                            value=int(reorder_point * 0.8),  # Estimate current stock
                            help="Enter your current inventory level for personalized recommendations"
                        )

                        with st.spinner("Generating inventory strategy..."):
                            inventory_recommendations = generate_inventory_recommendations(
                                preds, current_stock, sel_prod, lead_time
                            )

                            if "error" not in inventory_recommendations:
                                col1, col2 = st.columns(2)

                                with col1:
                                    st.metric("Recommended Reorder Quantity", inventory_recommendations['recommended_reorder_quantity'])
                                    st.metric("Safety Stock Level", inventory_recommendations['safety_stock_level'])
                                    st.metric("Risk Level", inventory_recommendations['risk_level'])

                                with col2:
                                    st.metric("Reorder Point", inventory_recommendations['reorder_point'])
                                    if inventory_recommendations['projected_stockout_week']:
                                        st.metric("Projected Stockout", f"Week {inventory_recommendations['projected_stockout_week']}")

                                st.subheader("AI Strategy Insights")
                                st.write(inventory_recommendations['ai_insights'])
                            else:
                                st.error(inventory_recommendations['error'])

                    with insight_tab4:
                        st.subheader("Business Alerts & Anomaly Detection")

                        # Business alerts
                        alert = format_business_alert(preds, current_stock, sel_prod)
                        if alert:
                            st.error(alert)
                        else:
                            st.success("✅ No critical inventory alerts at current stock levels")

                        # Anomaly detection
                        if not hist_data.empty:
                            with st.spinner("Detecting forecast anomalies..."):
                                anomaly_analysis = detect_anomalies(
                                    preds, hist_data, sel_prod, threshold=2.0
                                )
                                st.subheader("Anomaly Analysis")
                                st.write(anomaly_analysis)
                        else:
                            st.info("Insufficient historical data for anomaly detection")

                    # Quick Insights Summary
                    st.divider()
                    st.subheader("🎯 Quick AI Insights")
                    with st.spinner("Generating summary insights..."):
                        quick_insights = get_forecast_insights(preds, sel_prod)
                        st.info(quick_insights)

        except Exception as e:
            st.error(f"⚠️ Calculation Error: {e}")
            st.session_state.report_generated = False
    else:
        st.error("Model assets missing. Please run training_pipeline.py.")
        st.session_state.report_generated = False
else:
    # --- LANDING STATE ---
    st.info("👈 Select your parameters and click 'Run Intelligence Report' to begin.")
st.image("https://images.unsplash.com/photo-1510878939963-10214bb20ec8?auto=format&fit=crop&q=80&w=1200", 
             caption="iPlanet Demand Planner | Bangalore", width=1200)
