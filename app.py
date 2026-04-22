"""
AI Smart Expense Tracker
========================
Production-level Streamlit finance dashboard.
Stack: Python · Streamlit · SQLite · Matplotlib · scikit-learn
"""

import sqlite3
from datetime import date, datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  — must be the FIRST Streamlit call
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Expense Tracker",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STYLES
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; font-size: 15px; }
    .stApp { background-color: #f5f6fa; }
    #MainMenu, footer { visibility: hidden; }
    header[data-testid="stHeader"] { background: transparent; }

    /* sidebar */
    section[data-testid="stSidebar"] { background-color: #0f172a !important; border-right: 1px solid #1e293b; }
    section[data-testid="stSidebar"] * { color: #94a3b8 !important; }
    section[data-testid="stSidebar"] .sidebar-brand { color: #f1f5f9 !important; font-size: 1.1rem; font-weight: 700; letter-spacing: -0.02em; padding: 0.4rem 0 1.4rem; display: block; }
    section[data-testid="stSidebar"] .sidebar-label { color: #475569 !important; font-size: 0.68rem; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 0.35rem; display: block; }
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label { color: #cbd5e1 !important; font-size: 0.9rem !important; font-weight: 500 !important; }

    /* metric cards */
    div[data-testid="metric-container"] { background: #ffffff; border-radius: 10px; padding: 1rem 1.2rem; border: 1px solid #e2e8f0; box-shadow: 0 1px 4px rgba(0,0,0,0.04); }
    div[data-testid="metric-container"] label { font-size: 0.72rem !important; font-weight: 600 !important; letter-spacing: 0.05em; text-transform: uppercase; color: #64748b !important; }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] { font-size: 1.55rem !important; font-weight: 700 !important; color: #0f172a !important; }

    /* typography */
    .page-title { font-size: 1.35rem; font-weight: 700; color: #0f172a; letter-spacing: -0.02em; margin-bottom: 0.2rem; }
    .page-subtitle { font-size: 0.84rem; color: #64748b; margin-bottom: 1.6rem; }
    .section-label { font-size: 0.68rem; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: #94a3b8; margin-bottom: 0.5rem; }

    /* button */
    .stButton > button { background-color: #0f172a; color: #f8fafc; border: none; border-radius: 8px; padding: 0.52rem 1.4rem; font-size: 0.88rem; font-weight: 600; width: 100%; transition: background 0.18s; }
    .stButton > button:hover { background-color: #1e293b; color: #f8fafc; }

    /* inputs */
    .stNumberInput input, .stTextInput input { border-radius: 8px !important; border: 1px solid #e2e8f0 !important; background: #ffffff !important; font-size: 0.9rem !important; }

    .stDataFrame { border-radius: 10px; overflow: hidden; border: 1px solid #e2e8f0; }
    .stAlert { border-radius: 8px; font-size: 0.88rem; }
    hr { border: none; border-top: 1px solid #e2e8f0; margin: 1.4rem 0; }
    .app-footer { text-align: center; font-size: 0.76rem; color: #94a3b8; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# MATPLOTLIB THEME
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.facecolor": "#ffffff", "figure.facecolor": "#ffffff",
    "axes.edgecolor": "#e2e8f0", "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": "#f1f5f9", "grid.linewidth": 0.8,
    "xtick.color": "#64748b", "ytick.color": "#64748b",
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.labelsize": 10, "axes.labelcolor": "#475569",
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlecolor": "#0f172a",
    "text.color": "#0f172a", "legend.fontsize": 9, "legend.framealpha": 0.5,
    "figure.dpi": 120,
})

PALETTE = {"food": "#3b82f6", "travel": "#10b981", "shopping": "#f59e0b", "other": "#8b5cf6"}
ACCENT  = "#3b82f6"

# ─────────────────────────────────────────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────────────────────────────────────────
DB_FILE = "expenses.db"

def get_connection():
    return sqlite3.connect(DB_FILE)

def create_table():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS expenses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                amount REAL NOT NULL,
                category TEXT NOT NULL,
                description TEXT NOT NULL
            )
        """)

create_table()

# ─────────────────────────────────────────────────────────────────────────────
# CATEGORY DETECTION
# ─────────────────────────────────────────────────────────────────────────────
CATEGORY_KEYWORDS = {
    "food":     ["pizza","burger","cafe","coffee","restaurant","food","biryani","lunch","dinner","breakfast","snack","swiggy","zomato","dominos","kfc","mcdonalds","sandwich","chai"],
    "travel":   ["uber","ola","bus","auto","metro","train","taxi","fuel","petrol","flight","ticket","travel","rapido"],
    "shopping": ["shirt","shoes","amazon","flipkart","clothes","dress","jeans","myntra","bag","watch","mobile","headphones","meesho","shopping","gadget","earphone"],
}

def detect_category(description: str) -> str:
    desc = description.lower()
    for cat, kws in CATEGORY_KEYWORDS.items():
        if any(k in desc for k in kws):
            return cat
    return "other"

# ─────────────────────────────────────────────────────────────────────────────
# DB HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def add_expense(amount: float, description: str) -> str:
    cat = detect_category(description)
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO expenses (date, amount, category, description) VALUES (?, ?, ?, ?)",
            (date.today().strftime("%Y-%m-%d"), amount, cat, description),
        )
    return cat

def get_all_expenses() -> pd.DataFrame:
    with get_connection() as conn:
        return pd.read_sql_query("SELECT id, date, amount, category, description FROM expenses ORDER BY date DESC", conn)

def get_monthly_expenses(year: int, month: int) -> pd.DataFrame:
    with get_connection() as conn:
        return pd.read_sql_query("SELECT * FROM expenses WHERE date LIKE ?", conn, params=(f"{year:04d}-{month:02d}%",))

def get_monthly_totals() -> pd.DataFrame:
    with get_connection() as conn:
        return pd.read_sql_query("""
            SELECT substr(date,1,7) AS month, SUM(amount) AS total
            FROM expenses GROUP BY month ORDER BY month
        """, conn)

# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
def fmt_inr(v: float) -> str:
    return f"\u20b9{v:,.2f}"

def clean_spines(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<span class="sidebar-brand">Expense Tracker</span>', unsafe_allow_html=True)
    st.markdown('<span class="sidebar-label">Menu</span>', unsafe_allow_html=True)
    PAGES = ["Add Expense", "View Expenses", "Spending Chart", "Monthly Summary", "Predict Next Month"]
    page = st.radio("", PAGES, label_visibility="collapsed")
    st.markdown("---")
    st.markdown('<span class="sidebar-label">How it works</span>', unsafe_allow_html=True)
    st.markdown("<span style='font-size:0.82rem;color:#475569;'>Describe an expense in plain English and the AI auto-classifies it into Food, Travel, Shopping, or Other.</span>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: ADD EXPENSE
# ─────────────────────────────────────────────────────────────────────────────
if page == "Add Expense":
    st.markdown('<p class="page-title">Add Expense</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Category is detected automatically from your description.</p>', unsafe_allow_html=True)

    left, right = st.columns([1, 1], gap="large")
    with left:
        st.markdown('<p class="section-label">Amount (INR)</p>', unsafe_allow_html=True)
        amount = st.text_input("Amount", placeholder="Enter amount")
        st.markdown('<p class="section-label">Description</p>', unsafe_allow_html=True)
        description = st.text_input("Description", placeholder="e.g. lunch at cafe, uber to office, new headphones", label_visibility="collapsed")
        if st.button("Save Expense"):
            try:
                amount = float(amount)
                if amount <= 0:
                    st.error("Enter an amount greater than zero.")
                elif not description.strip():
                     st.error("Enter a short description.")
                else:
                     cat = add_expense(amount, description.strip())
                    st.success(f"Saved  —  detected category: **{cat.capitalize()}**")

    with right:
        st.markdown('<p class="section-label">Category Rules</p>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({
            "Category": ["Food", "Travel", "Shopping", "Other"],
            "Sample Keywords": ["pizza, burger, cafe, biryani, zomato", "uber, bus, metro, fuel, rapido", "shirt, shoes, amazon, flipkart, myntra", "anything not matched above"],
        }), use_container_width=True, hide_index=True)
        st.markdown('<p class="section-label">Entry Date</p>', unsafe_allow_html=True)
        st.info(f"Saving with today's date:  {date.today().strftime('%d %B %Y')}")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: VIEW EXPENSES
# ─────────────────────────────────────────────────────────────────────────────
elif page == "View Expenses":
    st.markdown('<p class="page-title">All Expenses</p>', unsafe_allow_html=True)
    df = get_all_expenses()
    if df.empty:
        st.warning("No expenses recorded yet.")
    else:
        total, count = df["amount"].sum(), len(df)
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Records", str(count))
        m2.metric("Total Spent", fmt_inr(total))
        m3.metric("Average per Entry", fmt_inr(total / count))
        st.markdown("<br>", unsafe_allow_html=True)
        BG = {"food":"#eff6ff","travel":"#ecfdf5","shopping":"#fffbeb","other":"#f5f3ff"}
        FG = {"food":"#1d4ed8","travel":"#065f46","shopping":"#92400e","other":"#5b21b6"}
        def _sc(v):
            return f"background-color:{BG.get(v,'#f8fafc')};color:{FG.get(v,'#334155')};font-weight:600;"
        disp = df.rename(columns={"id":"ID","date":"Date","amount":"Amount (INR)","category":"Category","description":"Description"})
        st.dataframe(disp, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: SPENDING CHART
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Spending Chart":
    st.markdown('<p class="page-title">Spending Insights</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Total spending broken down by category.</p>', unsafe_allow_html=True)
    df = get_all_expenses()
    if df.empty:
        st.warning("No data to display.")
    else:
        cat_totals = df.groupby("category")["amount"].sum().reset_index().sort_values("amount", ascending=False)
        colours = [PALETTE.get(c, "#94a3b8") for c in cat_totals["category"]]
        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.bar(cat_totals["category"].str.capitalize(), cat_totals["amount"], color=colours, width=0.42, edgecolor="white", linewidth=1.2, zorder=3)
        mx = cat_totals["amount"].max()
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width()/2, h + mx*0.015, fmt_inr(h), ha="center", va="bottom", fontsize=9, fontweight="600", color="#0f172a")
        ax.set_title("Total Spending by Category")
        ax.set_ylabel("Amount (INR)")
        clean_spines(ax)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        col_l, col_r = st.columns([1, 1])
        with col_l:
            st.markdown('<p class="section-label">Category Share</p>', unsafe_allow_html=True)
            fig2, ax2 = plt.subplots(figsize=(4.5, 4))
            _, _, autotexts = ax2.pie(cat_totals["amount"], labels=[c.capitalize() for c in cat_totals["category"]], colors=colours, autopct="%1.1f%%", startangle=140, pctdistance=0.80, wedgeprops=dict(width=0.55, edgecolor="#ffffff", linewidth=2))
            for at in autotexts: at.set_fontsize(9); at.set_fontweight("600")
            ax2.set_title("Category Distribution")
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)
        with col_r:
            st.markdown('<p class="section-label">Summary Table</p>', unsafe_allow_html=True)
            s = cat_totals.copy(); s.columns = ["Category", "Total (INR)"]
            s["Category"] = s["Category"].str.capitalize()
            s["Total (INR)"] = s["Total (INR)"].apply(fmt_inr)
            st.dataframe(s, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: MONTHLY SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Monthly Summary":
    st.markdown('<p class="page-title">Monthly Summary</p>', unsafe_allow_html=True)
    now = datetime.now()
    cy, cm, _ = st.columns([1,1,2])
    with cy: year  = st.selectbox("Year",  list(range(2022, now.year+1)), index=now.year-2022)
    with cm: month = st.selectbox("Month", list(range(1,13)), index=now.month-1, format_func=lambda m: datetime(2000,m,1).strftime("%B"))
    df = get_monthly_expenses(year, month)
    month_str = datetime(year, month, 1).strftime("%B %Y")
    st.markdown("<br>", unsafe_allow_html=True)
    if df.empty:
        st.warning(f"No expenses found for {month_str}.")
    else:
        total_spent, n = df["amount"].sum(), len(df)
        top_cat = df.groupby("category")["amount"].sum().idxmax()
        c1,c2,c3 = st.columns(3)
        c1.metric("Total Spent", fmt_inr(total_spent))
        c2.metric("Number of Entries", str(n))
        c3.metric("Top Category", top_cat.capitalize())
        st.markdown("<br>", unsafe_allow_html=True)
        cg = df.groupby("category")["amount"].sum().reset_index()
        fig, ax = plt.subplots(figsize=(6,3))
        ax.bar(cg["category"].str.capitalize(), cg["amount"], color=[PALETTE.get(c,"#94a3b8") for c in cg["category"]], width=0.38, edgecolor="white", linewidth=1.2, zorder=3)
        ax.set_title(f"Spending Breakdown — {month_str}")
        ax.set_ylabel("Amount (INR)")
        clean_spines(ax)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.dataframe(df[["date","amount","category","description"]].rename(columns={"date":"Date","amount":"Amount (INR)","category":"Category","description":"Description"}), use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: PREDICT NEXT MONTH
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Predict Next Month":
    st.markdown('<p class="page-title">Spending Prediction</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Linear regression on monthly totals to project next month\'s spend.</p>', unsafe_allow_html=True)
    monthly_df = get_monthly_totals()
    MIN_MONTHS = 2
    if len(monthly_df) < MIN_MONTHS:
        st.warning(f"Not enough data. Add expenses across at least **{MIN_MONTHS} different months** to enable prediction. You currently have **{len(monthly_df)}** month(s) of data.")
        st.info("Add expenses this month and next — the predictor activates automatically once two months of data exist.")
    else:
        monthly_df = monthly_df.reset_index(drop=True)
        monthly_df["month_index"] = np.arange(1, len(monthly_df)+1, dtype=float)
        X, y = monthly_df[["month_index"]].values, monthly_df["total"].values
        model = LinearRegression().fit(X, y)
        next_index    = float(len(monthly_df) + 1)
        predicted_val = max(float(model.predict([[next_index]])[0]), 0.0)
        last_actual   = float(monthly_df["total"].iloc[-1])
        delta         = predicted_val - last_actual
        trend_up      = delta > 0
        r2            = model.score(X, y)
        conf          = "High" if r2 >= 0.80 else ("Moderate" if r2 >= 0.50 else "Low")

        m1,m2,m3 = st.columns(3)
        m1.metric("Last Month Actual",    fmt_inr(last_actual))
        m2.metric("Predicted Next Month", fmt_inr(predicted_val), delta=f"{'+'if trend_up else ''}{fmt_inr(delta)}")
        m3.metric("Model Fit (R2)",       f"{r2:.2f}  ({conf})")

        if trend_up:
            st.error("The model predicts higher spending next month. Review discretionary categories to stay on budget.")
        else:
            st.success("The model predicts lower spending next month. You are on track to increase savings.")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<p class="section-label">Monthly Trend with Forecast</p>', unsafe_allow_html=True)
        actuals  = monthly_df["total"].tolist()
        x_labels = monthly_df["month"].tolist() + ["Forecast"]
        n        = len(x_labels)
        fig, ax  = plt.subplots(figsize=(9,4))
        ax.plot(range(len(actuals)), actuals, marker="o", color=ACCENT, linewidth=2, markersize=6, label="Actual", zorder=4)
        ax.fill_between(range(len(actuals)), actuals, alpha=0.08, color=ACCENT)
        ax.plot([len(actuals)-1, n-1], [last_actual, predicted_val], linestyle="--", color="#f59e0b", linewidth=1.8, marker="D", markersize=7, label="Forecast", zorder=4)
        ax.annotate(fmt_inr(predicted_val), xy=(n-1, predicted_val), xytext=(0,12), textcoords="offset points", ha="center", fontsize=9, fontweight="600", color="#f59e0b")
        ax.set_xticks(range(n))
        ax.set_xticklabels(x_labels, rotation=30, ha="right")
        ax.set_title("Monthly Spending Trend + Next Month Forecast")
        ax.set_ylabel("Amount (INR)")
        clean_spines(ax)
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        with st.expander("How does the prediction work?"):
            st.markdown(f"""
**Algorithm:** Ordinary Least Squares Linear Regression (scikit-learn)

**Feature:** Sequential month index (1, 2, 3 …).  **Target:** Total spending per month.

**Fit quality (R²):** `{r2:.4f}` — {conf.lower()} confidence.  
R² ranges from 0 to 1. Low R² with few data points is normal; accuracy improves significantly with six or more months.

**Negative guard:** Raw output is clamped to zero — spending cannot be negative.
            """)

        hist = monthly_df[["month","total"]].copy()
        hist.columns = ["Month","Total Spent (INR)"]
        hist["Total Spent (INR)"] = hist["Total Spent (INR)"].apply(fmt_inr)
        st.dataframe(hist, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("<p class='app-footer'>AI Expense Tracker &nbsp;·&nbsp; Python  Streamlit  SQLite  Matplotlib  scikit-learn</p>", unsafe_allow_html=True)
