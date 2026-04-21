import streamlit as st
import sqlite3
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
from datetime import datetime, date
from sklearn.linear_model import LinearRegression

# ─────────────────────────────────────────────
# PAGE CONFIG  (must be the very first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI Expense Tracker 💸",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS  – colourful, teen-friendly look
# ─────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ---- Google Font ---- */
    @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;900&display=swap');

    html, body, [class*="css"] {
        font-family: 'Nunito', sans-serif;
    }

    /* ---- gradient background ---- */
    .stApp {
        background: linear-gradient(135deg, #f0f4ff 0%, #fef9ff 50%, #fff4f0 100%);
    }

    /* ---- hero title ---- */
    .hero-title {
        background: linear-gradient(90deg, #6c63ff, #f77f00, #e63946);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.6rem;
        font-weight: 900;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .hero-sub {
        text-align: center;
        color: #888;
        font-size: 1rem;
        margin-bottom: 1.6rem;
    }

    /* ---- metric cards ---- */
    div[data-testid="metric-container"] {
        background: white;
        border-radius: 16px;
        padding: 1rem 1.4rem;
        box-shadow: 0 4px 18px rgba(108,99,255,0.10);
        border-left: 5px solid #6c63ff;
    }

    /* ---- section header ---- */
    .section-header {
        font-size: 1.5rem;
        font-weight: 800;
        color: #3a3a5c;
        margin-bottom: 0.8rem;
    }

    /* ---- sidebar ---- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #6c63ff 0%, #a78bfa 100%) !important;
        color: white !important;
    }
    section[data-testid="stSidebar"] * {
        color: white !important;
    }
    section[data-testid="stSidebar"] .stRadio > label {
        font-weight: 700;
        font-size: 1.05rem;
    }

    /* ---- success / info boxes ---- */
    .stAlert { border-radius: 12px; }

    /* ---- buttons ---- */
    .stButton > button {
        background: linear-gradient(90deg, #6c63ff, #a78bfa);
        color: white;
        font-weight: 700;
        border: none;
        border-radius: 12px;
        padding: 0.55rem 1.6rem;
        font-size: 1rem;
        transition: transform 0.15s, box-shadow 0.15s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(108,99,255,0.35);
    }

    /* ---- dataframe ---- */
    .stDataFrame { border-radius: 12px; overflow: hidden; }

    /* ---- divider ---- */
    hr { border-top: 2px dashed #e0e0e0; margin: 1.5rem 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# DATABASE SETUP
# ─────────────────────────────────────────────
DB_FILE = "expenses.db"


def get_connection():
    """Return a SQLite connection (creates DB file if missing)."""
    conn = sqlite3.connect(DB_FILE)
    return conn


def create_table():
    """Create the expenses table if it does not already exist."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS expenses (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            date        TEXT    NOT NULL,
            amount      REAL    NOT NULL,
            category    TEXT    NOT NULL,
            description TEXT    NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


# Create table on every app start
create_table()

# ─────────────────────────────────────────────
# CATEGORY DETECTION  (keyword-based AI logic)
# ─────────────────────────────────────────────
CATEGORY_KEYWORDS = {
    "food":     ["pizza", "burger", "cafe", "coffee", "restaurant", "food",
                 "biryani", "lunch", "dinner", "breakfast", "snack", "swiggy",
                 "zomato", "dominos", "kfc", "mcdonalds", "sandwich", "chai"],
    "travel":   ["uber", "ola", "bus", "auto", "metro", "train", "taxi",
                 "fuel", "petrol", "flight", "ticket", "travel", "rapido"],
    "shopping": ["shirt", "shoes", "amazon", "flipkart", "clothes", "dress",
                 "jeans", "myntra", "bag", "watch", "mobile", "headphones",
                 "meesho", "shopping", "gadget", "earphone"],
}


def detect_category(description: str) -> str:
    """
    Detect expense category from the description using keyword matching.
    Falls back to 'other' if no keyword matches.
    """
    desc_lower = description.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        for keyword in keywords:
            if keyword in desc_lower:
                return category
    return "other"


# ─────────────────────────────────────────────
# DATABASE HELPERS
# ─────────────────────────────────────────────
def add_expense(amount: float, description: str):
    """Insert a new expense record into the database."""
    category  = detect_category(description)
    today     = date.today().strftime("%Y-%m-%d")

    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO expenses (date, amount, category, description) VALUES (?, ?, ?, ?)",
        (today, amount, category, description),
    )
    conn.commit()
    conn.close()
    return category   # return so we can show the user what was detected


def get_all_expenses() -> pd.DataFrame:
    """Fetch all expenses as a Pandas DataFrame."""
    conn = get_connection()
    df   = pd.read_sql_query(
        "SELECT id, date, amount, category, description FROM expenses ORDER BY date DESC",
        conn,
    )
    conn.close()
    return df


def get_monthly_expenses(year: int, month: int) -> pd.DataFrame:
    """Fetch expenses for a specific year-month."""
    conn   = get_connection()
    prefix = f"{year:04d}-{month:02d}"
    df     = pd.read_sql_query(
        "SELECT * FROM expenses WHERE date LIKE ?",
        conn,
        params=(f"{prefix}%",),
    )
    conn.close()
    return df


def get_monthly_totals() -> pd.DataFrame:
    """
    Group expenses by YYYY-MM and return total spending per month.
    Used for the ML prediction feature.
    """
    conn = get_connection()
    df   = pd.read_sql_query(
        """
        SELECT substr(date, 1, 7) AS month, SUM(amount) AS total
        FROM   expenses
        GROUP  BY month
        ORDER  BY month
        """,
        conn,
    )
    conn.close()
    return df


# ─────────────────────────────────────────────
# CHART COLORS
# ─────────────────────────────────────────────
CATEGORY_COLORS = {
    "food":     "#2ecc71",   # green
    "travel":   "#3498db",   # blue
    "shopping": "#f39c12",   # orange
    "other":    "#e91e8c",   # pink
}

# ─────────────────────────────────────────────
# HERO TITLE
# ─────────────────────────────────────────────
st.markdown(
    '<p class="hero-title">💸 AI Smart Expense Tracker</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="hero-sub">Track • Analyse • Predict — built for Teenagers 🎯</p>',
    unsafe_allow_html=True,
)
st.markdown("---")

# ─────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────
st.sidebar.markdown("## 🧭 Navigation")
st.sidebar.markdown("Choose what you want to do 👇")

PAGES = [
    "➕  Add Expense",
    "📋  View Expenses",
    "📊  Spending Chart",
    "📅  Monthly Summary",
    "🤖  Predict Next Month",
]

page = st.sidebar.radio("", PAGES)

st.sidebar.markdown("---")
st.sidebar.markdown("**💡 Quick Tips**")
st.sidebar.info(
    "Use simple words in the description so the AI can detect your category automatically!\n\n"
    "e.g. *'pizza with friends'* → Food 🍕"
)

# ══════════════════════════════════════════════
# PAGE 1 – ADD EXPENSE
# ══════════════════════════════════════════════
if page == "➕  Add Expense":
    st.markdown('<p class="section-header">➕ Add a New Expense</p>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("#### 💵 Amount (₹)")
        amount = st.number_input(
            "Enter amount",
            min_value=0.0,
            step=1.0,
            format="%.2f",
            label_visibility="collapsed",
        )

        st.markdown("#### 📝 Description")
        description = st.text_input(
            "What did you spend on?",
            placeholder="e.g. pizza with friends, uber to school, new shoes...",
            label_visibility="collapsed",
        )

        if st.button("💾 Save Expense"):
            if amount <= 0:
                st.error("⚠️  Please enter an amount greater than zero.")
            elif not description.strip():
                st.error("⚠️  Please enter a description.")
            else:
                category = add_expense(amount, description.strip())

                # Colour-coded success message
                colour_map = {
                    "food": "🍔", "travel": "🚗",
                    "shopping": "🛍️", "other": "📦",
                }
                emoji = colour_map.get(category, "📦")

                st.success(
                    f"✅ Expense saved!  "
                    f"{emoji} Auto-detected category: **{category.upper()}**"
                )
                st.balloons()

    with col2:
        # Show category keyword hints
        st.markdown("#### 🤖 AI Category Detection")
        st.markdown("The AI reads your description and picks the category:")

        hint_data = {
            "Category": ["🍕 Food", "🚗 Travel", "🛍️ Shopping", "📦 Other"],
            "Example Keywords": [
                "pizza, burger, cafe, biryani, zomato…",
                "uber, bus, metro, fuel, rapido…",
                "shirt, shoes, amazon, flipkart, myntra…",
                "anything else",
            ],
        }
        st.dataframe(pd.DataFrame(hint_data), use_container_width=True, hide_index=True)

        st.markdown("#### 📅 Today's Date")
        st.info(f"📆 Your expense will be saved with today's date:  **{date.today().strftime('%B %d, %Y')}**")

# ══════════════════════════════════════════════
# PAGE 2 – VIEW EXPENSES
# ══════════════════════════════════════════════
elif page == "📋  View Expenses":
    st.markdown('<p class="section-header">📋 All Your Expenses</p>', unsafe_allow_html=True)

    df = get_all_expenses()

    if df.empty:
        st.warning("🫙  No expenses yet! Add your first expense using the sidebar.")
    else:
        # Rename columns for display
        display_df = df.rename(columns={
            "id":          "ID",
            "date":        "Date",
            "amount":      "Amount (₹)",
            "category":    "Category",
            "description": "Description",
        })

        # Colour-code category column
        def style_category(val):
            colour_map = {
                "food":     "background-color: #d5f5e3; color: #1e8449;",
                "travel":   "background-color: #d6eaf8; color: #1a5276;",
                "shopping": "background-color: #fdebd0; color: #a04000;",
                "other":    "background-color: #fce4ec; color: #c62828;",
            }
            return colour_map.get(val, "")

        styled = display_df.style.applymap(style_category, subset=["Category"])
        st.dataframe(styled, use_container_width=True, hide_index=True)

        st.markdown(f"**Total records:** {len(df)}  &nbsp;|&nbsp;  **Total spent:** ₹{df['amount'].sum():,.2f}")

# ══════════════════════════════════════════════
# PAGE 3 – SPENDING CHART
# ══════════════════════════════════════════════
elif page == "📊  Spending Chart":
    st.markdown('<p class="section-header">📊 Your Spending Insights</p>', unsafe_allow_html=True)

    df = get_all_expenses()

    if df.empty:
        st.warning("📭  No data to chart yet. Add some expenses first!")
    else:
        # Group spending by category
        category_totals = df.groupby("category")["amount"].sum().reset_index()
        category_totals.columns = ["Category", "Total (₹)"]
        category_totals = category_totals.sort_values("Total (₹)", ascending=False)

        # Build colours list matching our category order
        bar_colours = [CATEGORY_COLORS.get(c, "#aaa") for c in category_totals["Category"]]

        # --- Matplotlib figure ---
        fig, ax = plt.subplots(figsize=(8, 4.5))
        fig.patch.set_facecolor("#f8f9ff")
        ax.set_facecolor("#f8f9ff")

        bars = ax.bar(
            category_totals["Category"],
            category_totals["Total (₹)"],
            color=bar_colours,
            width=0.5,
            edgecolor="white",
            linewidth=1.5,
        )

        # Add value labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"₹{height:,.0f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
                color="#333",
            )

        ax.set_title("Your Spending Insights 💸", fontsize=16, fontweight="bold", color="#3a3a5c", pad=14)
        ax.set_xlabel("Category", fontsize=12, color="#555")
        ax.set_ylabel("Total Amount (₹)", fontsize=12, color="#555")
        ax.tick_params(colors="#555")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#ddd")
        ax.spines["bottom"].set_color("#ddd")

        # Legend patches
        legend_patches = [
            mpatches.Patch(color=CATEGORY_COLORS["food"],     label="🍕 Food"),
            mpatches.Patch(color=CATEGORY_COLORS["travel"],   label="🚗 Travel"),
            mpatches.Patch(color=CATEGORY_COLORS["shopping"], label="🛍️ Shopping"),
            mpatches.Patch(color=CATEGORY_COLORS["other"],    label="📦 Other"),
        ]
        ax.legend(handles=legend_patches, loc="upper right", framealpha=0.5)

        plt.tight_layout()
        st.pyplot(fig)

        # Pie chart as bonus
        st.markdown("#### 🥧 Category Share")
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        fig2.patch.set_facecolor("#f8f9ff")
        wedges, texts, autotexts = ax2.pie(
            category_totals["Total (₹)"],
            labels=category_totals["Category"],
            colors=[CATEGORY_COLORS.get(c, "#aaa") for c in category_totals["Category"]],
            autopct="%1.1f%%",
            startangle=140,
            pctdistance=0.82,
            wedgeprops=dict(width=0.6, edgecolor="white"),
        )
        for t in autotexts:
            t.set_fontsize(10)
            t.set_fontweight("bold")
        ax2.set_title("Spending Share", fontsize=13, fontweight="bold", color="#3a3a5c")
        plt.tight_layout()
        st.pyplot(fig2)

# ══════════════════════════════════════════════
# PAGE 4 – MONTHLY SUMMARY
# ══════════════════════════════════════════════
elif page == "📅  Monthly Summary":
    st.markdown('<p class="section-header">📅 Monthly Summary</p>', unsafe_allow_html=True)

    now   = datetime.now()
    year  = st.selectbox("Select Year",  list(range(2022, now.year + 1)), index=now.year - 2022)
    month = st.selectbox("Select Month", list(range(1, 13)), index=now.month - 1,
                         format_func=lambda m: datetime(2000, m, 1).strftime("%B"))

    df = get_monthly_expenses(year, month)

    month_label = datetime(year, month, 1).strftime("%B %Y")
    st.markdown(f"#### 📆 Summary for **{month_label}**")

    if df.empty:
        st.warning(f"📭  No expenses found for {month_label}.")
    else:
        total_spent   = df["amount"].sum()
        num_expenses  = len(df)
        top_category  = df.groupby("category")["amount"].sum().idxmax()

        # Metric cards (3 columns)
        c1, c2, c3 = st.columns(3)
        c1.metric("💰 Total Spent",        f"₹{total_spent:,.2f}")
        c2.metric("🔢 No. of Expenses",     str(num_expenses))
        c3.metric("🏆 Highest Category",    top_category.capitalize())

        st.markdown("---")

        # Mini bar chart for the selected month
        cat_group = df.groupby("category")["amount"].sum().reset_index()
        colours   = [CATEGORY_COLORS.get(c, "#aaa") for c in cat_group["category"]]

        fig, ax = plt.subplots(figsize=(6, 3))
        fig.patch.set_facecolor("#f8f9ff")
        ax.set_facecolor("#f8f9ff")
        ax.bar(cat_group["category"], cat_group["amount"], color=colours, width=0.4,
               edgecolor="white", linewidth=1.5)
        ax.set_title(f"Spending in {month_label}", fontsize=13, fontweight="bold", color="#3a3a5c")
        ax.set_ylabel("₹ Amount", fontsize=10, color="#555")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)

        # Expense list
        st.markdown("#### 🗂️ Expense Details")
        st.dataframe(
            df[["date", "amount", "category", "description"]].rename(columns={
                "date": "Date", "amount": "Amount (₹)",
                "category": "Category", "description": "Description",
            }),
            use_container_width=True,
            hide_index=True,
        )

# ══════════════════════════════════════════════
# PAGE 5 – PREDICT NEXT MONTH  (ML feature)
# ══════════════════════════════════════════════
elif page == "🤖  Predict Next Month":
    st.markdown('<p class="section-header">🤖 AI Spending Prediction</p>', unsafe_allow_html=True)
    st.markdown(
        "The AI uses **Linear Regression** to look at your past monthly spending "
        "and predict how much you might spend next month. 📈"
    )

    monthly_df = get_monthly_totals()

    if len(monthly_df) < 2:
        st.warning(
            "🤏  Not enough data yet!  "
            "Please add expenses across at least **2 different months** to enable prediction."
        )
        st.info("💡 Tip: Add a few expenses with different dates to build up your history.")
    else:
        # ── Feature Engineering ──────────────────────────
        # X = month index (1, 2, 3, …)   Y = total spending
        monthly_df["month_index"] = np.arange(1, len(monthly_df) + 1)

        X = monthly_df[["month_index"]].values   # 2-D array (required by sklearn)
        y = monthly_df["total"].values

        # ── Train the model ──────────────────────────────
        model = LinearRegression()
        model.fit(X, y)

        # ── Predict next month ───────────────────────────
        next_index     = len(monthly_df) + 1
        predicted_val  = model.predict([[next_index]])[0]
        predicted_val  = max(predicted_val, 0)   # spending can't be negative

        last_actual    = monthly_df["total"].iloc[-1]
        trend_up       = predicted_val > last_actual  # True = overspend trend

        # ── Display results ──────────────────────────────
        col_a, col_b = st.columns(2)
        col_a.metric("📊 Last Month Actual",    f"₹{last_actual:,.2f}")
        col_b.metric("🔮 Predicted Next Month", f"₹{predicted_val:,.2f}",
                     delta=f"{'▲' if trend_up else '▼'} ₹{abs(predicted_val - last_actual):,.2f}")

        if trend_up:
            st.error("⚠️  You may **overspend** next month. Consider cutting back on non-essentials! 🛑")
        else:
            st.success("🎉  You may **increase your savings** next month. Great job! 💪")

        # ── Trend Chart ──────────────────────────────────
        st.markdown("#### 📈 Your Monthly Spending Trend")

        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor("#f8f9ff")
        ax.set_facecolor("#f8f9ff")

        months_labels = monthly_df["month"].tolist() + ["Next Month"]
        all_amounts   = monthly_df["total"].tolist() + [predicted_val]

        # Actual spending line
        ax.plot(
            monthly_df["month"],
            monthly_df["total"],
            marker="o",
            color="#6c63ff",
            linewidth=2.5,
            markersize=8,
            label="Actual Spending",
            zorder=3,
        )

        # Predicted point (dashed connection)
        ax.plot(
            [monthly_df["month"].iloc[-1], "Next Month"],
            [last_actual, predicted_val],
            linestyle="--",
            color="#f77f00",
            linewidth=2,
            marker="*",
            markersize=14,
            label="Predicted",
            zorder=3,
        )

        # Shade area under actual line
        ax.fill_between(
            monthly_df["month"],
            monthly_df["total"],
            alpha=0.12,
            color="#6c63ff",
        )

        ax.set_xticks(range(len(months_labels)))
        ax.set_xticklabels(months_labels, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("₹ Amount", fontsize=11, color="#555")
        ax.set_title("Monthly Spending + Next Month Prediction", fontsize=13,
                     fontweight="bold", color="#3a3a5c")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(framealpha=0.4)
        plt.tight_layout()

        st.pyplot(fig)

        # ── How it works explainer ───────────────────────
        with st.expander("🧠 How does the AI prediction work?"):
            st.markdown(
                """
                1. **Data collection** – all your expenses are grouped by month to get a total per month.
                2. **Feature** – each month gets a sequential number (Month 1, 2, 3…).
                3. **Linear Regression** – sklearn draws the best-fit line through your historical totals.
                4. **Prediction** – the model extends that line to the *next* month number to guess future spending.

                > Linear Regression is one of the most fundamental Machine Learning algorithms.
                > It finds the relationship between your month number and spending amount,
                > then uses that relationship to predict the future.
                """
            )

        # ── Monthly data table ───────────────────────────
        st.markdown("#### 📋 Historical Monthly Data")
        st.dataframe(
            monthly_df[["month", "total"]].rename(
                columns={"month": "Month", "total": "Total Spent (₹)"}
            ),
            use_container_width=True,
            hide_index=True,
        )

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#aaa; font-size:0.85rem;'>"
    "💸 AI Smart Expense Tracker for Teenagers &nbsp;•&nbsp; "
    "Built with Python, Streamlit, SQLite, Matplotlib &amp; scikit-learn"
    "</p>",
    unsafe_allow_html=True,
)
