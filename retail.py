from datetime import datetime, timedelta
import os
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import streamlit as st

st.set_page_config(
    page_title="Retail Analytics Dashboard",
    page_icon="📊",
    layout="wide",
)

st.markdown(
    """
    <style>
        .main { background-color: #0f172a; color: #f8fafc; }
        .stMetric { background: #1e293b; padding: 15px; border-radius: 10px; border: 1px solid #334155; }
        .stMetric label { color: #94a3b8 !important; text-transform: uppercase; font-size: 0.8rem; }
        .stMetric [data-testid="stMetricValue"] { color: #f8fafc !important; font-size: 1.6rem !important; }
        h1, h2, h3 { color: #38bdf8 !important; }
    </style>
""",
    unsafe_allow_html=True,
)

CSV_FILE_PATH = "transactions.csv"

categories = ["Electronics", "Apparel", "Home & Kitchen", "Beauty", "Sports"]
products = {
    "Electronics": ["Smartphone", "Wireless Earbuds", "Smartwatch", "Bluetooth Speaker"],
    "Apparel": ["Denim Jeans", "Cotton T-Shirt", "Hoodie", "Sneakers"],
    "Home & Kitchen": ["Coffee Maker", "Blender", "Air Fryer", "Desk Lamp"],
    "Beauty": ["Facial Serum", "Moisturizer", "Sunscreen", "Lip Balm"],
    "Sports": ["Yoga Mat", "Dumbbell Set", "Resistance Bands", "Water Bottle"],
}


def generate_mock_data():
  data = []
  base_date = datetime.now() - timedelta(days=180)

  for i in range(1, 201):
    customer_id = f"CUST-{np.random.randint(100, 130):03d}"
    cat = np.random.choice(categories)
    prod = np.random.choice(products[cat])
    qty = int(np.random.randint(1, 5))
    unit_price = float(
        np.random.choice([299, 499, 999, 1499, 2999, 4999, 9999])
    )
    tx_date = base_date + timedelta(
        days=int(np.random.randint(0, 180)),
        hours=int(np.random.randint(0, 23)),
    )

    data.append({
        "customer_id": customer_id,
        "product_name": prod,
        "category": cat,
        "quantity": qty,
        "unit_price": unit_price,
        "transaction_date": tx_date.strftime("%Y-%m-%d %H:%M:%S"),
    })

  df = pd.DataFrame(data)
  df.to_csv(CSV_FILE_PATH, index=False)


@st.cache_data
def load_transaction_data():
  if not os.path.exists(CSV_FILE_PATH):
    generate_mock_data()

  df = pd.read_csv(CSV_FILE_PATH)
  df["transaction_date"] = pd.to_datetime(df["transaction_date"])
  df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(1)
  df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce").fillna(0)
  df["total_amount"] = df["quantity"] * df["unit_price"]
  return df


df = load_transaction_data()

st.title("Retail Analytics Dashboard")
st.markdown(
    '<p style="color: #94a3b8; margin-top: -15px;">Connected to Data Source'
    f" (<code>{CSV_FILE_PATH}</code>)</p>",
    unsafe_allow_html=True,
)

total_revenue = float(df["total_amount"].sum())
total_orders = int(len(df))
unique_customers = int(df["customer_id"].nunique())
aov = float(total_revenue / total_orders) if total_orders > 0 else 0.0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Revenue", f"₹{total_revenue:,.2f}")
col2.metric("Total Orders", f"{total_orders:,}")
col3.metric("Unique Customers", f"{unique_customers:,}")
col4.metric("Average Order Value (AOV)", f"₹{aov:,.2f}")

st.markdown("---")

st.sidebar.header("Filters")
search_query = st.sidebar.text_input(
    "Search Transaction", placeholder="Search Customer or Product..."
)
all_categories = ["ALL"] + sorted(df["category"].dropna().unique().tolist())
selected_category = st.sidebar.selectbox("Category", all_categories)

filtered_df = df.copy()
if search_query:
  query = search_query.lower()
  filtered_df = filtered_df[
      filtered_df["customer_id"].str.lower().str.contains(query, na=False)
      | filtered_df["product_name"].str.lower().str.contains(query, na=False)
  ]

if selected_category != "ALL":
  filtered_df = filtered_df[filtered_df["category"] == selected_category]


def get_rfm_segmentation(dataframe):
  if dataframe.empty:
    return pd.DataFrame()

  snapshot_date = dataframe["transaction_date"].max() + timedelta(days=1)

  rfm = (
      dataframe.groupby("customer_id")
      .agg({
          "transaction_date": lambda x: (snapshot_date - x.max()).days,
          "customer_id": "count",
          "total_amount": "sum",
      })
      .rename(columns={
          "transaction_date": "Recency",
          "customer_id": "Frequency",
          "total_amount": "Monetary",
      })
  )

  if len(rfm) < 3:
    rfm["Segment"] = "Standard Customer"
    summary = (
        rfm.groupby("Segment")
        .agg({
            "Recency": "mean",
            "Frequency": "mean",
            "Monetary": "mean",
            "Segment": "count",
        })
        .rename(columns={"Segment": "Count"})
        .reset_index()
    )
    return summary

  scaler = StandardScaler()
  rfm_scaled = scaler.fit_transform(rfm)

  kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
  rfm["Cluster"] = kmeans.fit_predict(rfm_scaled)

  cluster_means = (
      rfm.groupby("Cluster")["Monetary"].mean().sort_values(ascending=False)
  )
  labels = {
      cluster_means.index[0]: "High-Value VIP",
      cluster_means.index[1]: "Loyal Regular",
      cluster_means.index[2]: "At-Risk / Low Spender",
  }
  rfm["Segment"] = rfm["Cluster"].map(labels)

  summary = (
      rfm.groupby("Segment")
      .agg({
          "Recency": "mean",
          "Frequency": "mean",
          "Monetary": "mean",
          "Cluster": "count",
      })
      .rename(columns={"Cluster": "Count"})
      .reset_index()
  )

  return summary


chart_col1, chart_col2 = st.columns(2)

with chart_col1:
  st.subheader("Customer Segments (RFM Clustering)")
  rfm_data = get_rfm_segmentation(df)
  if not rfm_data.empty:
    fig_rfm = px.pie(
        rfm_data,
        names="Segment",
        values="Count",
        hole=0.4,
        color_discrete_sequence=["#38bdf8", "#4ade80", "#f87171", "#fbbf24"],
    )
    fig_rfm.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#f8fafc",
        margin=dict(t=20, b=20, l=20, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
    )
    st.plotly_chart(fig_rfm, use_container_width=True)
  else:
    st.info("No RFM data available.")

with chart_col2:
  st.subheader("Revenue by Category")
  cat_totals = (
      filtered_df.groupby("category")["total_amount"].sum().reset_index()
  )
  if not cat_totals.empty:
    fig_cat = px.bar(
        cat_totals,
        x="category",
        y="total_amount",
        labels={"category": "", "total_amount": "Revenue (₹)"},
        color_discrete_sequence=["#38bdf8"],
    )
    fig_cat.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#f8fafc",
        margin=dict(t=20, b=20, l=20, r=20),
        xaxis=dict(showgrid=False),
        yaxis=dict(gridcolor="#334155"),
    )
    st.plotly_chart(fig_cat, use_container_width=True)
  else:
    st.info("No category data available.")

st.subheader("RFM Segmentation Breakdown")
if not rfm_data.empty:
  display_rfm = rfm_data.copy()
  display_rfm["Recency"] = display_rfm["Recency"].round(0).astype(int).astype(str) + " days ago"
  display_rfm["Frequency"] = display_rfm["Frequency"].round(1).astype(str) + " orders"
  display_rfm["Monetary"] = display_rfm["Monetary"].apply(lambda x: f"₹{x:,.2f}")
  display_rfm = display_rfm.rename(columns={
      "Segment": "Segment",
      "Count": "Customer Count",
      "Recency": "Avg Recency",
      "Frequency": "Avg Frequency",
      "Monetary": "Avg Monetary Value (₹)",
  })
  st.dataframe(display_rfm, use_container_width=True, hide_index=True)
else:
  st.info("No RFM segmentation data available.")

st.subheader("Complete Transaction Records")
if not filtered_df.empty:
  display_tx = filtered_df.sort_values(by="transaction_date", ascending=False).copy()
  display_tx["unit_price"] = display_tx["unit_price"].apply(lambda x: f"₹{x:,.2f}")
  display_tx["total_amount"] = display_tx["total_amount"].apply(lambda x: f"₹{x:,.2f}")
  display_tx["transaction_date"] = display_tx["transaction_date"].dt.strftime(
      "%Y-%m-%d %H:%M:%S"
  )
  display_tx = display_tx.rename(columns={
      "customer_id": "Customer ID",
      "product_name": "Product Name",
      "category": "Category",
      "quantity": "Qty",
      "unit_price": "Unit Price (₹)",
      "total_amount": "Total Amount (₹)",
      "transaction_date": "Date",
  })
  st.dataframe(display_tx, use_container_width=True, hide_index=True)
else:
  st.info("No matching transaction records found.")