import os
import streamlit as st
import pandas as pd
import psycopg2
import plotly.express as px
from wordcloud import WordCloud

# Load environment variables
NEONDB_URI = os.getenv("NEONDB_URI")

# Streamlit app settings
st.set_page_config(page_title="NYT Business News", page_icon="📰", layout="wide")
st.title("📰 New York Times - Business News Explorer")
st.markdown("Explore and analyze business articles extracted by the ETL pipeline.")

# Load data from the database
@st.cache_data
def load_data():
    """Load data from the database."""
    with psycopg2.connect(NEONDB_URI) as conn:
        query_articles = """
            SELECT title, abstract, published_date
            FROM nyt_business_articles
            ORDER BY published_date DESC
        """
        df_articles = pd.read_sql(query_articles, conn)
        
        query_sentiments = """
            SELECT date, sentiment_label, prediction
            FROM sentiment_forecasts
            ORDER BY date DESC
        """
        df_sentiments = pd.read_sql(query_sentiments, conn)
    
    df_articles["published_date"] = pd.to_datetime(df_articles["published_date"])
    df_sentiments["date"] = pd.to_datetime(df_sentiments["date"])
    return df_articles, df_sentiments

df, df_sentiments = load_data()


# Sidebar filters
st.sidebar.header("📌 Filters")
date_range = st.sidebar.date_input(
    "📅 Select a date range",
    [
        min(df["published_date"].min(), df_sentiments["date"].min()),
        max(df["published_date"].max(), df_sentiments["date"].max()),
    ],
)

# Ensure it's a tuple of two values
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    st.error("Please select a valid date range.")
    st.stop()

# Filter data based on user input
df_filtered = df[
    (df["published_date"] >= pd.Timestamp(start_date))
    & (df["published_date"] <= pd.Timestamp(end_date))
]

df_sentiments_filtered = df_sentiments[
    (df_sentiments["date"] >= pd.Timestamp(start_date))
    & (df_sentiments["date"] <= pd.Timestamp(end_date))
]


# Keyword search
search_query = st.sidebar.text_input("🔍 Search by keyword (title or abstract)")
if search_query:
    df_filtered = df_filtered[
        df_filtered["title"].str.contains(search_query, case=False, na=False)
        | df_filtered["abstract"].str.contains(search_query, case=False, na=False)
    ]

# Display filtered data
st.subheader("📋 Recent Articles")
if df_filtered.empty:
    st.warning("No articles found for the selected filters.")
else:
    with st.expander("📖 Click to view articles"):
        st.dataframe(
            df_filtered[["published_date", "title", "abstract"]].reset_index(drop=True)
        )

st.divider()

# Visualizations
st.subheader("📊 Number of Articles Published Per Day")
df_count = (
    df_filtered.groupby(df_filtered["published_date"].dt.date)
    .size()
    .reset_index(name="Number of Articles")
)
fig = px.bar(
    df_count,
    x="published_date",
    y="Number of Articles",
    title="Articles Published Per Day",
    labels={"published_date": "Date", "Number of Articles": "Count"},
    color="Number of Articles",
    color_continuous_scale="blues",
)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# Sentiment analysis visualization
st.subheader("📊 Sentiment Forecasts Over Time")
if not df_sentiments_filtered.empty:
    fig_sentiment = px.line(
        df_sentiments_filtered,
        x="date",
        y="prediction",
        color="sentiment_label",
        title="Sentiment Predictions Over Time",
        labels={"date": "Date", "prediction": "Prediction Score"},
    )
    st.plotly_chart(fig_sentiment, use_container_width=True)
else:
    st.info("No sentiment data available for the selected date range.")

st.divider()

# Word cloud of keywords in titles
st.subheader("📌 Most Frequent Keywords in Titles")
if not df_filtered.empty:
    TEXT_TITLE = " ".join(df_filtered["title"].dropna())
    wordcloud = WordCloud(width=600, height=300, background_color="white").generate(TEXT_TITLE)
    st.image(wordcloud.to_array(), use_container_width=False, width=600)
else:
    st.info("No data available for word cloud generation.")

# Footer
st.markdown(
    """
    ---
    Powered by Apache Airflow | PostgreSQL | Streamlit
    """
)
