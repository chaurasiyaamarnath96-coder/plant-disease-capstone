from database import get_predictions

df = get_predictions()
import streamlit as st

st.metric(
    "Total Predictions", len(df)
)
st.metric("Average Confidence", round(df["confidence"].mean()*100, 2))
Disease = df["disease"].value_counts()
st.bar_chart(Disease)
st.dataframe(df)