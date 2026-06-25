import streamlit as st
import pandas as pd
from supabase import create_client

st.set_page_config(
    page_title="Database Records",
    layout="wide"
)

supabase = create_client(
    st.secrets["SUPABASE_URL"],
    st.secrets["SUPABASE_KEY"]
)

st.title("Thermal Inspection Database")

# Refresh button
st.button("Refresh")

try:

    response = (
        supabase
        .table("thermal_results")
        .select("*")
        .order("id", desc=True)
        .execute()
    )

    df = pd.DataFrame(response.data)

    if not df.empty:

        st.metric("Total Reports", len(df))

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "⬇ Download CSV",
            df.to_csv(index=False),
            "thermal_records.csv",
            "text/csv"
        )

    else:
        st.info("No records found.")

except Exception as e:
    st.error(f"Database Error: {e}")
