from supabase import create_client
import streamlit as st

supabase = create_client(
    st.secrets["SUPABASE_URL"],
    st.secrets["SUPABASE_KEY"]
)

def save_report(data):
    supabase.table("thermal_results").insert(data).execute()