# import streamlit as st
# import pandas as pd
# from supabase import create_client

# supabase = create_client(
#     st.secrets["SUPABASE_URL"],
#     st.secrets["SUPABASE_KEY"]
# )

# st.set_page_config(
#     page_title="Database Records",
#     layout="wide"
# )

# st.title("📊 Thermal Inspection Database")

# try:

#     response = (
#         supabase
#         .table("thermal_results")
#         .select("*")
#         .order("timestamp", desc=True)
#         .execute()
#     )

#     df = pd.DataFrame(response.data)

#     if not df.empty:
#         st.dataframe(
#             df,
#             use_container_width=True,
#             hide_index=True
#         )

#         st.download_button(
#             "⬇ Download CSV",
#             df.to_csv(index=False),
#             "thermal_records.csv",
#             "text/csv"
#         )

#     else:
#         st.info("No records found.")

# except Exception as e:
#     st.error(f"Database Error: {e}")
import streamlit as st

st.title("DATABASE PAGE TEST")
st.success("If you see this, multipage navigation works.")






