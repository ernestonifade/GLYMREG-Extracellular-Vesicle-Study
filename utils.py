import pandas as pd
import streamlit as st


def render_searchable_table(df, key_prefix, columns_to_show=None):
  if columns_to_show is None:
    columns_to_show = df.columns.tolist()

  # Robust CSS to target and hide ONLY the search button inside the dataframe toolbar
  st.markdown(
      """
        <style>
        /* Hide the specific search button tool item in Streamlit dataframes */
        [data-testid="stElementToolbar"] button[title*="Search"],
        [data-testid="stElementToolbar"] [aria-label*="Search"] {
            display: none !important;
        }
        /* Fallback: target by SVG search icon path if title attribute differs */
        [data-testid="stElementToolbar"] button svg path[d*="M15.5 14h-.79l-.28-.27"] {
            display: none !important;
        }
        </style>
    """,
      unsafe_allow_html=True,
  )

  search_key = f"{key_prefix}_search"
  if search_key not in st.session_state:
    st.session_state[search_key] = ""

  col1, col2 = st.columns([4, 1])

  with col1:
    search_query = st.text_input(
        "Search table:",
        placeholder="Type to filter rows...",
        key=search_key,
        label_visibility="collapsed",
    )


  def reset_table():
    st.session_state[search_key] = ""


  with col2:
    if st.button("Reset", key=f"{key_prefix}_reset"):
      reset_table()
      st.rerun()

  filtered_df = df.copy()
  current_query = st.session_state.get(search_key, "").strip().lower()

  if current_query:
    mask = False
    for col in filtered_df.columns:
      if filtered_df[col].dtype == object or pd.api.types.is_string_dtype(
          filtered_df[col]
      ):
        mask = mask | filtered_df[col].astype(str).str.lower().str.contains(
            current_query, na=False
        )
    filtered_df = filtered_df[mask]

  st.dataframe(
      filtered_df[columns_to_show], use_container_width=True, hide_index=True
  )
