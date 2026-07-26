import os
import pandas as pd
import streamlit as st

# Import figure modules from your figures folder
from figures.figure3 import render_figure3, load_fig3_results
from figures.figure4 import render_figure4, load_fig4_results

# --- 1. STREAMLIT PAGE CONFIGURATION ---
st.set_page_config(
    page_title="GLYMREG EV Study Dashboard",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for ultra-compact spacing, full-width tables, and clean layout
st.markdown("""
<style>
    /* Maximize container width and remove dead whitespace */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 98% !important;
    }
    h1 {
        font-size: 1.6rem !important;
        margin-bottom: -5px !important;
    }
    .stCaption {
        margin-bottom: 0.5rem !important;
    }
    /* Force tables to stretch wide and avoid horizontal scrolling */
    div[data-testid="stDataFrame"] {
        width: 100% !important;
    }
    div[data-testid="stDataFrame"] > div {
        width: 100% !important;
    }
    /* Style download button for instant 1-click export */
    .stDownloadButton button {
        background-color: #28a745 !important;
        color: white !important;
        font-weight: bold;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Compact Title Banner
st.title("🧬 GLYMREG Extracellular Vesicle Study")
st.caption("Interactive Manuscript Dashboard & Statistical Summary")

# --- 2. SIDEBAR NAVIGATION & REVIEWER GUIDANCE ---
st.sidebar.header("📌 Navigation")

# Helpful guidance for reviewers unfamiliar with Streamlit's collapsible sidebar
st.sidebar.info(
    "💡 **Tip for Reviewers:** Click the **`>`** or **`<<`** icon at the top-left of the sidebar to hide this menu and expand the dashboard to full screen."
)

selected_figure = st.sidebar.radio(
    "Select Figure:",
    [
        "Figure 1: Baseline Characteristics",
        "Figure 2: Global Proteomic Profiling",
        "Figure 3: Discovery Proteomics (518 Panel)",
        "Figure 4: Targeted Cytokine Validation"
    ],
    index=2
)

st.sidebar.markdown("---")

# --- 3. PAGE ROUTING & RENDER CALLS ---
if selected_figure == "Figure 1: Baseline Characteristics":
    st.header("Figure 1: Participant Baseline Characteristics")
    st.info("Figure 1 module is queued for upload.")

elif selected_figure == "Figure 2: Global Proteomic Profiling":
    st.header("Figure 2: Global Proteomic Profiling")
    st.info("Figure 2 module is queued for upload.")

elif selected_figure == "Figure 3: Discovery Proteomics (518 Panel)":
    render_figure3()

elif selected_figure == "Figure 4: Targeted Cytokine Validation":
    render_figure4()

# --- 4. ONE-CLICK INSTANT EXPORT HANDLER ---
st.sidebar.header("📥 Export Statistical Reports")

if selected_figure == "Figure 3: Discovery Proteomics (518 Panel)":
    ancova_df, posthoc_df, pca_scores_df, perm_df, long_df, _ = load_fig3_results()
    
    out_name = "Figure3_Proteomics_Full_Stats_Report.xlsx"
    with pd.ExcelWriter(out_name, engine='openpyxl') as writer:
        ancova_df.to_excel(writer, sheet_name='RM_ANCOVA_Model_Stats', index=False)
        posthoc_df.to_excel(writer, sheet_name='PostHoc_Pairwise_Contrasts', index=False)
        perm_df.to_excel(writer, sheet_name='PERMANOVA_Summary', index=False)
        pca_scores_df.to_excel(writer, sheet_name='PCA_Scores_and_EV', index=False)
        long_df.to_excel(writer, sheet_name='Raw_Proteomic_Data', index=False)
    
    with open(out_name, "rb") as f:
        st.sidebar.download_button(
            label="📥 Download Figure 3 Report (.xlsx)",
            data=f,
            file_name=out_name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

elif selected_figure == "Figure 4: Targeted Cytokine Validation":
    ancova_df, posthoc_df, df_emm, fig4_long_df = load_fig4_results()
    
    out_name = "Figure4_Cytokine_Full_Stats_Report.xlsx"
    with pd.ExcelWriter(out_name, engine='openpyxl') as writer:
        ancova_df.to_excel(writer, sheet_name='RM_ANCOVA_Model_Stats', index=False)
        posthoc_df.to_excel(writer, sheet_name='PostHoc_Pairwise_Contrasts', index=False)
        df_emm.to_excel(writer, sheet_name='Group_EMM_Summary', index=False)
        fig4_long_df.to_excel(writer, sheet_name='Processed_Cytokine_Data', index=False)
    
    with open(out_name, "rb") as f:
        st.sidebar.download_button(
            label="📥 Download Figure 4 Report (.xlsx)",
            data=f,
            file_name=out_name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
