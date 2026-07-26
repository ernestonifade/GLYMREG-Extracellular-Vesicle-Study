import os
import pandas as pd
import streamlit as st

# Import figure modules from your figures folder
from figures.figure3 import render_figure3, load_fig3_results
from figures.figure4 import render_figure4, load_fig4_results

# --- 1. STREAMLIT PAGE CONFIGURATION (WIDE & OPEN) ---
st.set_page_config(
    page_title="GLYMREG EV Study Dashboard",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for compact header, full-width fluid layout, and readable tables
st.markdown("""
<style>
    /* Reduce top whitespace and header size */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 98%;
    }
    h1 {
        font-size: 1.8rem !important;
        margin-bottom: 0px !important;
    }
    p {
        font-size: 0.95rem;
    }
    /* Style download button for single-click instant download */
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
st.markdown("---")

# --- 2. SIDEBAR NAVIGATION ---
st.sidebar.header("📌 Navigation")
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
