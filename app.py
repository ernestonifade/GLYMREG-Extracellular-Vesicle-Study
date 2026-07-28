import os
import pandas as pd
import streamlit as st

# Import figure modules from your figures folder
from figures.figure1 import render_figure1, load_fig1_results
from figures.figure2 import render_figure2, load_fig2_results
from figures.figure3 import render_figure3, load_fig3_results
from figures.figure4 import render_figure4, load_fig4_results
from figures.figure5 import render_figure5, load_fig5_results
from figures.figure6 import render_figure6, load_fig6_results
from figures.figure7 import render_figure7, load_fig7_results

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
        "Figure 1: EV Size Skewness",
        "Figure 2: Extracellular Vesicles (Concentration, Size & Correlation)",
        "Figure 3: Proteomics (518 Panel)",
        "Figure 4: Cytokine Analysis",
        "Figure 5: Protein vs Cytokine vs Blood Correlation",
        "Figure 6: EV vs Protein, Cytokine vs Blood Correlations",
        "Figure 7: Pathway Enrichment Protein, Cytokine Correlations"
    ],
    index=2
)

st.sidebar.markdown("---")

# --- 3. PAGE ROUTING & RENDER CALLS ---
if selected_figure == "Figure 1: EV Size Skewness":
    render_figure1()

elif selected_figure == "Figure 2: Extracellular Vesicles (Concentration, Size & Correlation)":
    render_figure2()

elif selected_figure == "Figure 3: Proteomics (518 Panel)":
    render_figure3()

elif selected_figure == "Figure 4: Cytokine Analysis":
    render_figure4()

elif selected_figure == "Figure 5: Protein vs Cytokine vs Blood Correlation":
    render_figure5()

elif selected_figure == "Figure 6: EV vs Protein, Cytokine vs Blood Correlations":
    render_figure6()

elif selected_figure == "Figure 7: Pathway Enrichment Protein, Cytokine Correlations":
    render_figure7()

# --- 4. ONE-CLICK INSTANT EXPORT HANDLER ---
st.sidebar.header("📥 Export Statistical Reports")

if selected_figure == "Figure 1: EV Size Skewness":
    stats_df, df_all = load_fig1_results()
    
    out_name = "Figure1D_EV_Size_Skewness_Data.xlsx"
    with pd.ExcelWriter(out_name, engine='openpyxl') as writer:
        stats_df.to_excel(writer, sheet_name='EV_Size_Skewness_Wilcoxon_Stats', index=False)
        df_all.to_excel(writer, sheet_name='Raw_EV_Data', index=False)
    
    with open(out_name, "rb") as f:
        st.sidebar.download_button(
            label="📥 Download Figure 1 Report (.xlsx)",
            data=f,
            file_name=out_name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

elif selected_figure == "Figure 2: Extracellular Vesicles (Concentration, Size & Correlation)":
    ancova_df, posthoc_df, corr_overall, corr_sex, long_df = load_fig2_results()
    
    out_name = "Figure2_EV_Full_Stats_Report.xlsx"
    with pd.ExcelWriter(out_name, engine='openpyxl') as writer:
        ancova_df.to_excel(writer, sheet_name='RM_ANCOVA_Stats', index=False)
        posthoc_df.to_excel(writer, sheet_name='PostHoc_Contrasts', index=False)
        corr_overall.to_excel(writer, sheet_name='Correlation_Overall', index=False)
        corr_sex.to_excel(writer, sheet_name='Correlation_Sex', index=False)
        long_df.to_excel(writer, sheet_name='Raw_Data', index=False)
    
    with open(out_name, "rb") as f:
        st.sidebar.download_button(
            label="📥 Download Figure 2 Report (.xlsx)",
            data=f,
            file_name=out_name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

elif selected_figure == "Figure 3: Proteomics (518 Panel)":
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

elif selected_figure == "Figure 4: Cytokine Analysis":
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

elif selected_figure == "Figure 5: Protein vs Cytokine vs Blood Correlation":
    data = load_fig5_results()
    
    out_name = "Figure5_MultiModal_Integration_Report.xlsx"
    with pd.ExcelWriter(out_name, engine='openpyxl') as writer:
        if not data['cyto_rm'].empty:
            data['cyto_rm'].to_excel(writer, sheet_name='Cyto_Protein_RM_Corr', index=False)
        if not data['blood_rm'].empty:
            data['blood_rm'].to_excel(writer, sheet_name='Blood_Protein_RM_Corr', index=False)
        if not data['cyto_baseline'].empty:
            data['cyto_baseline'].to_excel(writer, sheet_name='Cyto_Protein_Baseline', index=False)
        if not data['cyto_delta'].empty:
            data['cyto_delta'].to_excel(writer, sheet_name='Cyto_Protein_Delta_Windows', index=False)
        if not data['blood_baseline'].empty:
            data['blood_baseline'].to_excel(writer, sheet_name='Blood_Protein_Baseline', index=False)
        if not data['blood_delta'].empty:
            data['blood_delta'].to_excel(writer, sheet_name='Blood_Protein_Delta_Windows', index=False)
    
    with open(out_name, "rb") as f:
        st.sidebar.download_button(
            label="📥 Download Figure 5 Report (.xlsx)",
            data=f,
            file_name=out_name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
elif selected_figure == "Figure 6: EV vs Protein, Cytokine vs Blood Correlations":
    data = load_fig6_results()
    
    out_name = "Figure6_EV_MultiModal_Integration_Report.xlsx"
    with pd.ExcelWriter(out_name, engine='openpyxl') as writer:
        if not data['cyto_blood_rm'].empty:
            data['cyto_blood_rm'].to_excel(writer, sheet_name='Cyto_Blood_RM_Corr', index=False)
        if not data['evsize_rm'].empty:
            data['evsize_rm'].to_excel(writer, sheet_name='Protein_EVSize_RM_Corr', index=False)
        if not data['evconc_rm'].empty:
            data['evconc_rm'].to_excel(writer, sheet_name='Protein_EVConc_RM_Corr', index=False)
        if not data['cyto_blood_baseline'].empty:
            data['cyto_blood_baseline'].to_excel(writer, sheet_name='Cyto_Blood_Baseline', index=False)
        if not data['cyto_blood_delta'].empty:
            data['cyto_blood_delta'].to_excel(writer, sheet_name='Cyto_Blood_Delta_Windows', index=False)
        if not data['evsize_baseline'].empty:
            data['evsize_baseline'].to_excel(writer, sheet_name='Protein_EVSize_Baseline', index=False)
        if not data['evsize_delta'].empty:
            data['evsize_delta'].to_excel(writer, sheet_name='Protein_EVSize_Delta_Windows', index=False)
        if not data['evconc_baseline'].empty:
            data['evconc_baseline'].to_excel(writer, sheet_name='Protein_EVConc_Baseline', index=False)
        if not data['evconc_delta'].empty:
            data['evconc_delta'].to_excel(writer, sheet_name='Protein_EVConc_Delta_Windows', index=False)
    
    with open(out_name, "rb") as f:
        st.sidebar.download_button(
            label="📥 Download Figure 6 Report (.xlsx)",
            data=f,
            file_name=out_name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
elif selected_figure == "Figure 7: Pathway Enrichment Protein, Cytokine Correlations":
    data = load_fig7_results() if 'load_fig7_results' in globals() else {}
    
    out_name = "Figure7_Pathway_Enrichment_Report.xlsx"
    
    prot_path = find_pathway_file([
        'data/Prot_corr_significant_pathways.xlsx',
        '../data/Prot_corr_significant_pathways.xlsx',
        'Prot_corr_significant_pathways.xlsx'
    ])
    cyt_path = find_pathway_file([
        'data/Cyt_corr_significant_pathways.xlsx',
        '../data/Cyt_corr_significant_pathways.xlsx',
        'Cyt_corr_significant_pathways.xlsx'
    ])
    
    sheets_written = 0
    with pd.ExcelWriter(out_name, engine='openpyxl') as writer:
        if prot_path and os.path.exists(prot_path):
            df_prot = pd.read_excel(prot_path)
            if not df_prot.empty:
                df_prot.to_excel(writer, sheet_name='Protein_Pathways', index=False)
                sheets_written += 1
            
        if cyt_path and os.path.exists(cyt_path):
            df_cyt = pd.read_excel(cyt_path)
            if not df_cyt.empty:
                df_cyt.to_excel(writer, sheet_name='Cytokine_Pathways', index=False)
                sheets_written += 1
                
        # Fallback empty sheet if both files are missing to prevent openpyxl crash
        if sheets_written == 0:
            pd.DataFrame({'Note': ['No pathway files found']}).to_excel(writer, sheet_name='Info', index=False)
            
    with open(out_name, "rb") as f:
        st.sidebar.download_button(
            label="📥 Download Figure 7 Report (.xlsx)",
            data=f,
            file_name=out_name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
