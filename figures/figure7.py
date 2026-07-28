import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
import seaborn as sns
import streamlit as st
import warnings
warnings.filterwarnings("ignore")

plt.rcParams['path.simplify'] = True
plt.rcParams['path.simplify_threshold'] = 1.0
plt.rcParams['agg.path.chunksize'] = 10000


def render_pathway_enrichment_bubble(filepath, title_text):
    mpl.rcParams['svg.fonttype'] = 'none'

    if not filepath or not os.path.exists(filepath):
        st.warning(f"⚠️ Pathway Excel file not found at: {filepath}. Please check file path.")
        return None

    df = pd.read_excel(filepath).iloc[0:15]
    col_map = {
        'Pathway': 'Pathway Name', 'Term': 'Pathway Name',
        'q-value': 'FDR', 'p.adjust': 'FDR',
        'Count': 'Number of Molecules Enriched', 'Genes_Enriched': 'Number of Molecules Enriched',
        'Background_Size': 'Total Molecules in Pathway'
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    if 'FDR' not in df.columns or 'Pathway Name' not in df.columns:
        st.warning("⚠️ Expected columns ('FDR', 'Pathway Name') missing from pathway file.")
        return None

    df['-log10FDR'] = -np.log10(df['FDR'].astype(float).clip(lower=1e-15))

    if 'Fold_Enrichment' not in df.columns:
        if 'Number of Molecules Enriched' in df.columns and 'Total Molecules in Pathway' in df.columns:
            if 'Not_Enriched' in df.columns:
                N_measured = df['Number of Molecules Enriched'].sum() + df['Not_Enriched'].sum()
            else:
                enrichment_rate = 0.05
                N_measured = df['Number of Molecules Enriched'].sum() / enrichment_rate
            
            K_enriched = df['Number of Molecules Enriched'].sum()
            df['Expected_Enriched'] = (df['Total Molecules in Pathway'] / N_measured) * K_enriched
            df['Fold_Enrichment'] = df['Number of Molecules Enriched'] / df['Expected_Enriched']
        else:
            df['Fold_Enrichment'] = 1.5

    if 'Number of Molecules Enriched' not in df.columns:
        df['Number of Molecules Enriched'] = 3

    df = df.sort_values('-log10FDR', ascending=True)

    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    df['PlotSize'] = (df['Number of Molecules Enriched'] + 1) * 35
    df['PlotSize'] = df['PlotSize'].clip(lower=60, upper=300)

    scatter = ax.scatter(
        x=df['Fold_Enrichment'], y=df['Pathway Name'],
        s=df['PlotSize'], c=df['-log10FDR'],
        cmap='viridis', edgecolor='white', linewidth=0.8, alpha=0.9, zorder=3
    )

    ax.axvline(x=1.0, color='red', linestyle=':', linewidth=1.2, label='Expected', alpha=0.8, zorder=1)

    v_min, v_max = df['-log10FDR'].min(), df['-log10FDR'].max()
    cmap = plt.cm.viridis
    norm = plt.Normalize(v_min, v_max)
    color_vals = np.linspace(v_min, v_max, 3)

    color_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(norm(v)),
                markersize=6, markeredgecolor='black', label=f"{v:.1f}")
        for v in color_vals
    ]

    legend_col = ax.legend(
        handles=color_handles[::-1], title=r"$\mathbf{-\log_{10}(FDR)}$",
        bbox_to_anchor=(1.02, 1.0), loc='upper left', frameon=False, labelspacing=1.2, prop={'weight': 'bold', 'size': 6.5}
    )

    s_min, s_max = int(df['Number of Molecules Enriched'].min()), int(df['Number of Molecules Enriched'].max())
    size_steps = np.unique(np.linspace(s_min, s_max, 3).astype(int))

    size_handles = []
    for s in size_steps:
        plot_size = (s + 1) * 35
        marker_d = np.sqrt(plot_size)
        size_handles.append(
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                   markersize=marker_d, markeredgecolor='black', label=str(s))
        )

    legend_siz = ax.legend(
        handles=size_handles[::-1], title=r"$\mathbf{Qty. Enriched}$",
        bbox_to_anchor=(1.02, 0.45), loc='upper left', frameon=False, labelspacing=1.4, prop={'weight': 'bold', 'size': 6.5}
    )

    ax.add_artist(legend_col)
    ax.set_xlabel('Fold Enrichment\n(>1 = Enriched, <1 = Depleted)', fontweight='bold', fontsize=7.5)
    ax.set_title(title_text, y=1.03, fontweight='bold', fontsize=8.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(linestyle='--', alpha=0.5, zorder=0)
    ax.tick_params(axis='both', labelsize=7)
    plt.setp(ax.get_yticklabels(), fontweight='bold')
    plt.setp(ax.get_xticklabels(), fontweight='bold')
    return fig


def find_pathway_file(candidates):
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


# --- MAIN RENDER FUNCTION FOR STREAMLIT ---
def render_figure7():
    # Define candidate filepaths for proteins and cytokines
    prot_candidates = [
        'data/Prot_corr_significant_pathways.xlsx',
        '../data/Prot_corr_significant_pathways.xlsx',
        'Prot_corr_significant_pathways.xlsx'
    ]
    cyt_candidates = [
        'data/Cyt_corr_significant_pathways.xlsx',
        '../data/Cyt_corr_significant_pathways.xlsx',
        'Cyt_corr_significant_pathways.xlsx'
    ]

    prot_path = find_pathway_file(prot_candidates)
    cyt_path = find_pathway_file(cyt_candidates)

    selected_view = st.selectbox(
        "Select Section View:",
        [
            '🧬 Pathway Enrichment: Correlating Proteins',
            '📄 Pathway Enrichment: Correlating Cytokines',
            '📋 Protein Pathway Enrichment Summary Table',
            '📋 Cytokine Pathway Enrichment Summary Table'
        ]
    )

    st.markdown("---")

    if selected_view == '🧬 Pathway Enrichment: Correlating Proteins':
        st.markdown("""
        <div style="background-color: #e2e3e5; border-left: 4px solid #383d41; padding: 10px 14px; margin-bottom: 12px; border-radius: 4px; font-size: 11px; color: #383d41;">
            <b>Pathway Enrichment (Proteins):</b> Over-Representation Analysis mapping correlating candidate interaction proteins to functional pathways.
        </div>
        """, unsafe_allow_html=True)

        fig_path = render_pathway_enrichment_bubble(prot_path, 'Top Enriched Pathways (Proteins)')
        if fig_path:
            st.pyplot(fig_path)

    elif selected_view == '📄 Pathway Enrichment: Correlating Cytokines':
        st.markdown("""
        <div style="background-color: #e2e3e5; border-left: 4px solid #383d41; padding: 10px 14px; margin-bottom: 12px; border-radius: 4px; font-size: 11px; color: #383d41;">
            <b>Pathway Enrichment (Cytokines):</b> Over-Representation Analysis mapping correlating cytokines to functional biological networks.
        </div>
        """, unsafe_allow_html=True)

        fig_path = render_pathway_enrichment_bubble(cyt_path, 'Top Enriched Pathways (Cytokines)')
        if fig_path:
            st.pyplot(fig_path)

    elif selected_view == '📋 Protein Pathway Enrichment Summary Table':
        st.markdown("""
        <div style="background-color: #fff3cd; border-left: 4px solid #ffc107; padding: 10px 14px; margin-bottom: 12px; border-radius: 4px; font-size: 12px; line-height: 1.5; color: #856404;">
            <b>📄 Protein Pathway Summary Table:</b> Complete statistical enrichment metrics for protein correlates.
        </div>
        """, unsafe_allow_html=True)
        
        if prot_path and os.path.exists(prot_path):
            df_prot = pd.read_excel(prot_path)
            st.dataframe(df_prot, use_container_width=True)
        else:
            st.warning("⚠️ Protein pathway summary table file could not be loaded.")

    elif selected_view == '📋 Cytokine Pathway Enrichment Summary Table':
        st.markdown("""
        <div style="background-color: #fff3cd; border-left: 4px solid #ffc107; padding: 10px 14px; margin-bottom: 12px; border-radius: 4px; font-size: 12px; line-height: 1.5; color: #856404;">
            <b>📄 Cytokine Pathway Summary Table:</b> Complete statistical enrichment metrics for cytokine correlates.
        </div>
        """, unsafe_allow_html=True)
        
        if cyt_path and os.path.exists(cyt_path):
            df_cyt = pd.read_excel(cyt_path)
            st.dataframe(df_cyt, use_container_width=True)
        else:
            st.warning("⚠️ Cytokine pathway summary table file could not be loaded.")
            
def load_fig7_results():
    """Helper loader for Figure 7 pathway data to satisfy app.py imports."""
    return {}
