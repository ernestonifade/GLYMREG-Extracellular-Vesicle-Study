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


def render_pathway_enrichment_bubble_from_df(
    results_input=None,
    database_name="GO_Biological_Process",
    max_pvalue=0.05,
    pathway_indices=None,
):
  mpl.rcParams["svg.fonttype"] = "none"

  # 1. Handle file loading if a path or None is provided, otherwise use the passed dataframe
  if results_input is None or isinstance(results_input, (str, os.PathLike)):
    file_candidates = [
        "data/enrichment_permutation_results_for_correlating_proteins.csv",
        "../data/enrichment_permutation_results_for_correlating_proteins.csv",
        "enrichment_permutation_results_for_correlating_proteins.csv",
        # Fallback to excel just in case
        "enrichment_permutation_results_for_correlating_proteins.xlsx",
    ]

    if isinstance(results_input, (str, os.PathLike)):
      file_candidates.insert(0, str(results_input))

    filepath = None
    for path in file_candidates:
      if os.path.exists(path):
        filepath = path
        break

    if filepath is None:
      st.warning(
          "⚠️ Pathway results file not found. Please check your GitHub file"
          " path."
      )
      return None

    # Load dataframe based on file extension
    if filepath.endswith(".xlsx") or filepath.endswith(".xls"):
      df_master = pd.read_excel(filepath)
    else:
      df_master = pd.read_csv(filepath)
  else:
    df_master = results_input.copy()

  # 2. Handle filtering by database and strict p-value threshold
  if database_name.lower() == "all":
    top_lists = []
    for db, group in df_master.groupby("Database"):
      sig_group = group[group["Empirical_P_Value"] <= max_pvalue]
      top_db = sig_group.sort_values(
          by=["Empirical_P_Value", "Observed_Overlap"], ascending=[True, False]
      ).head(5)
      top_lists.append(top_db)
    df = pd.concat(top_lists, ignore_index=True)
    title_suffix = "Top Significant (p ≤ 0.05) Across All Databases"
  else:
    df = df_master[df_master["Database"] == database_name].copy()
    if df.empty:
      st.warning(f"⚠️ No results found for database: {database_name}")
      return None

    # Filter for significant pathways only
    df = df[df["Empirical_P_Value"] <= max_pvalue]

    df = (
        df.sort_values(
            by=["Empirical_P_Value", "Observed_Overlap"], ascending=[True, False]
        )
        .head(15)
        .copy()
    )
    title_suffix = database_name

  # 3. Apply clean index-based selection if specified
  if pathway_indices is not None and not df.empty:
    df = df.reset_index(drop=True)
    valid_indices = [i for i in pathway_indices if i < len(df)]
    df = df.iloc[valid_indices].copy()
    title_suffix = f"{database_name} (Indexed Selection)"

  if df.empty:
    st.warning(
        "⚠️ No pathways meet the significance threshold (p ≤ "
        f"{max_pvalue}) or valid indices."
    )
    return None

  # Map column names & calculations
  df["Pathway Name"] = df["Pathway"]
  df["Significance"] = df["Empirical_P_Value"]
  df["Number of Molecules Enriched"] = df["Observed_Overlap"]
  df["Fold_Enrichment"] = df["Observed_Overlap"] / df[
      "Mean_Random_Overlap"
  ].replace(0, 0.001)
  df["-log10Sig"] = -np.log10(df["Significance"].astype(float).clip(lower=1e-15))
  df = df.sort_values("-log10Sig", ascending=True)

  fig, ax = plt.subplots(figsize=(4.5, max(4.0, len(df) * 0.25)))
  df["PlotSize"] = (df["Number of Molecules Enriched"] + 1) * 35
  df["PlotSize"] = df["PlotSize"].clip(lower=60, upper=300)

  scatter = ax.scatter(
      x=df["Fold_Enrichment"],
      y=df["Pathway Name"],
      s=df["PlotSize"],
      c=df["-log10Sig"],
      cmap="viridis",
      edgecolor="white",
      linewidth=0.8,
      alpha=0.9,
      zorder=3,
  )

  ax.axvline(
      x=1.0,
      color="red",
      linestyle=":",
      linewidth=1.2,
      label="Expected",
      alpha=0.8,
      zorder=1,
  )

  v_min, v_max = df["-log10Sig"].min(), df["-log10Sig"].max()
  cmap = plt.cm.viridis
  norm = plt.Normalize(v_min, v_max)
  color_vals = np.linspace(v_min, v_max, 3)

  color_handles = [
      Line2D(
          [0],
          [0],
          marker="o",
          color="w",
          markerfacecolor=cmap(norm(v)),
          markersize=6,
          markeredgecolor="black",
          label=f"{v:.1f}",
      )
      for v in color_vals
  ]

  legend_col = ax.legend(
      handles=color_handles[::-1],
      title=r"$\mathbf{-\log_{10}(Emp. P)}$",
      bbox_to_anchor=(1.02, 1.0),
      loc="upper left",
      frameon=False,
      labelspacing=1.2,
      prop={"weight": "bold", "size": 6.5},
  )

  s_min, s_max = int(df["Number of Molecules Enriched"].min()), int(
      df["Number of Molecules Enriched"].max()
  )
  size_steps = np.unique(np.linspace(s_min, s_max, 3).astype(int))

  size_handles = []
  for s in size_steps:
    plot_size = (s + 1) * 35
    marker_d = np.sqrt(plot_size)
    size_handles.append(
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="gray",
            markersize=marker_d,
            markeredgecolor="black",
            label=str(s),
        )
    )

  legend_siz = ax.legend(
      handles=size_handles[::-1],
      title=r"$\mathbf{Qty. Enriched}$",
      bbox_to_anchor=(1.02, 0.45),
      loc="upper left",
      frameon=False,
      labelspacing=1.4,
      prop={"weight": "bold", "size": 6.5},
  )

  ax.add_artist(legend_col)
  ax.set_xlabel(
      "Enrichment Ratio (Obs / Exp)\n(>1 = Enriched, <1 = Depleted)",
      fontweight="bold",
      fontsize=7.5,
  )
  ax.set_title(
      f"Top Enriched Pathways ({title_suffix})",
      y=1.03,
      fontweight="bold",
      fontsize=8.5,
  )
  ax.spines["top"].set_visible(False)
  ax.spines["right"].set_visible(False)
  ax.grid(linestyle="--", alpha=0.5, zorder=0)
  ax.tick_params(axis="both", labelsize=7)
  plt.setp(ax.get_yticklabels(), fontweight="bold")
  plt.setp(ax.get_xticklabels(), fontweight="bold")
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
