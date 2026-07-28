import os
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pingouin as pg

# --- MATPLOTLIB GLOBAL TYPOGRAPHY SETTINGS ---
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 8,
    'axes.titlesize': 9,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'figure.dpi': 300,
    'savefig.dpi': 600
})

@st.cache_data
def load_fig6_results():
    results_dir = 'results'
    def safe_read(filename):
        path = os.path.join(results_dir, filename)
        if os.path.exists(path):
            return pd.read_csv(path)
        return pd.DataFrame()

    data = {
        'cyto_blood_rm': safe_read('fig6_cytokine_blood_delta_rm_corr.csv'),
        'evsize_rm': safe_read('fig6_protein_evsize_delta_rm_corr.csv'),
        'evconc_rm': safe_read('fig6_protein_evconcentration_delta_rm_corr.csv'),
        'cyto_blood_baseline': safe_read('fig6_cytokine_blood_baseline.csv'),
        'cyto_blood_delta': safe_read('fig6_cytokine_blood_delta_windows_partial_corr.csv'),
        'evsize_baseline': safe_read('fig6_protein_evsize_baseline.csv'),
        'evsize_delta': safe_read('fig6_protein_evsize_delta_windows_partial_corr.csv'),
        'evconc_baseline': safe_read('fig6_protein_evconcentration_baseline.csv'),
        'evconc_delta': safe_read('fig6_protein_evconcentration_delta_windows_partial_corr.csv')
    }
    return data

def render_clustermap(df_source, title_text, xlabel, ylabel):
    if df_source.empty:
        st.warning("⚠️ No data available for this integration view.")
        return

    filtered = df_source[(df_source['p_adj'] < 0.05) & (df_source['r_rm'].abs() > 0.5)].copy()
    if filtered.empty:
        st.info("ℹ️ No feature pairs met the strict filtering criteria (FDR < 0.05 & |r| > 0.5).")
        return

    df_heatmap = filtered.pivot_table(index='Variable_A', columns='Variable_B', values='r_rm').fillna(0.0)
    p_adj_pivot = filtered.pivot_table(index='Variable_A', columns='Variable_B', values='p_adj').fillna(1.0)

    if df_heatmap.shape[0] < 2 or df_heatmap.shape[1] < 2:
        st.warning("⚠️ Insufficient clustered dimensions after applying filters.")
        return

    fig, ax = plt.subplots(figsize=(6, 4.5))
    g = sns.clustermap(
        df_heatmap,
        cmap="RdBu_r",
        vmin=-1, vmax=1,
        linewidths=0.75,
        edgecolor="white",
        cbar_kws={"label": "Repeated Measures Correlation ($r_{rm}$)", "orientation": "horizontal"},
        cbar=True,
        figsize=(6, 5),
        dendrogram_ratio=0.08,
        tree_kws={"linewidths": 1.5, "colors": "#424242"}
    )

    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=45, ha="right")
    plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)

    g.ax_heatmap.xaxis.label.set_text(xlabel)
    g.ax_heatmap.xaxis.label.set_position((0.5, -0.1))
    g.ax_heatmap.yaxis.label.set_text(ylabel)

    g.ax_cbar.set_position([0.15, -0.02, 0.5, 0.03])
    plt.setp(g.ax_cbar.xaxis.label, fontsize=8, fontweight='bold')
    g.ax_cbar.tick_params(axis='x', labelsize=7)
    g.ax_cbar.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))

    g.fig.suptitle(title_text, weight="bold", x=0.45, y=1.03)

    try:
        row_labels_ordered = [label.get_text() for label in g.ax_heatmap.get_yticklabels()]
        col_labels_ordered = [label.get_text() for label in g.ax_heatmap.get_xticklabels()]
        reordered_p_adj = p_adj_pivot.loc[row_labels_ordered, col_labels_ordered]

        for i in range(reordered_p_adj.shape[0]):
            for j in range(reordered_p_adj.shape[1]):
                p_val = reordered_p_adj.iloc[i, j]
                ann = ''
                if p_val < 0.001: ann = '***'
                elif p_val < 0.01: ann = '**'
                elif p_val < 0.05: ann = '*'
                if ann:
                    g.ax_heatmap.text(j + 0.5, i + 0.5, ann, ha='center', va='center', color='black', fontsize=7)
    except Exception:
        pass

    st.pyplot(g.fig)
    plt.close(g.fig)

def render_correlation_plot(df):
    st.subheader("Temporal Correlation: EV Size vs. Histone H2A")
    if df.empty:
        st.warning("⚠️ Master dataframe unavailable for individual correlation plot.")
        return

    target_prot = "P04908;Q7L7L0;Q93077"
    target_ev = "Median Value (nm)"
    
    if target_prot not in df.columns or target_ev not in df.columns:
        st.info("ℹ️ Specific target columns for Histone H2A / EV Size not found in current view dataset.")
        return

    df_rm = df[["time", target_prot, target_ev, "ID"]].copy()
    df_rm = df_rm.rename(columns={target_ev: "Median_Value_nm", target_prot: "P04908_Q7L7L0_Q93077"})
    df_rm = df_rm.dropna(subset=["P04908_Q7L7L0_Q93077", "Median_Value_nm"])

    r_rm = -0.575
    p_adj = 0.0436

    sns.set_style("ticks")
    g = pg.plot_rm_corr(data=df_rm, x="P04908_Q7L7L0_Q93077", y="Median_Value_nm", subject="ID")
    ax = g.ax
    fig = plt.gcf()
    fig.set_size_inches(5, 4)

    plt.setp(ax.lines, alpha=0.8, linewidth=1.5)
    plt.setp(ax.collections, edgecolor="black", linewidth=0.5, sizes=[28], alpha=0.8)

    sns.regplot(
        x="P04908_Q7L7L0_Q93077",
        y="Median_Value_nm",
        data=df_rm,
        scatter=False,
        ax=ax,
        color="black",
        line_kws={"linewidth": 1.5, "zorder": 5},
    )

    stats_text = f"r_rm = {r_rm:.3f}\n95% CI: [-0.75, -0.32]\np_adj = {p_adj:.4f}"
    ax.text(0.05, 0.05, stats_text, transform=ax.transAxes, fontsize=8, fontweight='bold')

    ax.set_xlabel("Histone H2A type 1-C,3,1-B/E\n(Normalized Intensity, AU)", labelpad=12, fontweight='bold')
    ax.set_ylabel("$\Delta$ EV Size (nm)", labelpad=12, fontweight='bold')
    ax.set_title("Temporal Correlation:\nEV Size vs Histone H2A", pad=15, loc="left", fontweight='bold')

    plt.setp(ax.get_xticklabels(), fontweight='bold')
    plt.setp(ax.get_yticklabels(), fontweight='bold')
    sns.despine(trim=True)

    st.pyplot(fig)
    plt.close(fig)

def render_searchable_table(df_input, title_prefix):
    st.subheader(f"{title_prefix} - Repeated Measures Table")
    if df_input.empty:
        st.info("No records loaded.")
        return

    filtered_df = df_input[df_input['p_val'] < 0.05].copy()
    
    col1, col2 = st.columns([3, 1])
    with col1:
        search_query = st.text_input(f"🔍 Search {title_prefix} Features:", "", key=f"search_{title_prefix}")
    with col2:
        st.metric("Significant Pairs (p < 0.05)", len(filtered_df))

    if search_query:
        mask = filtered_df['Variable_A'].str.contains(search_query, case=False, na=False) | \
               filtered_df['Variable_B'].str.contains(search_query, case=False, na=False)
        filtered_df = filtered_df[mask]

    display_cols = [c for c in ['Variable_A', 'Variable_B', 'n', 'df', 'r_rm', 'CI_95%', 'p_val', 'p_adj', 'is_significant'] if c in filtered_df.columns]
    
    st.dataframe(
        filtered_df[display_cols].sort_values(by='p_val'),
        use_container_width=True,
        hide_index=True
    )

def render_partial_corr_view(baseline_df, delta_df, title_prefix):
    st.subheader(f"{title_prefix} - Cross-Sectional & Delta Partial Correlation Dynamics")
    
    time_points = ["Baseline (Time 1)", "Delta (Time 2 vs Time 1)", "Delta (Time 3 vs Time 1)", "Delta (Time 4 vs Time 1)"]
    selected_time = st.radio("⏱️ Jump to Time Point / Delta Window:", time_points, horizontal=True, key=f"radio_{title_prefix}")

    if "Baseline" in selected_time:
        active_df = baseline_df.copy()
    elif "Time 2" in selected_time:
        active_df = delta_df[delta_df['TimeWindow'] == 'delta_time2_vs_time1'].copy() if not delta_df.empty else pd.DataFrame()
    elif "Time 3" in selected_time:
        active_df = delta_df[delta_df['TimeWindow'] == 'delta_time3_vs_time1'].copy() if not delta_df.empty else pd.DataFrame()
    else:
        active_df = delta_df[delta_df['TimeWindow'] == 'delta_time4_vs_time1'].copy() if not delta_df.empty else pd.DataFrame()

    if active_df.empty:
        st.warning("⚠️ Data unavailable for the selected window slice.")
        return

    filtered_active = active_df[active_df['p_val'] < 0.05].copy()

    c1, c2 = st.columns([3, 1])
    with c1:
        query = st.text_input(f"🔍 Search {selected_time} features:", "", key=f"search_part_{title_prefix}")
    with c2:
        st.metric("Filtered Associations", len(filtered_active))

    if query:
        mask = filtered_active['Variable_A'].str.contains(query, case=False, na=False) | \
               filtered_active['Variable_B'].str.contains(query, case=False, na=False)
        filtered_active = filtered_active[mask]

    cols = [c for c in ['Variable_A', 'Variable_B', 'TimeWindow', 'n', 'df', 'r', 'CI_95%', 'p_val', 'p_adj', 'is_significant'] if c in filtered_active.columns]
    st.dataframe(filtered_active[cols].sort_values(by='p_val'), use_container_width=True, hide_index=True)

def render_figure6():
    st.title("🧬 Figure 6: Multi-Modal EV & Clinical Integration Dashboard")
    st.markdown("Explore integrated cross-sectional partial correlations and longitudinal repeated-measures dynamics across extracellular vesicles, cytokine networks, and hematological layers.")

    data = load_fig6_results()

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "1️⃣ Cyto vs Blood (RM Heatmap)",
        "2️⃣ Histone H2A vs EV Size",
        "3️⃣ Cyto vs Blood (RM Table)",
        "4️⃣ Prot vs EV Size (RM Table)",
        "5️⃣ Prot vs EV Conc (RM Table)",
        "6️⃣ Cyto vs Blood (Partial Corr)",
        "7️⃣ Prot vs EV Size (Partial Corr)",
        "8️⃣ Prot vs EV Conc (Partial Corr)"
    ])

    with tab1:
        render_clustermap(data['cyto_blood_rm'], 'Repeated Measures: Cytokines vs. Blood Cells', 'Blood Parameters', 'Cytokines')

    with tab2:
        # Load raw sample dataset fallback if accessible, else pass empty
        df_dummy = pd.DataFrame()
        render_correlation_plot(df_dummy)

    with tab3:
        render_searchable_table(data['cyto_blood_rm'], "Cytokines vs. Blood Cells")

    with tab4:
        render_searchable_table(data['evsize_rm'], "Proteins vs. EV Size")

    with tab5:
        render_searchable_table(data['evconc_rm'], "Proteins vs. EV Concentration")

    with tab6:
        render_partial_corr_view(data['cyto_blood_baseline'], data['cyto_blood_delta'], "Cytokines vs. Blood Cells")

    with tab7:
        render_partial_corr_view(data['evsize_baseline'], data['evsize_delta'], "Proteins vs. EV Size")

    with tab8:
        render_partial_corr_view(data['evconc_baseline'], data['evconc_delta'], "Proteins vs. EV Concentration")
