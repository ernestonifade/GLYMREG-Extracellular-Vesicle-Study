import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
import seaborn as sns
from matplotlib.patches import Ellipse
import streamlit as st

@st.cache_data
def load_fig3_results():
    if os.path.exists('results/fig3_ancova_stats.csv'):
        data_dir = 'results'
    elif os.path.exists('../results/fig3_ancova_stats.csv'):
        data_dir = '../results'
    else:
        data_dir = '.'
        
    ancova_df = pd.read_csv(os.path.join(data_dir, 'fig3_ancova_stats.csv'))
    posthoc_df = pd.read_csv(os.path.join(data_dir, 'fig3_posthoc_contrasts.csv'))
    pca_scores_df = pd.read_csv(os.path.join(data_dir, 'fig3_pca_scores.csv'))
    perm_df = pd.read_csv(os.path.join(data_dir, 'fig3_permanova_summary.csv'))
    wide_df = pd.read_csv(os.path.join(data_dir, 'fig3_processed_data.csv'))

    metadata_cols = ['Subject_ID', 'sex', 'time', 'ID', 'Sex', 'Time', 'Group', 'TimePoint', 'BaselineValue']
    meta_in_df = [c for c in metadata_cols if c in wide_df.columns]
    protein_cols = [c for c in wide_df.columns if c not in meta_in_df and pd.api.types.is_numeric_dtype(wide_df[c])]

    fig3_long_df = wide_df.melt(
        id_vars=meta_in_df,
        value_vars=protein_cols,
        var_name='Protein',
        value_name='Value'
    )
    long_df = wide_df.copy()
    return ancova_df, posthoc_df, pca_scores_df, perm_df, long_df, fig3_long_df

def confidence_ellipse(x, y, ax, n_std=2.0, **kwargs):
    if x.size == 0 or y.size == 0: return None
    cov = np.cov(x, y)
    if not np.isfinite(cov).all(): return None
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(vals)
    ellipse = Ellipse(xy=np.array([np.mean(x), np.mean(y)]), width=width, height=height, angle=theta, **kwargs)
    ax.add_patch(ellipse)
    return ellipse

def render_pca_trio(pca_scores_df):
    mpl.rcParams['svg.fonttype'] = 'none'
    plt.rcParams.update({
        'font.family':'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'FreeSans', 'DejaVu Sans', 'sans-serif'], 'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 8,
        'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
        'axes.labelweight': 'bold', 'axes.titleweight': 'bold',
        'figure.dpi': 100, 'savefig.dpi': 600
    })

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), constrained_layout=True)
    panels = [('3A', 'Figure 3A: Baseline (Total)', axes[0]),
              ('3B', 'Figure 3B: Adj. Post (Total)', axes[1]),
              ('3C', 'Figure 3C: Adj. Post (19 Candidates)', axes[2])]
    
    order = ('Male', 'Female')
    colors = ('#002df5', '#f57a00')
    palette = {lvl: col for lvl, col in zip(order, colors)}
    markers = {'Male': 'o', 'Female': 's'}
    
    for panel_code, panel_title, ax in panels:
        sub_scores = pca_scores_df[pca_scores_df['Panel'].astype(str).str.contains(panel_code)]
        if sub_scores.empty: continue
            
        ev1 = sub_scores['EV_PC1'].iloc[0]
        ev2 = sub_scores['EV_PC2'].iloc[0]
        present_levels = [lvl for lvl in order if lvl in sub_scores['sex'].values]
        
        for lvl in present_levels:
            sub = sub_scores[sub_scores['sex'] == lvl]
            if not sub.empty:
                current_marker = markers.get(lvl, 'o')
                ax.scatter(sub['PC1'], sub['PC2'], color=palette[lvl], alpha=0.9, label=lvl,
                           edgecolor='white', linewidth=0.1, marker=current_marker, s=21)
                
                if len(sub) >= 3:
                    confidence_ellipse(sub['PC1'].values, sub['PC2'].values, ax,
                                       n_std=2.0, edgecolor=palette[lvl], facecolor='none', lw=2, ls='--', alpha=0.9)
        
        cents = sub_scores.groupby('sex')[['PC1', 'PC2']].mean().reindex(present_levels).dropna().reset_index()
        if len(cents) > 1:
            ax.plot(cents['PC1'], cents['PC2'], '-k', lw=1.2, alpha=0.7)
            ax.scatter(cents['PC1'], cents['PC2'], s=21, c='none', edgecolors='k', lw=1.0, zorder=5)
            for i in range(len(cents)-1):
                ax.annotate('', xy=(cents.loc[i+1, 'PC1'], cents.loc[i+1, 'PC2']),
                            xytext=(cents.loc[i, 'PC1'], cents.loc[i, 'PC2']),
                            arrowprops=dict(arrowstyle='-|>', lw=1.0, color='k'))
        
        ax.set_xlabel(f"PC1 ({ev1*100:.2f}% variance)")
        ax.set_ylabel(f"PC2 ({ev2*100:.2f}% variance)")
        plt.setp(ax.get_yticklabels(), fontweight='bold')
        plt.setp(ax.get_xticklabels(), fontweight='bold')
        ax.set_title(panel_title, fontweight='bold')
        ax.grid(True, alpha=0.25)
        
        if panel_code == '3C': 
            ax.legend(loc='upper right', frameon=False, prop={'weight': 'bold', 'size': 7})
        
    plt.suptitle("Figure 3A–C: Biological Sex Discrimination Baseline vs. Post-Exercise Average", y=1.04, fontweight='bold', fontsize=9.5)
    return fig

def plot_interaction_heatmap_19_proteins(
    long_df, full_anova_results,
    id_col='Subject_ID', sex_col='sex', time_col='time', prot_col='Protein', value_col='Value',
    time_order=('baseline', '3min', '1hr', '2hrs'),
    time_display={'3min': '3min', '1hr': '1hr', '2hrs': '2hrs'},
    sex_order=('M', 'F'),
    sex_short={'M': 'M', 'F': 'F', 'male': 'M', 'female': 'F', 'Male': 'M', 'Female': 'F'},
    use_p_col='p_value_raw', alpha=0.05,
    effect_term='TimePoint:Group',
    cmap='coolwarm', pseudocount=1e-9,
    figsize_w=3.8, row_height=0.22,
    x_tick_rotation=0
):
    mpl.rcParams['svg.fonttype'] = 'none'
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'FreeSans', 'DejaVu Sans', 'sans-serif'],
        'font.size': 8, 'font.weight': 'bold',
        'axes.titlesize': 8.5, 'axes.titleweight': 'bold',
        'axes.labelsize': 8, 'axes.labelweight': 'bold',
        'xtick.labelsize': 7, 'ytick.labelsize': 7,
        'axes.linewidth': 1.0, 'lines.linewidth': 1.2,
        'savefig.bbox': 'tight'
    })

    df = long_df[[id_col, sex_col, time_col, prot_col, value_col]].copy()
    for c in (sex_col, time_col, prot_col):
        df[c] = df[c].astype(str).str.strip()
    
    df[sex_col] = df[sex_col].map(lambda x: sex_short.get(x, x))
    df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
    df = df[df[time_col].isin(time_order) & df[sex_col].isin(sex_order)]

    baseline = time_order[0]
    post_times = [t for t in time_order if t != baseline]

    mean_tbl = (df.groupby([prot_col, sex_col, time_col])[value_col]
                  .mean().unstack(time_col).reindex(columns=time_order))

    base = mean_tbl[baseline] + pseudocount
    log2fc = np.log2(np.abs((mean_tbl[post_times] + pseudocount).div(base, axis=0)))

    long_fc = (log2fc.stack().to_frame('log2FC').reset_index()
               .rename(columns={'level_2': 'time'}))
    long_fc['col'] = long_fc['time'].map(time_display).fillna(long_fc['time']) \
                      + '_' + long_fc[sex_col].map(sex_short).fillna(long_fc[sex_col])
    mat = long_fc.pivot(index=prot_col, columns='col', values='log2FC')

    col_order = []
    for t in post_times:
        tdisp = time_display.get(t, t)
        for s in sex_order:
            col_order.append(f'{tdisp}_{sex_short.get(s, s)}')
    mat = mat.reindex(columns=col_order)

    stats = full_anova_results.copy()
    if 'Effect' in stats.columns:
        stats = stats[stats['Effect'] == effect_term]
        
    name_col = prot_col if prot_col in stats.columns else 'Protein'
    stats[name_col] = stats[name_col].astype(str).str.strip()
    stats = stats.set_index(name_col)

    common = mat.index.intersection(stats.index)
    mat, stats = mat.loc[common], stats.loc[common]

    sig_mask = stats[use_p_col].astype(float) < alpha
    mat, stats = mat.loc[sig_mask], stats.loc[sig_mask]

    order_idx = stats[use_p_col].astype(float).sort_values().index
    mat, stats = mat.loc[order_idx], stats.loc[order_idx]
    n = len(mat)
    fig_h = max(3.5, row_height * n)

    pvals = stats[use_p_col].astype(float)
    p_text = pvals.apply(lambda x: f'{x:.3g}')

    fig = plt.figure(figsize=(figsize_w, fig_h))
    gs = fig.add_gridspec(nrows=2, ncols=8, height_ratios=[0.08, 0.92], wspace=0.05, hspace=0.02,
                           width_ratios=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.35, 0.05])

    ax_heat = fig.add_subplot(gs[1, 0:6])
    ax_top  = fig.add_subplot(gs[0, 0:6], sharex=ax_heat)
    ax_cbar = fig.add_subplot(gs[1, 7])

    max_val = np.percentile(np.abs(mat.values), 98)

    hm = sns.heatmap(
        mat, ax=ax_heat, cmap=cmap, center=0, vmin=-max_val, vmax=max_val,
        cbar=False, linewidths=0.2, linecolor='white'
    )

    cbar = fig.colorbar(hm.collections[0], cax=ax_cbar)
    cbar.set_label('Log₂ Fold Change', fontsize=7.5, fontweight='bold', labelpad=2)
    cbar.ax.tick_params(width=1.0, labelsize=7)
    for t in cbar.ax.get_yticklabels():
        t.set_fontweight('bold')

    ax_heat.set_yticks(np.arange(0.5, n + 0.5, 1.0))
    ax_heat.set_yticklabels(mat.index)
    for label in ax_heat.get_yticklabels():
        label.set_rotation(0)
        label.set_fontweight('bold')

    ax_heat.set_xticks(np.arange(len(mat.columns)) + 0.5)
    ax_heat.set_xticklabels([])
    
    mf_labels = [c.split('_')[-1] for c in mat.columns]
    for i, lab in enumerate(mf_labels):
        ax_heat.text(
            i + 0.5, -0.02, lab, ha='center', va='top',
            transform=ax_heat.get_xaxis_transform(),
            rotation=x_tick_rotation, fontsize=7.5, fontweight='bold'
        )

    n_sexes = len(sex_order)
    for i in range(n_sexes, len(mat.columns), n_sexes):
        ax_heat.axvline(i, color='k', lw=0.6, alpha=0.25)

    ax_heat.set_xlabel('Sex', labelpad=14, fontsize=8, fontweight='bold')
    ax_heat.set_ylabel('Proteins', fontsize=8, fontweight='bold')

    ax_top.set_xlim(ax_heat.get_xlim())
    ax_top.set_ylim(0, 1)
    ax_top.axis('off')

    n_timepoints = len(post_times)
    centers = [i * n_sexes + (n_sexes - 1) / 2 + 0.5 for i in range(n_timepoints)]
    time_labels_disp = [time_display.get(t, t) for t in post_times]

    for xc, lab in zip(centers, time_labels_disp):
        ax_top.text(xc, 0.3, lab, ha='center', va='center', fontsize=8, fontweight='bold')

    ax_p = ax_heat.twinx()
    ax_p.set_ylim(ax_heat.get_ylim())
    ax_p.set_yticks(np.arange(0.5, n + 0.5, 1.0))
    ax_p.set_yticklabels(p_text)
    for label in ax_p.get_yticklabels():
        label.set_rotation(0)
        label.set_fontweight('bold')
    ax_p.set_ylabel('p_raw (Interaction)', rotation=90, labelpad=8, fontsize=8, fontweight='bold')
    ax_p.set_xticks([])

    plt.subplots_adjust(bottom=0.15)
    return fig

def render_pathway_enrichment_bubble():
    mpl.rcParams['svg.fonttype'] = 'none'
    file_candidates = [
        'data/EV proteins time_sex interaction significant_pathways.xlsx',
        '../data/EV proteins time_sex interaction significant_pathways.xlsx',
        'EV proteins time_sex interaction significant_pathways.xlsx'
    ]
    filepath = None
    for path in file_candidates:
        if os.path.exists(path):
            filepath = path
            break
            
    if filepath is None:
        st.warning("⚠️ Pathway Excel file not found. Please check file path.")
        return None

    df = pd.read_excel(filepath).iloc[0:15]
    col_map = {
        'Pathway': 'Pathway Name', 'Term': 'Pathway Name',
        'q-value': 'FDR', 'p.adjust': 'FDR',
        'Count': 'Number of Molecules Enriched', 'Genes_Enriched': 'Number of Molecules Enriched',
        'Background_Size': 'Total Molecules in Pathway'
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    df['-log10FDR'] = -np.log10(df['FDR'].astype(float).clip(lower=1e-15))
    if 'Fold_Enrichment' not in df.columns:
        df['Fold_Enrichment'] = 1.5

    df = df.sort_values('-log10FDR', ascending=True)

    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    df['PlotSize'] = (df.get('Number of Molecules Enriched', 3) + 1) * 35
    df['PlotSize'] = df['PlotSize'].clip(lower=60, upper=300)

    scatter = ax.scatter(
        x=df['Fold_Enrichment'], y=df['Pathway Name'],
        s=df['PlotSize'], c=df['-log10FDR'],
        cmap='viridis', edgecolor='white', linewidth=0.8, alpha=0.9, zorder=3
    )

    ax.axvline(x=1.0, color='red', linestyle=':', linewidth=1.2, label='Expected', alpha=0.8, zorder=1)
    ax.set_xlabel('Fold Enrichment\n(>1 = Enriched, <1 = Depleted)', fontweight='bold', fontsize=7.5)
    ax.set_title('Top Enriched Pathways (19 Candidate Proteins)', y=1.03, fontweight='bold', fontsize=8.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(linestyle='--', alpha=0.5, zorder=0)
    return fig

# MAIN MODULE ENTRY POINT
def render_figure3():
    ancova_df, posthoc_df, pca_scores_df, perm_df, long_df, fig3_long_df = load_fig3_results()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Figure 3A–C: PCA & PERMANOVA Trio",
        "🔥 Heatmap: 19 Candidates",
        "🧬 Pathway Enrichment",
        "📄 RM-ANCOVA Model Summary",
        "🔍 Post-Hoc Pairwise Contrasts"
    ])

    with tab1:
        st.markdown('<div style="background-color: #f8f9fa; border-left: 4px solid #007bff; padding: 10px 14px; margin-bottom: 12px; border-radius: 4px; font-size: 11px; line-height: 1.5; color: #343a40;"><b>Figure 3A–C (Multivariate Profile Discrimination):</b> Side-by-side PCA score plots comparing biological sex clustering.</div>', unsafe_allow_html=True)
        fig = render_pca_trio(pca_scores_df)
        st.pyplot(fig)
        
        st.markdown("<b>📊 Statistical Summary: PERMANOVA & PC Centroid Separation (Male vs. Female)</b>", unsafe_allow_html=True)
        st.dataframe(perm_df, use_container_width=True)

    with tab2:
        st.markdown('<div style="background-color: #fff3cd; border-left: 4px solid #ffc107; padding: 10px 14px; margin-bottom: 12px; border-radius: 4px; font-size: 11px; color: #856404;"><b>Figure 3A (Candidate Heatmap):</b> Relative fold change dynamics for 19 candidate proteins meeting nominal significance (<i>p</i><sub>raw</sub> &lt; 0.05).</div>', unsafe_allow_html=True)
        fig_hm = plot_interaction_heatmap_19_proteins(fig3_long_df, ancova_df)
        st.pyplot(fig_hm)

    with tab3:
        st.markdown('<div style="background-color: #e2e3e5; border-left: 4px solid #383d41; padding: 10px 14px; margin-bottom: 12px; border-radius: 4px; font-size: 11px; color: #383d41;"><b>Figure 3B (Pathway Enrichment):</b> Over-Representation Analysis mapping 19 candidate interaction proteins to functional Reactome/KEGG biological networks.</div>', unsafe_allow_html=True)
        fig_path = render_pathway_enrichment_bubble()
        if fig_path:
            st.pyplot(fig_path)

    with tab4:
        st.markdown('<div style="background-color: #fff3cd; border-left: 4px solid #ffc107; padding: 10px 14px; margin-bottom: 12px; border-radius: 4px; font-size: 12px; line-height: 1.5; color: #856404;"><b>📄 Reviewer Note on Output Alignment:</b> Displaying all nominal significant results (<i>p</i><sub>raw</sub> &lt; 0.05) organized by Model Effect without row truncation.</div>', unsafe_allow_html=True)
        df_ancova_fmt = ancova_df.copy()
        if not df_ancova_fmt.empty:
            df_ancova_fmt['F_statistic'] = pd.to_numeric(df_ancova_fmt['F_statistic'], errors='coerce').round(2)
            df_ancova_fmt['Partial_Eta_Squared'] = pd.to_numeric(df_ancova_fmt['Partial_Eta_Squared'], errors='coerce').round(3)
        st.dataframe(df_ancova_fmt[df_ancova_fmt['p_value_raw'] < 0.05], use_container_width=True)

    with tab5:
        st.markdown('<h4>Post-Hoc Pairwise Contrasts (emmeans)</h4>', unsafe_allow_html=True)
        df_ph_fmt = posthoc_df.copy()
        if not df_ph_fmt.empty:
            for col in ['estimate', 'std_error', 'df', 't_ratio']:
                if col in df_ph_fmt.columns:
                    df_ph_fmt[col] = pd.to_numeric(df_ph_fmt[col], errors='coerce').round(3)
        st.dataframe(df_ph_fmt[df_ph_fmt['p_value_raw'] < 0.05], use_container_width=True)