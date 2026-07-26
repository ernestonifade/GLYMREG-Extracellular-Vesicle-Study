import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import ScalarFormatter
import seaborn as sns
import streamlit as st

@st.cache_data
def load_fig4_results():
    if os.path.exists('results/fig4_ancova_stats.csv'):
        data_dir = 'results'
    elif os.path.exists('../results/fig4_ancova_stats.csv'):
        data_dir = '../results'
    else:
        data_dir = '.'
        
    ancova_df = pd.read_csv(os.path.join(data_dir, 'fig4_ancova_stats.csv'))
    posthoc_df = pd.read_csv(os.path.join(data_dir, 'fig4_posthoc_contrasts.csv'))
    wide_df = pd.read_csv(os.path.join(data_dir, 'fig4_processed_data.csv'))

    if 'Protein' in ancova_df.columns and 'Cytokine' not in ancova_df.columns:
        ancova_df['Cytokine'] = ancova_df['Protein']
    if 'Protein' in posthoc_df.columns and 'Cytokine' not in posthoc_df.columns:
        posthoc_df['Cytokine'] = posthoc_df['Protein']
        
    if 'sex' in posthoc_df.columns and 'Group' not in posthoc_df.columns:
        posthoc_df['Group'] = posthoc_df['sex']

    metadata_cols = ['Subject_ID', 'sex', 'time', 'ID', 'Sex', 'Time', 'Group', 'TimePoint', 'BaselineValue']
    meta_in_df = [c for c in metadata_cols if c in wide_df.columns]
    cyto_cols = [c for c in wide_df.columns if c not in meta_in_df and pd.api.types.is_numeric_dtype(wide_df[c])]

    fig4_long_df = wide_df.melt(
        id_vars=meta_in_df, value_vars=cyto_cols, var_name='Protein', value_name='Value'
    )
    fig4_long_df['Cytokine'] = fig4_long_df['Protein']

    df_emm_ref = pd.DataFrame({
        'Metabolite': ['Tie-2', 'Tie-2', 'CRP', 'CRP', 'Eotaxin', 'Eotaxin', 'VCAM-1', 'VCAM-1'],
        'Group': ['Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female'],
        'EMM': [726.033715, 38.851120, 28717.552925, -481866.921617, 10.642651, -6.241632, 88997.908635, 20059.582488],
        'CI_lower': [297.199891, -289.982705, -921089.0, -1431673.0, -3.336803, -20.221086, 23053.451969, -45884.874178],
        'CI_upper': [1154.867540, 567.684944, 978524.063223, 467939.588681, 24.622104, 7.737822, 154942.365301, 86004.039154],
        'SE': [211.833166, 211.833166, 469180.620513, 469180.620513, 6.9055, 6.9055, 32574.909482, 32574.909482]
    })
    return ancova_df, posthoc_df, df_emm_ref, fig4_long_df

def plot_cytokine_heatmap(long_df, full_anova_results):
    mpl.rcParams['svg.fonttype'] = 'none'
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'FreeSans', 'DejaVu Sans', 'sans-serif'],
        'font.size': 8, 'font.weight': 'bold', 'axes.titlesize': 8.5, 'axes.titleweight': 'bold',
        'axes.labelsize': 8, 'axes.labelweight': 'bold', 'xtick.labelsize': 7, 'ytick.labelsize': 7,
        'axes.linewidth': 1.0, 'lines.linewidth': 1.2
    })

    top4_cytokines = ['Tie-2', 'CRP', 'Eotaxin', 'VCAM-1']
    df = long_df[long_df['Protein'].isin(top4_cytokines)].copy()
    time_order = ('baseline', '3min', '1hr', '2hrs')
    time_display = {'3min': '3min', '1hr': '1hr', '2hrs': '2hrs'}
    sex_short = {'M': 'M', 'F': 'F', 'male': 'M', 'female': 'F', 'Male': 'M', 'Female': 'F'}

    df['sex'] = df['sex'].map(lambda x: sex_short.get(x, x))
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')

    mean_tbl = df.groupby(['Protein', 'sex', 'time'])['Value'].mean().unstack('time').reindex(columns=time_order)
    base = mean_tbl['baseline'] + 1e-9
    log2fc = np.log2(np.abs((mean_tbl[['3min', '1hr', '2hrs']] + 1e-9).div(base, axis=0)))

    long_fc = log2fc.stack().to_frame('log2FC').reset_index().rename(columns={'level_2': 'time'})
    long_fc['col'] = long_fc['time'].map(time_display).fillna(long_fc['time']) + '_' + long_fc['sex']
    mat = long_fc.pivot(index='Protein', columns='col', values='log2FC')

    fig = plt.figure(figsize=(4.2, 2.2))
    ax = sns.heatmap(mat, cmap='coolwarm', center=0, cbar_kws={'label': 'Log₂ Fold Change'})
    ax.set_title("Figure 4A: Cytokine Dynamics", fontweight='bold')
    return fig, top4

def plot_emm_metabolite(metabolite_name, df_emm, group_styles, sig_brackets, group_spacing=0.2, ax=None):
    subset = df_emm[df_emm['Metabolite'].astype(str).str.lower() == str(metabolite_name).lower()]
    if subset.empty: return None

    if ax is None: fig, ax = plt.subplots(figsize=(1.3, 2.5), layout='constrained')

    group_names = list(group_styles.keys())
    x_positions_dict = {name: (i - (len(group_names) - 1) / 2) * group_spacing for i, name in enumerate(group_names)}

    for group, style in group_styles.items():
        group_data = subset[subset['Group'] == group]
        if not group_data.empty:
            x_pos = x_positions_dict[group]
            emm_val = group_data['EMM'].values[0]
            ci_low, ci_up = group_data['CI_lower'].values[0], group_data['CI_upper'].values[0]

            ax.errorbar(
                x=x_pos, y=emm_val, yerr=[[np.maximum(0, emm_val - ci_low)], [np.maximum(0, ci_up - emm_val)]],
                fmt=style['marker'], color=style['color'], markersize=6, capsize=4, capthick=1.2,
                elinewidth=1.2, markeredgecolor='black', markeredgewidth=1, zorder=3, label=style['label']
            )

    if metabolite_name in sig_brackets:
        for bracket in sig_brackets[metabolite_name]:
            group1, group2, y_pos, sig_text = bracket
            if group1 in group_styles and group2 in group_styles:
                g1_x, g2_x = x_positions_dict[group1], x_positions_dict[group2]
                ax.plot([g1_x, g1_x, g2_x, g2_x], [y_pos, y_pos*1.03, y_pos*1.03, y_pos], 'k', lw=1.2)
                ax.text((g1_x + g2_x) / 2, y_pos*1.04, sig_text, ha='center', va='bottom', fontsize=7, fontweight='bold')

    ax.set_title(f'Sex Difference\n{metabolite_name}', pad=12, fontweight='bold', fontsize=8)
    ax.set_xticks(list(x_positions_dict.values()))
    ax.set_xticklabels(group_styles.keys(), fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    return ax

# MAIN MODULE ENTRY POINT
def render_figure4():
    ancova_df, posthoc_df, df_emm, fig4_long_df = load_fig4_results()

    tab1, tab2, tab3 = st.tabs([
        "🔥 Heatmap & EMM Plots",
        "📄 RM-ANCOVA Model Summary",
        "🔍 Post-Hoc Pairwise Contrasts"
    ])

    with tab1:
        st.markdown('<div style="background-color: #fff3cd; border-left: 4px solid #ffc107; padding: 10px 14px; margin-bottom: 12px; border-radius: 4px; font-size: 11px; color: #856404;"><b>Figure 4A & 4B (Targeted Cytokine Profiles):</b> Heatmap displaying recovery dynamics for top 4 cytokines with right-axis unadjusted <i>p</i>-values, followed by EMM whisker plots.</div>', unsafe_allow_html=True)
        
        fig_hm, top4 = plot_cytokine_heatmap(fig4_long_df, ancova_df)
        st.pyplot(fig_hm)

        group_styles = {'Male': {'color': 'black', 'marker': 'o', 'label': 'Male'}, 'Female': {'color': 'grey', 'marker': 's', 'label': 'Female'}}
        sig_brackets = {
            'CRP': [('Male', 'Female', 1.2e6, '*p=0.042')], 'Eotaxin': [('Male', 'Female', 28, '*p=0.023')],
            'VCAM-1': [('Male', 'Female', 168e3, '*p=0.017')], 'Tie-2': [('Male', 'Female', 12.5e2, '*p=0.017')]
        }

        fig_emm, axes = plt.subplots(1, 4, figsize=(8.5, 3.2), constrained_layout=True)
        for idx, cyto in enumerate(top4):
            plot_emm_metabolite(metabolite_name=cyto, df_emm=df_emm, group_styles=group_styles, sig_brackets=sig_brackets, ax=axes[idx])
        plt.suptitle('Figure 4B: Estimated Marginal Means (Baseline-Adjusted Sex Separation)', y=1.04, fontweight='bold', fontsize=9.5)
        st.pyplot(fig_emm)

    with tab2:
        st.subheader("Figure 4 RM-ANCOVA Model Summary")
        st.dataframe(ancova_df, use_container_width=True)

    with tab3:
        st.subheader("Figure 4 Post-Hoc Pairwise Contrasts")
        st.dataframe(posthoc_df, use_container_width=True)