import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import ScalarFormatter
import seaborn as sns
import streamlit as st
import warnings
warnings.filterwarnings("ignore")

# Performance tweaks for fast Matplotlib rendering
plt.rcParams['path.simplify'] = True
plt.rcParams['path.simplify_threshold'] = 1.0
plt.rcParams['agg.path.chunksize'] = 10000

# --- 1. DIRECT PRE-COMPUTED DATA LOADER & REFERENCE EMM DATA ---
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

    # Normalize column names across outputs
    if 'Protein' in ancova_df.columns and 'Cytokine' not in ancova_df.columns:
        ancova_df['Cytokine'] = ancova_df['Protein']
    if 'Protein' in posthoc_df.columns and 'Cytokine' not in posthoc_df.columns:
        posthoc_df['Cytokine'] = posthoc_df['Protein']
        
    # Normalize posthoc column names
    if 'sex' in posthoc_df.columns and 'Group' not in posthoc_df.columns:
        posthoc_df['Group'] = posthoc_df['sex']
    elif 'Group' not in posthoc_df.columns:
        posthoc_df['Group'] = np.nan  # Prevents KeyError if missing entirely

    # Identify metadata vs cytokine columns
    metadata_cols = ['Subject_ID', 'sex', 'time', 'ID', 'Sex', 'Time', 'Group', 'TimePoint', 'BaselineValue']
    meta_in_df = [c for c in metadata_cols if c in wide_df.columns]
    cyto_cols = [c for c in wide_df.columns if c not in meta_in_df and pd.api.types.is_numeric_dtype(wide_df[c])]

    # Create dedicated long-format dataframe for heatmap function
    fig4_long_df = wide_df.melt(
        id_vars=meta_in_df,
        value_vars=cyto_cols,
        var_name='Protein',
        value_name='Value'
    )
    fig4_long_df['Cytokine'] = fig4_long_df['Protein']

    # EXACT USER REFERENCE EMM DATAFRAME
    df_emm_ref = pd.DataFrame({
        'Metabolite': ['Tie-2', 'Tie-2', 'CRP', 'CRP', 'Eotaxin', 'Eotaxin', 'VCAM-1', 'VCAM-1'],
        'Group': ['Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female'],
        'EMM': [726.033715, 38.851120, 28717.552925, -481866.921617, 10.642651, -6.241632, 88997.908635, 20059.582488],
        'CI_lower': [297.199891, -289.982705, -921089.0, -1431673.0, -3.336803, -20.221086, 23053.451969, -45884.874178],
        'CI_upper': [1154.867540, 567.684944, 978524.063223, 467939.588681, 24.622104, 7.737822, 154942.365301, 86004.039154],
        'SE': [211.833166, 211.833166, 469180.620513, 469180.620513, 6.9055, 6.9055, 32574.909482, 32574.909482]
    })

    return ancova_df, posthoc_df, df_emm_ref, fig4_long_df

# Alias to satisfy app.py import requirements cleanly
load_fig4_results = load_fig4_results

# --- 2. HEATMAP WITH RIGHT Y-AXIS P-VALUES & CUSTOM EMM WHISKER PLOTS ---

def plot_cytokine_heatmap(
    long_df, full_anova_results,
    id_col='Subject_ID', sex_col='sex', time_col='time', prot_col='Protein', value_col='Value',
    time_order=('baseline', '3min', '1hr', '2hrs'),
    time_display={'3min': '3min', '1hr': '1hr', '2hrs': '2hrs'},
    sex_order=('M', 'F'),
    sex_short={'M': 'M', 'F': 'F', 'male': 'M', 'female': 'F', 'Male': 'M', 'Female': 'F'},
    cmap='coolwarm', pseudocount=1e-9,
    figsize_w=4.2, row_height=0.45
):
    mpl.rcParams['svg.fonttype'] = 'none'
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'FreeSans', 'DejaVu Sans', 'sans-serif'],
        'font.size': 8, 'font.weight': 'bold',
        'axes.titlesize': 8.5, 'axes.titleweight': 'bold',
        'axes.labelsize': 8, 'axes.labelweight': 'bold',
        'xtick.labelsize': 7, 'ytick.labelsize': 7,
        'axes.linewidth': 1.0, 'lines.linewidth': 1.2
    })

    group_stats = full_anova_results[full_anova_results['Effect'].astype(str).str.lower() == 'group'].copy()
    if not group_stats.empty and 'p_value_raw' in group_stats.columns:
        group_stats['p_value_raw'] = pd.to_numeric(group_stats['p_value_raw'], errors='coerce')
        group_stats = group_stats.sort_values('p_value_raw')
        top4_cytokines = group_stats['Cytokine'].head(4).tolist()
    else:
        top4_cytokines = ['Tie-2', 'CRP', 'Eotaxin', 'VCAM-1']

    df = long_df[[id_col, sex_col, time_col, prot_col, value_col]].copy()
    df = df[df[prot_col].isin(top4_cytokines)]

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
    mat = mat.reindex([c for c in top4_cytokines if c in mat.index])

    n = len(mat)
    fig_h = max(2.2, row_height * n)

    fig = plt.figure(figsize=(figsize_w, fig_h))
    gs = fig.add_gridspec(nrows=2, ncols=8, height_ratios=[0.12, 0.88], wspace=0.05, hspace=0.02,
                           width_ratios=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.35, 0.05])

    ax_heat = fig.add_subplot(gs[1, 0:6])
    ax_top  = fig.add_subplot(gs[0, 0:6], sharex=ax_heat)
    ax_cbar = fig.add_subplot(gs[1, 7])

    max_val = np.percentile(np.abs(mat.values), 98) if not mat.empty else 1.0

    hm = sns.heatmap(
        mat, ax=ax_heat, cmap=cmap, center=0, vmin=-max_val, vmax=max_val,
        cbar=False, linewidths=0.5, linecolor='white'
    )

    cbar = fig.colorbar(hm.collections[0], cax=ax_cbar)
    cbar.set_label('Log₂ Fold Change', fontsize=7.5, fontweight='bold', labelpad=2)
    cbar.ax.tick_params(width=1.0, labelsize=7)
    for t in cbar.ax.get_yticklabels():
        t.set_fontweight('bold')

    ax_heat.set_yticks(np.arange(0.5, n + 0.5, 1.0))
    ax_heat.set_yticklabels(mat.index, fontweight='bold', rotation=0)

    p_text_dict = {'Tie-2': 'p=0.017', 'VCAM-1': 'p=0.017', 'Eotaxin': 'p=0.023', 'CRP': 'p=0.042'}
    p_text_list = []
    for c in mat.index:
        if c in p_text_dict:
            p_text_list.append(p_text_dict[c])
        elif not group_stats.empty and c in group_stats['Cytokine'].values:
            pval = group_stats[group_stats['Cytokine'] == c]['p_value_FDR'].values[0]
            p_text_list.append(f"{pval:.3g}" if pd.notnull(pval) else "N/A")
        else:
            p_text_list.append("N/A")

    ax_p = ax_heat.twinx()
    ax_p.set_ylim(ax_heat.get_ylim())
    ax_p.set_yticks(np.arange(0.5, n + 0.5, 1.0))
    ax_p.set_yticklabels(p_text_list, fontweight='bold', rotation=0)

    ax_p.set_ylabel('p_adj value', rotation=90, labelpad=10, fontsize=8, fontweight='bold')
    ax_p.tick_params(axis='y', pad=6, labelsize=7.5)
    ax_p.set_xticks([])

    ax_heat.set_xticks(np.arange(len(mat.columns)) + 0.5)
    ax_heat.set_xticklabels([])
    
    mf_labels = [c.split('_')[-1] for c in mat.columns]
    for i, lab in enumerate(mf_labels):
        ax_heat.text(
            i + 0.5, -0.05, lab, ha='center', va='top',
            transform=ax_heat.get_xaxis_transform(),
            fontsize=7.5, fontweight='bold'
        )

    n_sexes = len(sex_order)
    for i in range(n_sexes, len(mat.columns), n_sexes):
        ax_heat.axvline(i, color='k', lw=0.6, alpha=0.25)

    ax_heat.set_xlabel('Sex / Timepoints', labelpad=14, fontsize=8, fontweight='bold')
    ax_heat.set_ylabel('Top 4 Cytokines', fontsize=8, fontweight='bold')

    ax_top.set_xlim(ax_heat.get_xlim())
    ax_top.set_ylim(0, 1)
    ax_top.axis('off')

    n_timepoints = len(post_times)
    centers = [i * n_sexes + (n_sexes - 1) / 2 + 0.5 for i in range(n_timepoints)]
    time_labels_disp = [time_display.get(t, t) for t in post_times]

    for xc, lab in zip(centers, time_labels_disp):
        ax_top.text(xc, 0.3, lab, ha='center', va='center', fontsize=8, fontweight='bold')

    plt.subplots_adjust(bottom=0.2)
    return fig, mat.index.tolist()

def plot_emm_metabolite(metabolite_name, df_emm, group_styles, sig_brackets,
                        figsize=(1.3, 2.5), save_path=None, show=True, group_spacing=0.2, ax=None):
    mpl.rcParams['svg.fonttype'] = 'none'
    
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'FreeSans', 'DejaVu Sans', 'sans-serif'],
        'font.size': 8, 'font.weight': 'bold',
        'axes.titlesize': 8.5, 'axes.titleweight': 'bold',
        'axes.labelsize': 8, 'axes.labelweight': 'bold',
        'xtick.labelsize': 7, 'ytick.labelsize': 7,
        'axes.linewidth': 1.0, 'lines.linewidth': 1.2,
        'savefig.bbox': 'standard'
    })

    subset = df_emm[df_emm['Metabolite'].astype(str).str.lower() == str(metabolite_name).lower()]

    if subset.empty:
        if ax:
            ax.text(0.5, 0.5, f"No Data\n({metabolite_name})", ha='center', va='center')
        return None, None

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, layout='constrained')
    else:
        fig = ax.get_figure()

    group_names = list(group_styles.keys())
    num_groups = len(group_names)

    x_positions_dict = {
        name: (i - (num_groups - 1) / 2) * group_spacing
        for i, name in enumerate(group_names)
    }

    for i, (group, style) in enumerate(group_styles.items()):
        group_data = subset[subset['Group'] == group]

        if not group_data.empty:
            x_pos = x_positions_dict[group]
            emm_val = group_data['EMM'].values[0]
            ci_low = group_data['CI_lower'].values[0]
            ci_up = group_data['CI_upper'].values[0]

            ax.errorbar(
                x=x_pos,
                y=emm_val,
                yerr=[
                    [np.maximum(0, emm_val - ci_low)],
                    [np.maximum(0, ci_up - emm_val)]
                ],
                fmt=style['marker'],
                color=style['color'],
                markersize=6,
                capsize=4,
                capthick=1.2,
                elinewidth=1.2,
                markeredgecolor='black',
                markeredgewidth=1,
                zorder=3,
                label=style['label']
            )

    if metabolite_name in sig_brackets:
        for bracket in sig_brackets[metabolite_name]:
            group1, group2, y_pos, sig_text = bracket
            if group1 in group_styles and group2 in group_styles:
                g1_x = x_positions_dict[group1]
                g2_x = x_positions_dict[group2]

                y_range_calc = ax.get_ylim()[1] - ax.get_ylim()[0]
                if y_range_calc == 0:
                    y_range_calc = np.max(subset['EMM']) * 0.1 if np.max(subset['EMM']) != 0 else 1.0

                handle_height = y_range_calc * 0.03
                text_offset = y_range_calc * 0.03

                ax.plot([g1_x, g1_x, g2_x, g2_x],
                        [y_pos, y_pos + handle_height, y_pos + handle_height, y_pos],
                        'k', lw=1.2)
                ax.text((g1_x + g2_x) / 2, y_pos + text_offset, sig_text,
                        ha='center', va='bottom', fontsize=7, fontweight='bold')

    ax.set_title(f'Sex Difference\n{metabolite_name}', pad=12, fontweight='bold', fontsize=8)
    ax.set_xticks(list(x_positions_dict.values()))
    ax.set_xticklabels(group_styles.keys(), fontweight='bold')

    formatter = ScalarFormatter(useOffset=True, useMathText=True)
    formatter.set_powerlimits((-3, 3))
    ax.yaxis.set_major_formatter(formatter)

    ax.set_xlabel('Sex', fontweight='bold', fontsize=7.5)
    ax.set_ylabel('Estimated Marginal Mean\n(Normalized expression)', fontweight='bold', fontsize=7.5)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    y_min_data = subset['CI_lower'].min()
    y_max_data = subset['CI_upper'].max()

    max_bracket_y_pos = y_max_data
    if metabolite_name in sig_brackets:
        for bracket in sig_brackets[metabolite_name]:
            max_bracket_y_pos = max(max_bracket_y_pos, bracket[2])

    initial_y_range = np.maximum(1e-9, y_max_data - y_min_data)
    estimated_buffer = initial_y_range * 0.06
    effective_max_y = max(y_max_data, max_bracket_y_pos + estimated_buffer)
    total_y_range = effective_max_y - y_min_data

    ax.set_ylim(
        y_min_data - total_y_range * 0.1,
        effective_max_y + total_y_range * 0.15
    )

    min_x = min(x_positions_dict.values())
    max_x = max(x_positions_dict.values())
    ax.set_xlim(min_x - 0.11, max_x + 0.11)

    if save_path:
        plt.savefig(save_path, format='svg', dpi=600, bbox_inches='tight')

    if show and ax is None:
        plt.show()

    return fig, ax

# --- MAIN RENDER FUNCTION FOR STREAMLIT ---
def render_figure4():
    ancova_df, posthoc_df, df_emm, fig4_long_df = load_fig4_results()

    # Streamlit Selectbox replacing ipywidgets dropdown
    selected_view = st.selectbox(
        "Select Section View:",
        [
            '🔥 Heatmap & EMM Plots (Top 4 Group Cytokines)',
            '📄 RM-ANCOVA Model Summary (Main & Interaction Effects)',
            '🔍 Post-Hoc Pairwise Contrasts (emmeans)'
        ],
        key="fig4_selectbox"
    )

    st.markdown("---")

    if selected_view == '🔥 Heatmap & EMM Plots (Top 4 Group Cytokines)':
        st.markdown("""
        <div style="background-color: #fff3cd; border-left: 4px solid #ffc107; padding: 10px 14px; margin-bottom: 12px; border-radius: 4px; font-size: 11px; color: #856404;">
            <b>Figure 4A & 4B (Targeted Cytokine Profiles):</b> Heatmap displaying recovery dynamics for top 4 cytokines with right-axis unadjusted <i>p</i>-values, followed by EMM whisker plots arranged left-to-right in identical order.
        </div>
        """, unsafe_allow_html=True)

        fig_hm, ordered_heatmap_cytokines = plot_cytokine_heatmap(fig4_long_df, ancova_df)
        st.pyplot(fig_hm)

        group_styles = {
            'Male': {'color': 'black', 'marker': 'o', 'label': 'Male'},
            'Female': {'color': 'grey', 'marker': 's', 'label': 'Female'}
        }
        sig_brackets = {
            'CRP': [('Male', 'Female', 1.2e6, '*p=0.042')],
            'Eotaxin': [('Male', 'Female', 28, '*p=0.023')],
            'VCAM-1': [('Male', 'Female', 168e3, '*p=0.017')],
            'Tie-2': [('Male', 'Female', 12.5e2, '*p=0.017')]
        }

        cytokines_to_plot = ordered_heatmap_cytokines if ordered_heatmap_cytokines else ['Tie-2', 'CRP', 'Eotaxin', 'VCAM-1']

        fig, axes = plt.subplots(1, 4, figsize=(8.5, 3.2), constrained_layout=True)

        for idx, cyto in enumerate(cytokines_to_plot):
            plot_emm_metabolite(
                metabolite_name=cyto,
                df_emm=df_emm,
                group_styles=group_styles,
                sig_brackets=sig_brackets,
                group_spacing=0.2,
                show=False,
                ax=axes[idx]
            )

        plt.suptitle('Figure 4B: Estimated Marginal Means (Baseline-Adjusted Sex Separation)', y=1.04, fontweight='bold', fontsize=9.5)
        st.pyplot(fig)

    elif selected_view == '📄 RM-ANCOVA Model Summary (Main & Interaction Effects)':
        st.markdown("""
        <div style="background-color: #fff3cd; border-left: 4px solid #ffc107; padding: 10px 14px; margin-bottom: 12px; border-radius: 4px; font-size: 12px; line-height: 1.5; color: #856404;">
            <b>📄 RM-ANCOVA Model Summary:</b> Statistical testing for main effects of TimePoint, Group (Sex), and TimePoint × Group interaction across target cytokines.
        </div>
        """, unsafe_allow_html=True)

        df_ancova_fmt = ancova_df.copy()
        if not df_ancova_fmt.empty:
            df_ancova_fmt['F_statistic'] = pd.to_numeric(df_ancova_fmt['F_statistic'], errors='coerce').round(2)
            df_ancova_fmt['Partial_Eta_Squared'] = pd.to_numeric(df_ancova_fmt['Partial_Eta_Squared'], errors='coerce').round(3)
            df_ancova_fmt['p_value_raw_fmt'] = df_ancova_fmt['p_value_raw'].apply(lambda p: f"{p:.4f}" if pd.notnull(p) and p >= 0.0001 else ("< 0.0001" if pd.notnull(p) else "N/A"))
            df_ancova_fmt['p_value_FDR_fmt'] = df_ancova_fmt['p_value_FDR'].apply(lambda p: f"{p:.4f}" if pd.notnull(p) and p >= 0.0001 else ("< 0.0001" if pd.notnull(p) else "N/A"))

        effect_map = [
            ('TimePoint:Group', '1. Time × Group Interaction Effects'),
            ('Group', '2. Group / Sex Main Effects'),
            ('TimePoint', '3. Time Main Effects')
        ]

        cols_to_show = ['Cytokine', 'Effect', 'N', 'num_df', 'den_df', 'F_statistic', 'p_value_raw_fmt', 'p_value_FDR_fmt', 'Partial_Eta_Squared', 'Significant_FDR']
        cols_exist = [c for c in cols_to_show if c in df_ancova_fmt.columns]
        rename_dict = {'p_value_raw_fmt': 'p_value_raw', 'p_value_FDR_fmt': 'p_value_FDR'}

        for eff_key, eff_title in effect_map:
            sub = df_ancova_fmt[df_ancova_fmt['Effect'].astype(str).str.contains(eff_key, case=False, na=False)].sort_values('p_value_raw')
            st.markdown(f'<h4 style="margin-top:22px; margin-bottom:6px; color:#2c3e50;">{eff_title}</h4>', unsafe_allow_html=True)
            if not sub.empty:
                st.dataframe(sub[cols_exist].rename(columns=rename_dict), use_container_width=True)
            else:
                st.markdown('<p style="font-size:11px; color:#7f8c8d; font-style:italic;">No statistics recorded for this term.</p>', unsafe_allow_html=True)

        ancova_note = """
        <div style="background-color: #f8f9fa; border-left: 4px solid #007bff; padding: 14px; margin-top: 25px; border-radius: 4px; font-size: 12px; line-height: 1.6; color: #212529;">
            <b>📊 Notes on Cytokine ANCOVA Terms & Column Layout:</b><br>
            • <b>Side-by-Side Statistics:</b> <code>p_value_raw</code> (uncorrected ANOVA <i>p</i>) and <code>p_value_FDR</code> (Benjamini-Hochberg adjusted).<br>
            • <b>Partial_Eta_Squared (η<sub>p</sub>²):</b> Effect size estimate (Small ≈ 0.01, Medium ≈ 0.06, Large ≥ 0.14).
        </div>
        """
        st.markdown(ancova_note, unsafe_allow_html=True)

    elif selected_view == '🔍 Post-Hoc Pairwise Contrasts (emmeans)':
        df_ph_fmt = posthoc_df.copy()
        if not df_ph_fmt.empty:
            for col in ['estimate', 'std_error', 'df', 't_ratio']:
                if col in df_ph_fmt.columns:
                    df_ph_fmt[col] = pd.to_numeric(df_ph_fmt[col], errors='coerce').round(3)
            
            df_ph_fmt['p_value_raw_fmt'] = df_ph_fmt['p_value_raw'].apply(lambda p: f"{p:.4f}" if pd.notnull(p) and p >= 0.0001 else ("< 0.0001" if pd.notnull(p) else "N/A"))
            if 'p_value_FDR' in df_ph_fmt.columns:
                df_ph_fmt['p_value_FDR_fmt'] = df_ph_fmt['p_value_FDR'].apply(lambda p: f"{p:.4f}" if pd.notnull(p) and p >= 0.0001 else ("< 0.0001" if pd.notnull(p) else "N/A"))

        ph_cols = ['Cytokine', 'contrast', 'TimePoint', 'Group', 'estimate', 'std_error', 'df', 't_ratio', 'p_value_raw_fmt', 'p_value_FDR_fmt']
        existing_cols = [c for c in ph_cols if c in df_ph_fmt.columns]
        ph_rename = {'p_value_raw_fmt': 'p_value_raw', 'p_value_FDR_fmt': 'p_value_FDR'}

        contrast_str = df_ph_fmt['contrast'].astype(str) if 'contrast' in df_ph_fmt.columns else pd.Series('', index=df_ph_fmt.index)
        
        if 'Contrast_Type' in df_ph_fmt.columns:
            is_between = df_ph_fmt['Contrast_Type'] == 'Between-Group'
        else:
            is_between = contrast_str.str.contains('Male - Female|Female - Male|M - F|F - M|Male vs Female', case=False, na=False)
        
        between_sub = df_ph_fmt[is_between].sort_values('p_value_raw') if 'p_value_raw' in df_ph_fmt.columns else df_ph_fmt[is_between]
        within_sub = df_ph_fmt[~is_between].sort_values('p_value_raw') if 'p_value_raw' in df_ph_fmt.columns else df_ph_fmt[~is_between]

        st.markdown('<h4 style="margin-top:15px; margin-bottom:6px; color:#2c3e50;">1. Between-Group Pairwise Contrasts (Male vs. Female)</h4>', unsafe_allow_html=True)
        if not between_sub.empty:
            st.dataframe(between_sub[existing_cols].rename(columns=ph_rename), use_container_width=True)
        else:
            st.markdown('<p style="font-size:11px; color:#7f8c8d; font-style:italic;">No between-group contrasts found.</p>', unsafe_allow_html=True)

        st.markdown('<h4 style="margin-top:25px; margin-bottom:6px; color:#2c3e50;">2. Within-Group Pairwise Contrasts (Recovery Time Shifting)</h4>', unsafe_allow_html=True)
        if not within_sub.empty:
            st.dataframe(within_sub[existing_cols].rename(columns=ph_rename), use_container_width=True)
        else:
            st.markdown('<p style="font-size:11px; color:#7f8c8d; font-style:italic;">No within-group contrasts recorded in output table.</p>', unsafe_allow_html=True)

        posthoc_note = """
        <div style="background-color: #f8f9fa; border-left: 4px solid #28a745; padding: 14px; margin-top: 25px; border-radius: 4px; font-size: 12px; line-height: 1.6; color: #212529;">
            <b>📊 Notes on Cytokine Pairwise Contrasts:</b><br>
            • <b>estimate:</b> Difference in Baseline-Adjusted Estimated Marginal Means (EMMs).<br>
            • <b>t_ratio & std_error:</b> Test statistic and standard error for the specified contrast.
        </div>
        """
        st.markdown(posthoc_note, unsafe_allow_html=True)
