import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import streamlit as st
import warnings
warnings.filterwarnings("ignore")

# --- 1. FAST PRE-COMPUTED DATA LOADER ---
@st.cache_data
def load_fig1_results():
    if os.path.exists('results/fig1_skewness_stats.csv'):
        data_dir = 'results'
    elif os.path.exists('../results/fig1_skewness_stats.csv'):
        data_dir = '../results'
    else:
        data_dir = '.'
        
    stats_df = pd.read_csv(os.path.join(data_dir, 'fig1_skewness_stats.csv'))
    df_all = pd.read_csv(os.path.join(data_dir, 'fig1_processed_data.csv'))
    return stats_df, df_all

# Alias to satisfy app.py import requirements cleanly
load_fig1_results = load_fig1_results

# --- 2. MAIN RENDER FUNCTION FOR STREAMLIT ---
def render_figure1():
    stats_df, df_all = load_fig1_results()

    # Streamlit Selectbox replacing ipywidgets dropdown
    selected_time = st.selectbox(
        "Select Time View:",
        ['All Time Points (2x2 Grid)', 'Baseline', '3 min', '1 hour', '2 hours'],
        key="fig1_selectbox"
    )

    st.markdown("---")

    mpl.rcParams['svg.fonttype'] = 'none'
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'FreeSans', 'DejaVu Sans', 'sans-serif'],
        'font.size': 8, 'font.weight': 'bold',
        'axes.titlesize': 8.5, 'axes.titleweight': 'bold',
        'axes.labelsize': 8, 'axes.labelweight': 'bold',
        'xtick.labelsize': 7, 'ytick.labelsize': 7,
        'axes.linewidth': 1.0, 'lines.linewidth': 1.2
    })
    
    sex_markers = {'Male': 'o', 'Female': 's'}
    
    if selected_time == 'All Time Points (2x2 Grid)':
        fig, axes = plt.subplots(2, 2, figsize=(6, 6), constrained_layout=True)
        time_points = ['Baseline', '3 min', '1 hour', '2 hours']
        
        for ax, t_pt in zip(axes.flat, time_points):
            subset = df_all[df_all['time'] == t_pt]
            
            sns.scatterplot(
                data=subset, x='Median Value (nm)', y='Mean Value (nm)',
                hue='sex', style='sex', ax=ax, s=35, 
                palette={'Male': '0.1', 'Female': '0.6'}, markers=sex_markers,
                edgecolor='black', linewidth=0.5
            )
            
            all_vals = pd.concat([subset['Mean Value (nm)'], subset['Median Value (nm)']])
            lims = [all_vals.min() * 0.9, all_vals.max() * 1.1]
            ax.plot(lims, lims, 'k--', alpha=0.4, linewidth=1)
            ax.set_xlim(lims); ax.set_ylim(lims)
            
            p_col = next((col for col in ['p_value_raw', 'p_value_FDR', 'p-value', 'p_value_wilcoxon'] if col in stats_df.columns), None)

            if p_col:
                m_sub = stats_df[(stats_df['Time Point'] == t_pt) & (stats_df['Sex'] == 'Male')]
                f_sub = stats_df[(stats_df['Time Point'] == t_pt) & (stats_df['Sex'] == 'Female')]

                m_p = m_sub[p_col].values[0] if not m_sub.empty else np.nan
                f_p = f_sub[p_col].values[0] if not f_sub.empty else np.nan

                m_str = f"{m_p:.2g}" if (not np.isnan(m_p) and m_p >= 0.001) else ("< 0.001" if not np.isnan(m_p) else "N/A")
                f_str = f"{f_p:.2g}" if (not np.isnan(f_p) and f_p >= 0.001) else ("< 0.001" if not np.isnan(f_p) else "N/A")

                ax.text(0.05, 0.95, f"Male p: {m_str}\nFemale p: {f_str}", 
                        transform=ax.transAxes, ha='left', va='top', fontsize=8, fontweight='bold',
                        bbox=dict(facecolor='white', alpha=0.85, edgecolor='none', pad=2))
            
            ax.set_title(f'Time: {t_pt}')
            ax.set_xlabel('Median Size (nm)')
            ax.set_ylabel('Mean Size (nm)')
            
            if t_pt == '2 hours':
                ax.legend(prop={'weight':'bold', 'size':7}, loc='lower right', frameon=True)
            else:
                if ax.get_legend(): ax.get_legend().remove()
                
        plt.suptitle('EV Size Skewness (Mean vs Median)', fontweight='bold')
        st.pyplot(fig)
        
        st.markdown("<b>Statistical Summary (Wilcoxon Signed-Ranked Test):</b>", unsafe_allow_html=True)
        st.dataframe(stats_df, use_container_width=True)
        
    else:
        fig, ax = plt.subplots(figsize=(4.5, 4), constrained_layout=True)
        subset = df_all[df_all['time'] == selected_time]
        
        sns.scatterplot(
            data=subset, x='Median Value (nm)', y='Mean Value (nm)',
            hue='sex', style='sex', ax=ax, s=45, 
            palette={'Male': '0.1', 'Female': '0.6'}, markers=sex_markers,
            edgecolor='black', linewidth=0.5
        )
        
        all_vals = pd.concat([subset['Mean Value (nm)'], subset['Median Value (nm)']])
        lims = [all_vals.min() * 0.9, all_vals.max() * 1.1]
        ax.plot(lims, lims, 'k--', alpha=0.4, linewidth=1)
        ax.set_xlim(lims); ax.set_ylim(lims)
        
        p_col = next((col for col in ['p_value_raw', 'p_value_FDR', 'p-value', 'p_value_wilcoxon'] if col in stats_df.columns), None)
        
        if p_col:
            m_sub = stats_df[(stats_df['Time Point'] == selected_time) & (stats_df['Sex'] == 'Male')]
            f_sub = stats_df[(stats_df['Time Point'] == selected_time) & (stats_df['Sex'] == 'Female')]

            m_p = m_sub[p_col].values[0] if not m_sub.empty else np.nan
            f_p = f_sub[p_col].values[0] if not f_sub.empty else np.nan

            m_str = f"{m_p:.2g}" if (not np.isnan(m_p) and m_p >= 0.001) else ("< 0.001" if not np.isnan(m_p) else "N/A")
            f_str = f"{f_p:.2g}" if (not np.isnan(f_p) and f_p >= 0.001) else ("< 0.001" if not np.isnan(f_p) else "N/A")

            ax.text(0.05, 0.95, f"Male p: {m_str}\nFemale p: {f_str}", 
                    transform=ax.transAxes, ha='left', va='top', fontsize=8, fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.85, edgecolor='none', pad=2))
        
        ax.set_title(f'Time: {selected_time}')
        ax.set_xlabel('Median Size (nm)')
        ax.set_ylabel('Mean Size (nm)')
        ax.legend(prop={'weight':'bold', 'size':8}, loc='lower right', frameon=True)
        st.pyplot(fig)
        
        st.markdown(f"<b>Statistical Summary (Wilcoxon Signed-Ranked Test) ({selected_time}):</b>", unsafe_allow_html=True)
        st.dataframe(stats_df[stats_df['Time Point'] == selected_time], use_container_width=True)
