import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib as mpl
import seaborn as sns
from matplotlib.lines import Line2D
import streamlit as st
import warnings
warnings.filterwarnings("ignore")

# --- 1. FAST PRE-COMPUTED DATA LOADER ---
@st.cache_data
def load_fig2_results():
    if os.path.exists('results/fig2_ancova_stats.csv'):
        data_dir = 'results'
    elif os.path.exists('../results/fig2_ancova_stats.csv'):
        data_dir = '../results'
    else:
        data_dir = '.'
        
    ancova_df = pd.read_csv(os.path.join(data_dir, 'fig2_ancova_stats.csv'))
    posthoc_df = pd.read_csv(os.path.join(data_dir, 'fig2_posthoc_contrasts.csv'))
    corr_overall = pd.read_csv(os.path.join(data_dir, 'fig2_corr_overall.csv'))
    corr_sex = pd.read_csv(os.path.join(data_dir, 'fig2_corr_sex.csv'))
    long_df = pd.read_csv(os.path.join(data_dir, 'fig2_processed_data.csv'))
    return ancova_df, posthoc_df, corr_overall, corr_sex, long_df

# Alias to satisfy app.py import requirements cleanly
load_fig2_results = load_fig2_results

# --- 2. UTILITY FUNCTIONS ---
def add_bracket(ax, x1, x2, y, h, text):
    """Draws a significance bracket between two x coordinates."""
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.2, c='black')
    ax.text((x1+x2)*0.5, y+h, text, ha='center', va='bottom', color='black', fontsize=7, fontweight='bold')

def get_stars(p):
    """Returns standard significance stars."""
    if p < 0.0001: return '****'
    elif p < 0.001: return '***'
    elif p < 0.01: return '**'
    elif p < 0.05: return '*'
    return 'ns'

# --- 3. MAIN RENDER FUNCTION FOR STREAMLIT ---
def render_figure2():
    ancova_df, posthoc_df, corr_overall, corr_sex, long_df = load_fig2_results()

    # Streamlit Selectbox replacing ipywidgets dropdown
    selected_view = st.selectbox(
        "Select Section View:",
        [
            '📄 Primary Manuscript Report (Jamovi PDF Output)', 
            'Figure 2A: EV Concentration Trajectory', 
            'Figure 2B: EV Size Trajectory', 
            'Figure 2CD: EMM Group Differences', 
            'Figure 2E: EV Conc vs Size Correlation (Overall)',
            'Figure 2E: EV Conc vs Size Correlation (By Sex)',      
            'ANCOVA Table (DF, F-stat, p-FDR, pes)',
            'Post-Hoc Pairwise Comparisons (Within & Between)'
        ],
        key="fig2_selectbox"
    )

    st.markdown("---")

    mpl.rcParams['svg.fonttype'] = 'none'
    plt.rcParams.update({
        'font.family': 'sans-serif', 
        'font.sans-serif': ['Arial', 'Helvetica', 'FreeSans', 'DejaVu Sans', 'sans-serif'],
        'font.size': 8, 'font.weight': 'bold',
        'axes.titlesize': 8.5, 'axes.titleweight': 'bold',
        'axes.labelsize': 8, 'axes.labelweight': 'bold',
        'xtick.labelsize': 7, 'ytick.labelsize': 7,
        'axes.linewidth': 1.0, 'lines.linewidth': 1.2
    })
    
    sex_colors = {'Male': '0.2', 'Female': '0.6'}
    sex_markers = {'Male': 'o', 'Female': 's'}
    time_order = ["baseline", "3min", "1hr", "2hrs"]

    # --- VIEW 1: EMBEDDED JAMOVI PDF REPORT ---
    if selected_view == '📄 Primary Manuscript Report (Jamovi PDF Output)':
        pdf_url = "https://raw.githubusercontent.com/ernestonifade/GLYMREG-Extracellular-Vesicle-Study/main/data/Jamovi_Statistical_Report_Figure2.pdf"
        pdfjs_viewer_url = f"https://mozilla.github.io/pdf.js/web/viewer.html?file={pdf_url}"

        st.markdown(f"""
        <div style="background-color: #e3f2fd; border-left: 5px solid #2196f3; padding: 12px 16px; margin-bottom: 15px; border-radius: 4px; font-size: 12px; line-height: 1.6; color: #0d47a1;">
            <b>📄 Primary Manuscript Statistical Report (Jamovi v2.3+):</b><br>
            This viewer displays the primary statistical export generated in Jamovi, which was used for reporting EV concentration/size p-values and annotated figure brackets in the manuscript. 
            <i>To explore the automated batch-processing model used for high-dimensional protein/cytokine screening, select any of the interactive views from the dropdown menu above.</i>
            <br><br>
            <a href="{pdf_url}" target="_blank" style="background-color: #2196f3; color: white; padding: 6px 12px; border-radius: 4px; text-decoration: none; font-weight: bold; display: inline-block;">
                📥 Open / Download Full Jamovi PDF in New Tab ↗
            </a>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f'<iframe src="{pdfjs_viewer_url}" width="100%" height="750px" style="border: 1px solid #cccccc; border-radius: 4px; margin-top: 5px;"></iframe>', unsafe_allow_html=True)

    # --- VIEW 2: EV CONCENTRATION TRAJECTORY ---
    elif selected_view == 'Figure 2A: EV Concentration Trajectory':
        fig, ax = plt.subplots(figsize=(4.5, 3.5), layout='constrained')
        summary = (long_df.groupby(["time", "sex"])["Concentration/ml"].agg(mean="mean", std="std", count="count").reset_index())
        summary["sem"] = summary["std"] / np.sqrt(summary["count"].clip(lower=1))
        
        for sex_val, group in summary.groupby("sex"):
            group = group.set_index("time").reindex(time_order).reset_index()
            ax.errorbar(
                np.arange(len(time_order)), group["mean"], yerr=1.96*group["sem"],
                fmt=f"{sex_markers.get(sex_val, 'o')}-", capsize=4, label=sex_val,
                color=sex_colors.get(sex_val, 'black'), linewidth=1.5, markersize=6
            )
        ax.set_xticks(np.arange(len(time_order)))
        ax.set_xticklabels(['Baseline', '3 min', '1 hour', '2 hours'])
        ax.set_xlabel("Time Point"); ax.set_ylabel("EV Concentration (/ml) (± 95% CI)")
        ax.set_title("Temporal Dynamics: EV Concentration")
        
        summary['ci_upper'] = summary['mean'] + (1.96 * summary['sem'])  
        y_max = summary['ci_upper'].max() * 1.15  
        add_bracket(ax, 0, 3, y_max, y_max*0.03, "ns (p = 0.309)")
        ax.set_ylim(0, summary['ci_upper'].max() * 1.30)  
        
        formatter = mticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True); formatter.set_powerlimits((-1, 1))
        ax.yaxis.set_major_formatter(formatter)
        ax.legend(frameon=False, prop={'weight': 'bold', 'size': 7})
        ax.grid(True, alpha=0.2, linestyle='--')
        st.pyplot(fig)

    # --- VIEW 3: EV SIZE TRAJECTORY ---
    elif selected_view == 'Figure 2B: EV Size Trajectory':
        fig, ax = plt.subplots(figsize=(4.5, 3.5), layout='constrained')
        summary = (long_df.groupby(["time", "sex"])["Median Value (nm)"].agg(mean="mean", std="std", count="count").reset_index())
        summary["sem"] = summary["std"] / np.sqrt(summary["count"].clip(lower=1))
        
        for sex_val, group in summary.groupby("sex"):
            group = group.set_index("time").reindex(time_order).reset_index()
            ax.errorbar(
                np.arange(len(time_order)), group["mean"], yerr=1.96*group["sem"],
                fmt=f"{sex_markers.get(sex_val, 'o')}-", capsize=4, label=sex_val,
                color=sex_colors.get(sex_val, 'black'), linewidth=1.5, markersize=6
            )
        ax.set_xticks(np.arange(len(time_order)))
        ax.set_xticklabels(['Baseline', '3 min', '1 hour', '2 hours'])
        ax.set_xlabel("Time Point"); ax.set_ylabel("Median Size (nm) (± 95% CI)")
        ax.set_title("Temporal Dynamics: EV Size")
        
        y_max = summary['mean'].max() * 1.12
        add_bracket(ax, 2, 3, y_max, y_max*0.02, "* p = 0.048")
        y_max = summary['mean'].max() * 1.20
        add_bracket(ax, 1, 2, y_max, y_max*0.02, "* p = 0.029")
        
        ax.legend(frameon=False, prop={'weight': 'bold', 'size': 7})
        ax.grid(True, alpha=0.2, linestyle='--')
        st.pyplot(fig)

    # --- VIEW 4: EMM GROUP DIFFERENCES ---
    elif selected_view == 'Figure 2CD: EMM Group Differences':
        fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.5), layout='constrained')
        df_emm = pd.DataFrame({
            'Metabolite': ['EV Concentration (/ml)'] * 2 + ['EV Size (nm)'] * 2,
            'Group': ['Male', 'Female'] * 2,
            'EMM': [2.56e11, 1.94e11, 102.0, 120.0],
            'CI_lower': [1.44e11, 8.17e10, 93.7, 111.1],
            'CI_upper': [3.68e11, 3.06e11, 111.0, 129.0]
        })
        
        # Subplot 1: EV Size
        sub_size = df_emm[df_emm['Metabolite'] == 'EV Size (nm)']
        for idx, row in sub_size.iterrows():
            x_pos = 0 if row['Group'] == 'Male' else 1
            axes[0].errorbar(x_pos, row['EMM'], yerr=[[row['EMM'] - row['CI_lower']], [row['CI_upper'] - row['EMM']]],
                       fmt='o' if x_pos==0 else 's', color='0.2' if x_pos==0 else '0.6', capsize=5, markersize=7)
        axes[0].set_xticks([0, 1]); axes[0].set_xticklabels(['Male', 'Female']); axes[0].set_xlim(-0.2, 1.2)
        axes[0].set_ylabel('Adjusted EMM (nm)'); axes[0].set_title('EV Size')
        add_bracket(axes[0], 0, 1, 131, 2, "** p = 0.009")
        axes[0].grid(axis='y', linestyle='--', alpha=0.5)

        # Subplot 2: EV Concentration
        sub_conc = df_emm[df_emm['Metabolite'] == 'EV Concentration (/ml)']
        for idx, row in sub_conc.iterrows():
            x_pos = 0 if row['Group'] == 'Male' else 1
            axes[1].errorbar(x_pos, row['EMM'], yerr=[[row['EMM'] - row['CI_lower']], [row['CI_upper'] - row['EMM']]],
                       fmt='o' if x_pos==0 else 's', color='0.2' if x_pos==0 else '0.6', capsize=5, markersize=7)
        axes[1].set_xticks([0, 1]); axes[1].set_xticklabels(['Male', 'Female']); axes[1].set_xlim(-0.2, 1.2)
        axes[1].set_ylabel('Adjusted EMM (/ml)'); axes[1].set_title('EV Concentration')
        add_bracket(axes[1], 0, 1, 3.9e11, 1e10, "p = 0.419 (ns)")
        
        formatter = mticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True); formatter.set_powerlimits((-1, 1))
        axes[1].yaxis.set_major_formatter(formatter)
        axes[1].grid(axis='y', linestyle='--', alpha=0.5)
        
        plt.suptitle("Estimated Marginal Means (Baseline Adjusted)", fontweight='bold')
        st.pyplot(fig)

    # --- VIEW 5: ANCOVA SUMMARY TABLE ---
    elif selected_view == 'ANCOVA Table (DF, F-stat, p-FDR, pes)':
        st.markdown("""
        <div style="background-color: #fff3cd; border-left: 4px solid #ffc107; padding: 10px 14px; margin-bottom: 12px; border-radius: 4px; font-size: 12px; line-height: 1.5; color: #856404;">
            <b>📄 Reviewer Note on Output Alignment:</b> The table below uses an automated <code>R (afex/emmeans)</code> engine. 
            To inspect the original primary <b>Jamovi statistical report</b> used for manuscript figures: 
            <a href="https://github.com/ernestonifade/GLYMREG-Extracellular-Vesicle-Study/raw/main/data/Jamovi_Statistical_Report_Figure2.pdf" target="_blank" style="color: #533f03; font-weight: bold; text-decoration: underline;">
                Download Jamovi PDF (GitHub) ↗
            </a>
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("RM-ANCOVA Model Summary (Main & Interaction Effects)")
        st.dataframe(ancova_df, use_container_width=True)
        
        ancova_note = """
        <div style="background-color: #f8f9fa; border-left: 4px solid #007bff; padding: 14px; margin-top: 15px; border-radius: 4px; font-size: 12px; line-height: 1.6; color: #212529;">
            <b>📊 Notes on ANCOVA Model Terms & Column Definitions:</b><br>
            • <b>Model Effect Definitions:</b>
              <ul style="margin: 3px 0 8px 18px; padding: 0;">
                <li><b>Time Effect (Main Effect):</b> Evaluates overall changes across recovery time points.</li>
                <li><b>Group Effect / Sex (Main Effect):</b> Evaluates overall differences between Males and Females.</li>
                <li><b>Time × Group Effect (Interaction):</b> Evaluates whether temporal trajectory across recovery differs between sexes.</li>
              </ul>
            • <b>Partial_Eta_Squared (η<sub>p</sub>²):</b> Effect size (Small ≈ 0.01, Medium ≈ 0.06, Large ≥ 0.14).
        </div>
        """
        st.markdown(ancova_note, unsafe_allow_html=True)

    # --- VIEW 6: POST-HOC PAIRWISE CONTRASTS ---
    elif selected_view == 'Post-Hoc Pairwise Comparisons (Within & Between)':
        st.subheader("Post-Hoc Pairwise Contrasts (emmeans)")
        st.dataframe(posthoc_df, use_container_width=True)
        
        posthoc_note = """
        <div style="background-color: #f8f9fa; border-left: 4px solid #28a745; padding: 14px; margin-top: 15px; border-radius: 4px; font-size: 12px; line-height: 1.6; color: #212529;">
            <b>📊 Notes on Pairwise Contrasts & Column Layout:</b><br>
            • <b>estimate:</b> Difference in Baseline-Adjusted Estimated Marginal Means (EMMs).<br>
            • <b>p_value_raw & p_value_adjusted:</b> Unadjusted and False Discovery Rate adjusted p-values.
        </div>
        """
        st.markdown(posthoc_note, unsafe_allow_html=True)

    # --- VIEW 7: CORRELATION OVERALL ---
    elif selected_view == 'Figure 2E: EV Conc vs Size Correlation (Overall)':
        st.markdown("""
        <div style="background-color: #f8f9fa; border-left: 4px solid #28a745; padding: 10px 14px; margin-bottom: 12px; border-radius: 4px; font-size: 11px; line-height: 1.5; color: #343a40;">
            <b>Figure 2E (Overall Pooled):</b> Pearson correlation (r) and 95% Confidence Intervals calculated across the combined cohort at each recovery time point regardless of sex.
        </div>
        """, unsafe_allow_html=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(6.5, 5.2), layout='constrained')
        axes = axes.ravel()
        for idx, t_val in enumerate(time_order):
            sub = long_df[long_df['time'] == t_val].dropna(subset=["Median Value (nm)", "Concentration/ml"])
            r_info = corr_overall[corr_overall['TimePoint'] == t_val].iloc[0]
            sns.regplot(data=sub, x="Median Value (nm)", y="Concentration/ml", ax=axes[idx], color='#0056b3', ci=95)
            axes[idx].set_title(f"Time: {t_val.capitalize()}", fontweight='bold')
            axes[idx].legend([Line2D([0], [0], color='#0056b3', lw=1.5)], [f"Overall{get_stars(r_info['p_value_raw'])}: r={r_info['r']:.2f}"], frameon=False, loc='best')
        plt.suptitle("Figure 2E: EV Concentration vs Size Correlation (Overall)", fontweight='bold')
        st.pyplot(fig)

        disp_df_formatted = corr_overall.copy()
        disp_df_formatted['r'] = disp_df_formatted['r'].round(3)
        disp_df_formatted['CI [95%]'] = disp_df_formatted.apply(lambda r: f"[{r['CI_95_lower']:.2f}, {r['CI_95_upper']:.2f}]", axis=1)
        disp_df_formatted['p_value_raw'] = disp_df_formatted['p_value_raw'].apply(lambda p: f"{p:.4f}" if p >= 0.0001 else "< 0.0001")
        disp_df_formatted['p_value_FDR'] = disp_df_formatted['p_value_FDR'].apply(lambda p: f"{p:.4f}" if p >= 0.0001 else "< 0.0001")
        
        st.markdown("<b>📊 Statistical Summary: Pearson Correlation (r) & FDR Corrections</b>", unsafe_allow_html=True)
        st.dataframe(disp_df_formatted[['TimePoint', 'N', 'r', 'CI [95%]', 'p_value_raw', 'p_value_FDR']], use_container_width=True)

    # --- VIEW 8: CORRELATION BY SEX ---
    elif selected_view == 'Figure 2E: EV Conc vs Size Correlation (By Sex)':
        st.markdown("""
        <div style="background-color: #f8f9fa; border-left: 4px solid #007bff; padding: 10px 14px; margin-bottom: 12px; border-radius: 4px; font-size: 11px; line-height: 1.5; color: #343a40;">
            <b>Figure 2E (Stratified by Sex):</b> Pearson correlation (r) and 95% Confidence Intervals calculated independently for Males and Females at each recovery time point.
        </div>
        """, unsafe_allow_html=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(6.5, 5.2), layout='constrained')
        axes = axes.ravel()
        for idx, t_val in enumerate(time_order):
            sub = long_df[long_df['time'] == t_val].dropna(subset=["Median Value (nm)", "Concentration/ml", "sex"])
            for s_val, col in zip(['Female', 'Male'], ['0.6', '0.1']):
                s_sub = sub[sub['sex'] == s_val]
                if not s_sub.empty:
                    sns.regplot(data=s_sub, x="Median Value (nm)", y="Concentration/ml", ax=axes[idx], color=col, ci=95)
            axes[idx].set_title(f"Time: {t_val.capitalize()}", fontweight='bold')
        plt.suptitle("Figure 2E: EV Concentration vs Size Correlation (By Sex)", fontweight='bold')
        st.pyplot(fig)

        disp_df_formatted = corr_sex.copy()
        disp_df_formatted['r'] = disp_df_formatted['r'].round(3)
        disp_df_formatted['CI [95%]'] = disp_df_formatted.apply(lambda r: f"[{r['CI_95_lower']:.2f}, {r['CI_95_upper']:.2f}]", axis=1)
        disp_df_formatted['p_value_raw'] = disp_df_formatted['p_value_raw'].apply(lambda p: f"{p:.4f}" if p >= 0.0001 else "< 0.0001")
        disp_df_formatted['p_value_FDR'] = disp_df_formatted['p_value_FDR'].apply(lambda p: f"{p:.4f}" if p >= 0.0001 else "< 0.0001")

        st.markdown("<b>📊 Statistical Summary: Pearson Correlation (r) & FDR Corrections</b>", unsafe_allow_html=True)
        st.dataframe(disp_df_formatted[['TimePoint', 'Sex', 'N', 'r', 'CI [95%]', 'p_value_raw', 'p_value_FDR']], use_container_width=True)
