# figures/figure9.py
import os
import uuid
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
import streamlit as st

mpl.rcParams["svg.fonttype"] = "none"


@st.cache_data
def run_pls_pipeline(target_column="Median Value (nm)"):
  script_dir = os.path.dirname(os.path.abspath(__file__))
  root_dir = os.path.dirname(script_dir)  # moves up to root project directory

  def get_data_path(filename):
    path_data_folder = os.path.join(root_dir, "data", filename)
    path_root = os.path.join(root_dir, filename)
    path_local_data = os.path.join(script_dir, "data", filename)
    path_local = os.path.join(script_dir, filename)

    for p in [path_data_folder, path_root, path_local_data, path_local]:
      if os.path.exists(p):
        return p
    return filename

  e1 = pd.read_excel(get_data_path("bodymetrics.xlsx"))
  df_metrics_prep = e1.copy()
  df_metrics_prep.columns = df_metrics_prep.columns.str.strip()

  df_metrics_prep["time"] = (
      df_metrics_prep["time"].astype(str).str.strip().str.lower()
  )
  df_metrics_prep["sex_clean"] = (
      df_metrics_prep["sex"].astype(str).str.strip().str.lower()
  )
  df_metrics_prep["sex_encoded"] = df_metrics_prep["sex_clean"].map(
      {"male": 0, "m": 0, "1": 0, "female": 1, "f": 1, "2": 1}
  )

  is_male = (df_metrics_prep["sex_encoded"] == 0).astype(int)
  is_female = (df_metrics_prep["sex_encoded"] == 1).astype(int)

  exclude_cols = ["Subject_ID", "time", "sex", "sex_clean", "sex_encoded"]
  raw_body_metrics = [col for col in e1 if col not in exclude_cols]

  static_predictors = ["sex_encoded"]

  for metric in raw_body_metrics:
    if metric in df_metrics_prep.columns:
      df_metrics_prep[metric] = pd.to_numeric(
          df_metrics_prep[metric], errors="coerce"
      )
      male_col = f"{metric}_Male"
      female_col = f"{metric}_Female"
      df_metrics_prep[male_col] = df_metrics_prep[metric] * is_male
      df_metrics_prep[female_col] = df_metrics_prep[metric] * is_female
      static_predictors.extend([male_col, female_col])

  for col in static_predictors:
    df_metrics_prep[col] = pd.to_numeric(df_metrics_prep[col], errors="coerce")

  X_matrix = (
      df_metrics_prep[df_metrics_prep["time"] == "time1"][
          ["Subject_ID"] + static_predictors
      ]
      .set_index("Subject_ID")
      .dropna()
  )

  df1 = pd.read_excel(get_data_path("df1_EVs.xlsx"))
  df2 = pd.read_excel(get_data_path("df2_EVs.xlsx"))
  df3 = pd.read_excel(get_data_path("df3_EVs.xlsx"))
  df4 = pd.read_excel(get_data_path("df4_EVs.xlsx"))
  df = pd.concat([df1, df2, df3, df4], axis=0)

  df_biomarker_prep = df[["Subject_ID", "time", "sex", target_column]].copy()
  df_biomarker_prep.columns = df_biomarker_prep.columns.str.strip()
  df_biomarker_prep["time"] = (
      df_biomarker_prep["time"].astype(str).str.strip().str.lower()
  )

  admin_cols_y = ["Subject_ID", "time", "sex"]
  dynamic_cols = [c for c in df_biomarker_prep.columns if c not in admin_cols_y]

  for col in dynamic_cols:
    df_biomarker_prep[col] = pd.to_numeric(
        df_biomarker_prep[col], errors="coerce"
    )

  delta_t2 = (
      df_biomarker_prep[df_biomarker_prep["time"] == "time2"][
          ["Subject_ID"] + dynamic_cols
      ]
      .set_index("Subject_ID")
      .sub(
          df_biomarker_prep[df_biomarker_prep["time"] == "time1"]
          .set_index("Subject_ID")[dynamic_cols],
          fill_value=0,
      )
  )
  delta_t3 = (
      df_biomarker_prep[df_biomarker_prep["time"] == "time3"][
          ["Subject_ID"] + dynamic_cols
      ]
      .set_index("Subject_ID")
      .sub(
          df_biomarker_prep[df_biomarker_prep["time"] == "time1"]
          .set_index("Subject_ID")[dynamic_cols],
          fill_value=0,
      )
  )
  delta_t4 = (
      df_biomarker_prep[df_biomarker_prep["time"] == "time4"][
          ["Subject_ID"] + dynamic_cols
      ]
      .set_index("Subject_ID")
      .sub(
          df_biomarker_prep[df_biomarker_prep["time"] == "time1"]
          .set_index("Subject_ID")[dynamic_cols],
          fill_value=0,
      )
  )

  Y_matrix_raw = (delta_t2 + delta_t3 + delta_t4) / 3
  Y_matrix_raw.columns = [f"{c}_avg_delta" for c in Y_matrix_raw.columns]

  final_data = X_matrix.join(Y_matrix_raw).dropna()
  X = final_data[static_predictors]
  Y = final_data[[c for c in final_data.columns if c not in static_predictors]]

  scaler_x, scaler_y = StandardScaler(), StandardScaler()
  X_scaled = scaler_x.fit_transform(X)
  Y_scaled = scaler_y.fit_transform(Y)

  n_comp = 2
  pls = PLSRegression(n_components=n_comp)
  y_pred_cv = cross_val_predict(pls, X_scaled, Y_scaled, cv=5)
  r2_cv = np.corrcoef(Y_scaled.flatten(), y_pred_cv.flatten())[0, 1] ** 2

  pls.fit(X_scaled, Y_scaled)

  total_variance_y = np.sum(np.var(Y_scaled, axis=0))
  Y_pred_c1 = np.outer(pls.x_scores_[:, 0], pls.y_loadings_[:, 0])
  pct_comp1 = (np.sum(np.var(Y_pred_c1, axis=0)) / total_variance_y) * 100

  Y_pred_c2 = np.outer(pls.x_scores_[:, 1], pls.y_loadings_[:, 1])
  pct_comp2 = (np.sum(np.var(Y_pred_c2, axis=0)) / total_variance_y) * 100
  total_pct = pct_comp1 + pct_comp2

  t, w, q = pls.x_scores_, pls.x_weights_, pls.y_loadings_
  p, h = w.shape
  vips = np.zeros((p,))
  s = np.diag(t.T @ t @ q.T @ q).reshape(h, 1)
  total_s = np.sum(s)
  for i in range(p):
    vips[i] = np.sqrt(
        p
        * np.sum(
            [
                s[j] * (w[i, j] / np.linalg.norm(w[:, j])) ** 2
                for j in range(h)
            ]
        )
        / total_s
    )

  predictor_importance = pd.DataFrame(
      {"Metric": static_predictors, "VIP Score": vips}
  ).sort_values(by="VIP Score", ascending=False)

  comp1_scores, _ = pls.transform(X_scaled, Y_scaled)
  comp1_scores = comp1_scores[:, 0]

  Y_pred_scaled = pls.predict(X_scaled)
  Y_pred_original = scaler_y.inverse_transform(Y_pred_scaled)
  Y_actual_original = scaler_y.inverse_transform(Y_scaled)

  return (
      predictor_importance,
      r2_cv,
      pct_comp1,
      pct_comp2,
      total_pct,
      comp1_scores,
      Y,
      X,
      Y_actual_original,
      Y_pred_original,
      pls,
      X_scaled,
      Y_scaled,
  )


def load_fig9_results():
  return run_pls_pipeline()


def render_figure9():
  st.title(
      "📊 Figure 9: Biophysical predictors of EV size shifts"
      " (Multivariate PLS)"
  )
  st.markdown(
      "Multivariate evaluation of baseline biophysical profiles against"
      " exercise-induced shifts in extracellular vesicle (EV) size."
  )

  (
      predictor_importance,
      r2_cv,
      pct_comp1,
      pct_comp2,
      total_pct,
      comp1_scores,
      Y_outcome,
      X_data,
      Y_actual,
      Y_pred,
      pls,
      X_scaled,
      Y_scaled,
  ) = run_pls_pipeline()

  # Main canvas view selector consistent with other figure modules
  view_selection = st.radio(
      "Select Figure 9 View:",
      [
          "View 1: Model Overview & Summary",
          "View 2: Component 1 Scatter Plot",
          "View 3: Actual vs. Predicted Parity Plot",
          "View 4: Complete VIP Table",
      ],
      horizontal=True,
      key="fig9_nav_view_selector",
  )
  st.markdown("---")

  if view_selection == "View 1: Model Overview & Summary":
    st.subheader("View 1: Model Performance & Feature Importance (VIP)")
    col1, col2 = st.columns([1, 1])

    with col1:
      st.markdown("### Model Summary Metrics")
      metrics_df = pd.DataFrame({
          "Metric Parameter": [
              "Cross-Validated R² (cv=5)",
              "Component 1 Variance (%)",
              "Component 2 Variance (%)",
              "Total Cumulative Variance (%)",
          ],
          "Value": [
              f"{r2_cv:.3f}",
              f"{pct_comp1:.2f}%",
              f"{pct_comp2:.2f}%",
              f"{total_pct:.2f}%",
          ],
      })
      st.table(metrics_df)

    with col2:
      st.markdown("### Top Feature Importance (VIP)")
      fig, ax = plt.subplots(figsize=(5, 4))
      colors = [
          "#0072B2" if val >= 1.0 else "#b0bec5"
          for val in predictor_importance["VIP Score"]
      ]
      sns.barplot(
          x=predictor_importance["VIP Score"][:10],
          y=predictor_importance["Metric"][:10],
          palette=colors[:10],
          ax=ax,
          hue=predictor_importance["Metric"][:10],
          legend=False,
      )
      ax.axvline(1.0, color="#b71c1c", linestyle="--", linewidth=1.2)
      ax.set_title("Top 10 VIP Scores (VIP > 1.0 Threshold)", fontweight="bold")
      ax.set_xlabel("VIP Score Value", fontweight="bold")
      ax.set_ylabel("")
      sns.despine(ax=ax, trim=True)
      st.pyplot(fig)

  elif view_selection == "View 2: Component 1 Scatter Plot":
    st.subheader(
        "View 2: Biophysical Baseline Gradient vs. EV Size Shift"
    )
    st.markdown(
        "Explore how exercise-induced shifts in EV size align with"
        " key biophysical predictors or latent PLS components."
    )

    X_comp_scores, _ = pls.transform(X_scaled, Y_scaled)

    # Dictionary containing ONLY Component 1, Component 2, and your selected key VIP metrics
    available_x_options = {
        "PLS Component 1 (Default)": X_comp_scores[:, 0],
        "PLS Component 2": X_comp_scores[:, 1],
    }

    # Safely add only your chosen key VIP metrics if they exist in X_data
    target_metrics = {
	"Male_Fat%": "FAT%_Male",
	"Female_Fat%": "FAT%_Female",
        "Male_VO2peak": "VO2peak(ml/kg/min)_Male",
	"Female_VO2peak": "VO2peak(ml/kg/min)_Female",
        "Male_Waist_CM": "Waist Circumference(cm)_Male",
        "Female_Waist_CM": "Waist Circumference(cm)_Female",
        "Male_Diastolic_BP": "Diastolic BP(mm Hg)_Male",
	"Female_Diastolic_BP": "Diastolic BP(mm Hg)_Female",
	
    }

    for label, col_name in target_metrics.items():
      if col_name in X_data.columns:
        available_x_options[label] = X_data[col_name].values

    # Contextual dropdown selector
    selected_x_label = st.selectbox(
        "Select X-Axis Predictor / Component:",
        list(available_x_options.keys()),
        index=0,
        key="fig8_view2_xaxis_selector",
    )

    current_x_data = available_x_options[selected_x_label]

    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    sns.scatterplot(
        x=current_x_data,
        y=Y_outcome.iloc[:, 0],
        hue=X_data["sex_encoded"].map({0: "Male", 1: "Female"}),
        palette={"Male": "red", "Female": "blue"},
        s=45,
        alpha=0.85,
        edgecolor="black",
        linewidth=0.6,
        ax=ax,
    )
    ax.legend(loc="lower left", frameon=False)

    sns.regplot(
        x=current_x_data,
        y=Y_outcome.iloc[:, 0],
        ax=ax,
        scatter=False,
        color="#333333",
        line_kws={"linewidth": 2.0, "linestyle": "-"},
        ci=95,
    )

    formatter = plt.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))
    ax.yaxis.set_major_formatter(formatter)

    ax.set_title(
        f"Directional Impact of\n{selected_x_label} on $\Delta$ EV"
        " Size",
        fontweight="bold",
    )
    ax.set_xlabel(selected_x_label, fontweight="bold")
    ax.set_ylabel(
        r"$\Delta$ EV Size (Post - Pre, (nm))", fontweight="bold"
    )
    sns.despine(ax=ax, trim=True)
    st.pyplot(fig)
    
  elif view_selection == "View 3: Actual vs. Predicted Parity Plot":
    st.subheader("View 3: Multivariate PLS Model Accuracy (Parity Plot)")
    fig, ax = plt.subplots(figsize=(3.5, 3.5))

    # Plot sex-stratified scatter points matching View 2 styling
    sns.scatterplot(
        x=Y_actual.flatten(),
        y=Y_pred.flatten(),
        hue=X_data["sex_encoded"].map({0: "Male", 1: "Female"}),
        palette={"Male": "red", "Female": "blue"},
        s=55,
        alpha=0.85,
        edgecolor="k",
        linewidth=0.6,
        ax=ax,
    )

    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()]),
    ]
    ax.plot(
        lims,
        lims,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label="Ideal Fit",
    )
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    # Embed model performance metrics as a clean text annotation/legend box on the plot
    text_str = f"Cross-Validated $R^2$ (cv=5): {r2_cv:.3f}\nTotal Cumulative Var: {total_pct:.2f}%"
    props = dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="#cccccc")
    ax.text(
        0.05,
        0.95,
        text_str,
        transform=ax.transAxes,
        fontsize=7,
        verticalalignment="top",
        bbox=props,
    )

    ax.set_xlabel("Actual Baseline-Adjusted Shift", fontweight="bold")
    ax.set_ylabel(
        "PLS Model Predicted Shift (Full Multivariate)", fontweight="bold"
    )
    ax.set_title(
        "Multivariate PLS Model: Actual vs. Predicted Responses", fontweight="bold"
    )
    ax.legend(loc="lower right", frameon=False)
    sns.despine(ax=ax, trim=True)
    st.pyplot(fig)
    
  elif view_selection == "View 4: Complete VIP Table":
    st.subheader("View 4: Complete Variable Importance in Projection Table")
    st.markdown(
        "Complete list of all sex-stratified body metrics ranked by their VIP"
        " score contribution."
    )
    display_table = predictor_importance.copy()
    display_table["Significance"] = np.where(
        display_table["VIP Score"] >= 1.0, "Significant (> 1.0)", "Below Threshold"
    )
    st.dataframe(display_table.reset_index(drop=True), use_container_width=True)
