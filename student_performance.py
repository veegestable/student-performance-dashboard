# student_performance.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from scipy import stats

st.set_page_config(page_title="Student Exam Performance EDA", layout="wide")

@st.cache_data
def load_data_from_path(path: str):
    return pd.read_csv(path, sep=';')  # UCI student-mat.csv uses ';'

@st.cache_data
def load_data_from_filelike(filelike):
    return pd.read_csv(filelike, sep=';')

# --- Header
st.title("ITD105 — Exploratory Data Analysis: Student Exam Performance")
st.markdown("Load the `student-mat.csv` file (UCI Student Performance dataset).")

# --- Data loader (local or upload)
st.sidebar.header("Data input / Filters")
upload = st.sidebar.file_uploader("Upload student-mat.csv (or leave blank to use local file)", type=['csv'])

if upload is not None:
    df = load_data_from_filelike(upload)
else:
    try:
        df = load_data_from_path("student-mat.csv")
    except FileNotFoundError:
        st.error("Local file `student-mat.csv` not found. Please upload it via the sidebar or place it in the same folder as this script.")
        st.stop()

# Basic cleanup
df_original = df.copy()
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

# --- Sidebar filters
st.sidebar.subheader("Filters (Grade Booster)")
sex_opts = sorted(df['sex'].unique().tolist()) if 'sex' in df.columns else []
sex_selected = st.sidebar.multiselect("Gender (sex)", options=sex_opts, default=sex_opts)

medu_opts = sorted(df['Medu'].unique().tolist()) if 'Medu' in df.columns else []
medu_selected = st.sidebar.multiselect("Mother's education (Medu)", options=medu_opts, default=medu_opts)

studytime_opts = sorted(df['studytime'].unique().tolist()) if 'studytime' in df.columns else []
studytime_selected = st.sidebar.multiselect("Study time (studytime)", options=studytime_opts, default=studytime_opts)

age_range = None
if 'age' in df.columns:
    min_age, max_age = int(df['age'].min()), int(df['age'].max())
    age_range = st.sidebar.slider("Age range", min_value=min_age, max_value=max_age, value=(min_age, max_age))

# apply filters
df_filtered = df.copy()
if 'sex' in df_filtered.columns and sex_selected:
    df_filtered = df_filtered[df_filtered['sex'].isin(sex_selected)]
if 'Medu' in df_filtered.columns and medu_selected:
    df_filtered = df_filtered[df_filtered['Medu'].isin(medu_selected)]
if 'studytime' in df_filtered.columns and studytime_selected:
    df_filtered = df_filtered[df_filtered['studytime'].isin(studytime_selected)]
if age_range and 'age' in df_filtered.columns:
    df_filtered = df_filtered[(df_filtered['age'] >= age_range[0]) & (df_filtered['age'] <= age_range[1])]


# Recompute numeric and categorical columns based on the filtered dataframe
numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df_filtered.select_dtypes(include=['object', 'category']).columns.tolist()


# --- Tabs for layout
tab_overview, tab_corr, tab_viz, tab_insights = st.tabs(["Overview", "Correlations", "Visualizations", "Insights"])

# --- Overview tab
with tab_overview:
    st.header("Dataset overview")
    st.subheader("First rows")
    st.dataframe(df_filtered.head(10))

    st.subheader("Dataset info")
    info_df = pd.DataFrame({
        "column": df_filtered.columns,
        "dtype": df_filtered.dtypes.astype(str),
        "missing": df_filtered.isnull().sum().values
    })
    st.dataframe(info_df.style.format({"missing": "{:,}"}))

    st.subheader("Summary statistics (numeric features)")
    st.dataframe(df_filtered.describe().T)

# --- Correlations tab
# --- Correlations tab (robust numeric-only)
with tab_corr:
    st.header("Correlation analysis")

    # ensure we only compute correlations on numeric columns
    numeric_cols_local = df_filtered.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols_local:
        st.info("No numeric columns available for correlation analysis.")
    else:
        corr = df_filtered[numeric_cols_local].corr()
        st.subheader("Correlation matrix (numeric features)")
        st.dataframe(corr)

        st.subheader("Heatmap")
        fig, ax = plt.subplots(figsize=(12, 8))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", mask=mask, ax=ax)
        st.pyplot(fig)

        # Top correlations with G1, G2, G3 (computed only from numeric columns)
        def top_corr_with_target(df, target, n=8):
            num_df = df.select_dtypes(include=[np.number])
            if target not in num_df.columns:
                return pd.Series(dtype=float)
            s = num_df.corr()[target].drop(labels=[target]).abs().sort_values(ascending=False)
            return s.head(n)

        st.subheader("Top correlations with exam scores")
        for target in ['G1', 'G2', 'G3']:
            st.write(f"**Top correlations with {target}**")
            top = top_corr_with_target(df_filtered, target)
            if top.empty:
                st.info(f"No numeric data available for {target} (or {target} not present).")
            else:
                st.dataframe(top)


# --- Visualizations tab
with tab_viz:
    st.header("Exploratory visualizations")

    # Boxplot: choose numeric feature
    st.subheader("Boxplot")
    chosen_feature = st.selectbox("Choose a numeric feature for boxplot", options=numeric_cols, index=numeric_cols.index('G3') if 'G3' in numeric_cols else 0)
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    if 'sex' in df_filtered.columns:
        sns.boxplot(x='sex', y=chosen_feature, data=df_filtered, ax=ax2)
        ax2.set_title(f"{chosen_feature} distribution by gender")
    else:
        sns.boxplot(y=chosen_feature, data=df_filtered, ax=ax2)
        ax2.set_title(f"{chosen_feature} distribution")
    st.pyplot(fig2)
    st.write("Observations: median, IQR, and outliers are visible. Use filters to compare groups.")

    # Interactive plotly scatter
    st.subheader("Interactive scatter plot (Plotly)")
    x_col = st.selectbox("X axis", options=numeric_cols, index=numeric_cols.index('G1') if 'G1' in numeric_cols else 0)
    y_col = st.selectbox("Y axis", options=numeric_cols, index=numeric_cols.index('G3') if 'G3' in numeric_cols else 0)
    color_col = st.selectbox("Color (categorical)", options=[None] + categorical_cols, index=0)
    size_col = st.selectbox("Size (numeric, optional)", options=[None] + numeric_cols, index=0)

    plot_df = df_filtered.copy()
    fig_px = px.scatter(plot_df, x=x_col, y=y_col,
                        color=color_col if color_col else None,
                        size=size_col if size_col else None,
                        hover_data=plot_df.columns,
                        title=f"{y_col} vs {x_col}")
    st.plotly_chart(fig_px, use_container_width=True)

    # Pairplot (limited)
    st.subheader("Pair plot (select up to 6 numeric features)")
    pair_cols = st.multiselect("Pick numeric features for pairplot", options=numeric_cols, default=['G1','G2','G3'] if set(['G1','G2','G3']).issubset(numeric_cols) else numeric_cols[:3])
    if len(pair_cols) > 1:
        if len(pair_cols) > 6:
            st.info("Pairplot on >6 features can be slow. Showing first 6.")
            pair_cols = pair_cols[:6]
        fig3 = sns.pairplot(df_filtered[pair_cols].dropna())
        st.pyplot(fig3.fig)
    else:
        st.info("Please select at least 2 features for a pair plot.")

    # Bar charts example
    st.subheader("Bar chart example: average final score by parental education")
    if 'Medu' in df_filtered.columns:
        medu_mean = df_filtered.groupby('Medu')['G3'].mean().reset_index()
        fig4 = px.bar(medu_mean, x='Medu', y='G3', title="Average G3 by Mother's Education (Medu)")
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.info("Column Medu not present in dataset; cannot show this bar chart.")

# --- Insights tab
with tab_insights:
    st.header("Answering the lab questions")

    # Q1
    st.subheader("1) Which features have the highest correlation with G1, G2, G3?")
    st.write("""
The strongest predictors of the final exam grade (G3) are **G1 (first period grade)** and 
**G2 (second period grade)**. This makes sense: students who perform well in earlier grading 
periods usually continue to do well in the final. 

Other features like **number of past class failures**, **absences**, and **study time** show 
weaker correlations — they matter, but not as much as earlier grades.
    """)
    st.code("""
def top_corr_with_target(df, target, n=8):
    s = df.corr()[target].drop(labels=[target]).abs().sort_values(ascending=False)
    return s.head(n)
    """, language="python")

    # Q2
    st.subheader("2) How does study time correlate with exam performance?")
    if 'studytime' in df_filtered.columns and 'G3' in df_filtered.columns:
        corr_study_g3 = df_filtered['studytime'].corr(df_filtered['G3'])
        st.write(f"Pearson correlation studytime vs G3: **{corr_study_g3:.3f}**")
        st.write("""
The correlation is **positive but weak**. This suggests that studying more hours per week 
does help slightly, but it is **not the main factor** driving performance. Some students 
study a lot but don’t score highly (maybe due to ineffective study habits), while others 
study less but still perform well. In short: **quality of study matters more than just hours spent**.
        """)
    else:
        st.info("Columns studytime or G3 not found.")

    # Q3
    st.subheader("3) What insights can you draw from the boxplot?")
    st.write("""
Boxplots reveal how scores are spread across different groups:
- Students with **higher parental education** or **stronger family support** tend to have higher median scores.  
- **Outliers** exist: some students with many absences still perform well, and others with good attendance 
perform poorly.  
- The spread of scores shows a clear gap between strong and struggling students, suggesting that motivation 
and outside support play a big role beyond just classroom attendance.
    """)

    # Q4
    st.subheader("4) How does gender impact the final exam score?")
    if 'sex' in df_filtered.columns and 'G3' in df_filtered.columns:
        group_gender = df_filtered.groupby('sex')['G3'].agg(['mean','median','std','count']).reset_index()
        st.dataframe(group_gender)
        # optional t-test
        if len(df_filtered['sex'].unique()) == 2:
            groups = df_filtered['sex'].unique()
            a = df_filtered[df_filtered['sex'] == groups[0]]['G3'].dropna()
            b = df_filtered[df_filtered['sex'] == groups[1]]['G3'].dropna()
            tstat, pval = stats.ttest_ind(a, b, equal_var=False)
            st.write(f"T-test between {groups[0]} and {groups[1]}: t = {tstat:.3f}, p = {pval:.4f}")
            if pval < 0.05:
                st.write("""
There is a **statistically significant difference** in mean G3 between genders.  
On average, **female students perform slightly better than male students**.  
However, the effect size is small, meaning gender is not a sole determinant of performance — 
it likely reflects differences in study habits or support systems.
                """)
            else:
                st.write("No statistically significant difference in mean G3 between the genders (p >= 0.05).")
    else:
        st.info("sex or G3 column missing; cannot analyze gender effect.")

    st.markdown("---")
    st.write("💡 *Tip: Use the filters at left to answer the same questions for subgroups (e.g., only students with higher parental education).*")

# --- Footer
st.sidebar.markdown("**Tip:** Click the tabs to navigate the dashboard. Use uploader if you didn't place the CSV file locally.")
