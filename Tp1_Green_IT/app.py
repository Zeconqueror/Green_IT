import streamlit as st
import pandas as pd
import plotly.express as px

# =======================================
# 1️⃣ Configuration
# =======================================
st.set_page_config(page_title="LLM Evaluation Dashboard", layout="wide")
st.title("🌍 LLM Evaluation Dashboard: Percformance vs Impact")

# =======================================
# 2️⃣ Load Data
# =======================================
df = pd.read_csv("data.csv", sep=";")

# --- Dataset preview ---
with st.expander("📂 Show/Hide Dataset Preview", expanded=False):
    st.subheader("Dataset Preview")
    st.dataframe(df)

# =======================================
# 3️⃣ Filters Removed (all data used)
# =======================================
# On utilise tout le dataset sans filtre
filtered_df = df.copy()

# =======================================
# 1️⃣ Model Overview: Performance & Consumption (Line Charts, compact)
# =======================================

st.subheader("Model Overview: Metrics Summary")

# --- View mode uniquement ---
view_option = st.radio(
    "Select View Mode",
    options=["View by Model", "View by Model Category"]
)

# --- Agrégation selon le mode de vue ---
if view_option == "View by Model":
    summary_df = df.groupby("Model", as_index=False).agg({
        "Answer_quality": "mean",
        "Electricity_consumption": "mean",
        "CO2_emission": "mean",
        "Inference_timing": "mean"
    }).rename(columns={
        "Answer_quality": "Avg Quality",
        "Electricity_consumption": "Avg Electricity (Wh)",
        "CO2_emission": "Avg CO2 (g)",
        "Inference_timing": "Avg Inference Time (s)"
    })
    x_axis = "Model"
else:
    summary_df = df.groupby("Model_Category", as_index=False).agg({
        "Answer_quality": "mean",
        "Electricity_consumption": "mean",
        "CO2_emission": "mean",
        "Inference_timing": "mean"
    }).rename(columns={
        "Answer_quality": "Avg Quality",
        "Electricity_consumption": "Avg Electricity (Wh)",
        "CO2_emission": "Avg CO2 (g)",
        "Inference_timing": "Avg Inference Time (s)"
    })
    x_axis = "Model_Category"


# --- 4️⃣ Visualisation ---
metrics = ["Avg Quality", "Avg Electricity (Wh)", "Avg CO2 (g)", "Avg Inference Time (s)"]
st.subheader("Visual Comparison by Metric")

cols = st.columns(2)  # 2 graphiques par ligne

for i, metric in enumerate(metrics):
    fig = px.line(
        summary_df.sort_values(metric, ascending=False),
        x=x_axis,
        y=metric,
        text=metric,
        markers=True,
        color_discrete_sequence=["#004080"],
    )
    fig.update_traces(texttemplate="%{text:.1f}", textposition="top center")
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis_title="",
        yaxis_title=metric,
        height=300
    )
    # Afficher 2 graphiques par ligne
    cols[i % 2].plotly_chart(fig, use_container_width=True)





# =======================================
# 4️⃣ Heatmap: Average Answer Quality per Model Type & Question Type (Monochrome)
# =======================================
st.subheader("Average Answer Quality per Model Type & Question Type")
# Calculer la moyenne par Model_Category et Question_Category
heatmap_data = filtered_df.groupby(
    ["Model_Category", "Question_Category"], as_index=False
)["Answer_quality"].mean()

# Créer la heatmap monochrome
fig_heatmap = px.density_heatmap(
    heatmap_data,
    x="Question_Category",
    y="Model_Category",
    z="Answer_quality",
    color_continuous_scale=["#e0f3ff", "#004080"],  # clair → foncé (monochrome bleu)
    text_auto=True,
    labels={"Answer_quality": "Avg Answer Quality"}
)

# Mise en forme UX/UI
fig_heatmap.update_traces(textfont_size=14, textfont_color="black")
fig_heatmap.update_layout(
    title={
        'text': "",
        'y': 0.95,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top'
    },
    xaxis_title="Question Type",
    yaxis_title="Model Type",
    yaxis={'categoryorder': 'total ascending'},
    plot_bgcolor="white",
    paper_bgcolor="white",
    coloraxis_colorbar=dict(
        title="Avg Quality",
        tickvals=[1, 2, 3, 4, 5]
    )
)

# Affichage dans Streamlit
st.plotly_chart(fig_heatmap, use_container_width=True)


# =======================================
# 5️⃣ Line Chart: Avg Answer Quality & Inference Time per Model Type
# =======================================

st.subheader("Avg Answer Quality & Inference Time per Model Type")

# Préparation des données agrégées
line_data = filtered_df.groupby("Model_Category", as_index=False).agg({
    "Answer_quality": "mean",
    "Inference_timing": "mean"
}).round(1)

# On crée une version "longue" pour tracer deux courbes sur le même graphique
line_melted = line_data.melt(
    id_vars="Model_Category",
    value_vars=["Answer_quality", "Inference_timing"],
    var_name="Metric",
    value_name="Value"
)

# Remplacer les noms des métriques pour un affichage plus propre
line_melted["Metric"] = line_melted["Metric"].replace({
    "Answer_quality": "Avg Answer Quality",
    "Inference_timing": "Avg Inference Time (s)"
})

# Création du graphique
fig_line = px.line(
    line_melted,
    x="Model_Category",
    y="Value",
    color="Metric",
    markers=True,
    text="Value",
    color_discrete_map={
        "Avg Answer Quality": "#004080",
        "Avg Inference Time (s)": "#82B1FF"
    })

# Mise en forme UX/UI
fig_line.update_traces(
    texttemplate="%{text:.1f}",
    textposition="top center",
    line=dict(width=3)
)
fig_line.update_layout(
    plot_bgcolor="white",
    paper_bgcolor="white",
    xaxis_title="Model Type",
    yaxis_title="Average Value",
    legend_title="Metric",
    title_x=0.5,
    height=400  # Réduit la hauteur pour une intégration fluide
)

# Affichage dans Streamlit
st.plotly_chart(fig_line, use_container_width=True)


# ---  Line Chart: Average CO2 & Electricity by Model_Category ---
st.markdown("Line Chart: Average Consumption per Inference Time by Model Category")

line_df = filtered_df.groupby(["Model_Category", "Inference_timing"], as_index=False).agg({
    "CO2_emission": "mean",
    "Electricity_consumption": "mean"
})

fig_line = px.line(
    line_df,
    x="Inference_timing",
    y=["CO2_emission", "Electricity_consumption"],
    color="Model_Category",
    markers=True,
    labels={
        "value": "Average Consumption",
        "Inference_timing": "Inference Time (s)",
        "variable": "Metric"
    },
    color_discrete_sequence=["#004080", "#1f77b4", "#3b82f6", "#60a5fa"]
)
fig_line.update_layout(
    plot_bgcolor="white",
    paper_bgcolor="white",
    legend_title="Model Category / Metric"
)
st.plotly_chart(fig_line, use_container_width=True)

# =======================================
# Bar Chart: Average CO₂ & Electricity per Question Category (Blue & Yellow)
# =======================================
st.subheader("📊 Average CO₂ & Electricity per Question Category")

# --- Calcul des moyennes par catégorie de question ---
avg_df = (
    filtered_df.groupby("Question_Category", as_index=False)
    .agg({
        "CO2_emission": "mean",
        "Electricity_consumption": "mean"
    })
    .rename(columns={
        "CO2_emission": "Average_CO2",
        "Electricity_consumption": "Average_Energy"
    })
)

# --- Transformation pour affichage côte à côte ---
melted_avg_df = avg_df.melt(
    id_vars="Question_Category",
    value_vars=["Average_CO2", "Average_Energy"],
    var_name="Metric",
    value_name="Average_Value"
)

# --- Bar chart (bleu & jaune) ---
fig_bar = px.bar(
    melted_avg_df,
    x="Question_Category",
    y="Average_Value",
    color="Metric",
    barmode="group",
    color_discrete_map={
        "Average_CO2": "#007BFF",   # 🔵 bleu vif
        "Average_Energy": "#FFD166" # 🟡 jaune doux
    },
    text_auto=".2s"
)

# --- Mise en forme ---
fig_bar.update_traces(
    textfont_size=12,
    textangle=0,
    textposition="outside",
    cliponaxis=False
)

fig_bar.update_layout(
    xaxis_title="Question Category",
    yaxis_title="Average Value",
    legend_title="Metric",
    plot_bgcolor="white",
    paper_bgcolor="white",
    margin=dict(l=30, r=30, t=40, b=30),
    font=dict(size=13),
)

st.plotly_chart(fig_bar, use_container_width=True)



# =======================================
#  Consumption: Electricity & CO2
# =======================================
st.subheader("CO2 vs Electricity Scatter Plot (log scale)")

# --- Scatter plot CO2 vs Electricity (en g et Wh) ---
fig_scatter = px.scatter(
    filtered_df,
    x="Electricity_consumption",  # Wh
    y="CO2_emission",             # g
    color="Model_Category",
    hover_data=["Model", "Question_Category", "Prompt_id"],
    size="Answer_quality",
    size_max=7,  
    log_x=True,
    log_y=True,
    color_discrete_sequence=["#004080", "#1f77b4", "#3b82f6", "#60a5fa"]  # nuances de bleu foncé
)

fig_scatter.update_layout(
    plot_bgcolor="white",
    paper_bgcolor="white",
    xaxis_title="Electricity Consumption (Wh) [log scale]",
    yaxis_title="CO2 Emission (g) [log scale]",
    legend_title="Model Category"
)
st.plotly_chart(fig_scatter, use_container_width=True)




# =======================================
# 7️⃣ Extended: Download PPT
# =======================================

with st.expander("📂Extended: TP1Compar'AI Presentation", expanded=False):
    st.subheader("Here is the PowerPoint file (.pptx)")

    # Chemin du fichier PPT déjà généré
    ppt_file_path = "dashboard.pptx"

    # Bouton de téléchargement
    with open(ppt_file_path, "rb") as f:
        ppt_bytes = f.read()
        st.download_button(
            label="📥 Download PPT",
            data=ppt_bytes,
            file_name="dahsboard.pptx",
            mime="application/vnd.openxmlformats-officedocument.presentationml.presentation"
        )
