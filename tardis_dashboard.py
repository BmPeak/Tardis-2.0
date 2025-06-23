import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import locale

def load_data():
    return pd.read_csv("cleaned_dataset.csv")

def load_model():
    return joblib.load("model.pkl")

def time_to_minutes(t):
    # Accept int hour, convert to minutes (e.g. 12 -> 720)
    if isinstance(t, int):
        return t * 60
    # or if string like "00:03"
    if isinstance(t, str) and ":" in t:
        h, m = map(int, t.split(":"))
        return h * 60 + m
    return np.nan

df = load_data()
model = load_model()

locale.setlocale(locale.LC_TIME, "fr_FR.UTF-8")
df["Date"] = pd.to_datetime(df["Date"])
df["day_of_week"] = df["Date"].dt.strftime("%A")
day_order = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]

st.set_page_config(page_title="Tardis Display Dashboard", layout="wide")
st.sidebar.markdown(
    """
    <div style="display: flex; align-items: center;">
        <img src="https://upload.wikimedia.org/wikipedia/fr/thumb/a/a1/Logo_SNCF_%282011%29.svg/1200px-Logo_SNCF_%282011%29.svg.png" style="height:60px;">
    </div>
    """,
    unsafe_allow_html=True
)
st.title("🚆 TARDIS - Dashboard de la prédiction du retard des trains")

st.sidebar.header("📥 Prédiction du retard")

departure_station = st.sidebar.selectbox("Gare de départ", df["Departure station"].unique())
arrival_station = st.sidebar.selectbox("Gare d'arrivée", df["Arrival station"].unique())
avg_journey_time = st.sidebar.slider("Durée moyenne du trajet (min)", 10, 300, 60)
scheduled_trains = st.sidebar.number_input("Nombre de trains programmés", min_value=1, value=10)
day_of_week = st.sidebar.selectbox("Jour de la semaine", day_order)
avg_delay_hour = st.sidebar.slider("Heure moyenne de départ (retard)", 0, 23, 12)
avg_delay_late_hour = st.sidebar.slider("Heure moyenne de départ (trains en retard)", 0, 23, 12)

input_data = pd.DataFrame([{
    "Departure station": departure_station,
    "Arrival station": arrival_station,
    "Average journey time": avg_journey_time,
    "Number of scheduled trains": scheduled_trains,
    "day_of_week": day_of_week,
    "Average delay departure in Hour": time_to_minutes(avg_delay_hour),
    "Average delay late departure in Hour": time_to_minutes(avg_delay_late_hour),
}])

if st.sidebar.button("Prédire"):
    try:
        prediction = model.predict(input_data)[0]
        st.sidebar.success(f"Prédiction d'un délai de: {round(prediction, 2)} minutes")
    except Exception as e:
        st.sidebar.error(f"Prediction failed: {e}")

# --- Main Dashboard ---

st.subheader("📊 Résumé des statistiques")
col1, col2, col3 = st.columns(3)
col1.metric("Retard Moyen", f"{df['Average delay of all trains at departure'].mean():.2f} min")
col2.metric("Retard Maximum", f"{df['Average delay of all trains at departure'].max(): .0f} min")
col3.metric("Pourcentage des trains en retard", f"{(df['Average delay of all trains at departure'] <= 0).mean() * 100:.1f}%")

st.divider()

st.subheader("📈 Distribution du retard")
fig1, ax1 = plt.subplots()
sns.histplot(df["Average delay of all trains at departure"], bins=40, kde=True, ax=ax1)
st.pyplot(fig1)

st.subheader("📍 Retard Moyen par Gare de départ")
station_delay = df.groupby("Departure station")["Average delay of all trains at departure"].mean().sort_values(ascending=False)
st.bar_chart(station_delay)

st.subheader("📆 Retard selon les jours de la semaine")
day_delay = df.groupby("day_of_week")["Average delay of all trains at departure"].mean().reindex(day_order)
st.line_chart(day_delay)

st.subheader("🔥 Correlation Heatmap")
fig2, ax2 = plt.subplots(figsize=(10, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax2)
st.pyplot(fig2)
