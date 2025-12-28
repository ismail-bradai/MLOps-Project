import streamlit as st
import requests
import pandas as pd
import json
import os
from datetime import datetime

# Configuration
API_URL = os.getenv("API_URL", "http://backend:8000" if os.path.exists("/.dockerenv") else "http://127.0.0.1:8000")

st.set_page_config(
    page_title="Traffic Severity Detection",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        color: black;
    }
    .prediction-card {
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        color: white;
    }
    .sev-1 { background-color: #4CAF50; } /* Low Impact */
    .sev-2 { background-color: #FFC107; color: black; } /* Minor */
    .sev-3 { background-color: #FF9800; } /* Major */
    .sev-4 { background-color: #F44336; } /* Siginificant */
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("🚦 Traffic Severity")
    page = st.radio("Navigation", ["🏠 Home", "🔍 Predict Severity", "📂 Batch Analysis"])
    
    st.markdown("---")
    st.caption("System Status")
    try:
        r = requests.get(f"{API_URL}/health", timeout=2)
        if r.status_code == 200:
            st.success("API Online")
        else:
            st.warning("API Degrading")
    except:
        st.error("API Offline")

if page == "🏠 Home":
    st.title("US Accidents Severity Detection")
    st.markdown("""
    Predict the severity of traffic accidents based on weather, location, and road conditions.
    
    **Severity Levels:**
    - **1**: Short delay, low impact.
    - **2**: Moderate delay.
    - **3**: Long delay, significant impact.
    - **4**: Very long delay, severe impact.
    """)
    
    try:
        info = requests.get(f"{API_URL}/model-info").json()
        st.json(info)
    except:
        pass

elif page == "🔍 Predict Severity":
    st.title("Predict Accident Severity")
    
    with st.form("accident_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("📍 Location & Time")
            lat = st.number_input("Latitude", value=37.7749, format="%.4f")
            lng = st.number_input("Longitude", value=-122.4194, format="%.4f")
            date = st.date_input("Date")
            time = st.time_input("Time")
            side = st.selectbox("Side of Road", ["R", "L"])
            
        with col2:
            st.subheader("☁️ Weather")
            temp = st.number_input("Temperature (F)", value=70.0)
            humid = st.number_input("Humidity (%)", value=50.0)
            vis = st.number_input("Visibility (mi)", value=10.0)
            wind = st.number_input("Wind Speed (mph)", value=5.0)
            precip = st.number_input("Precipitation (in)", value=0.0)
            cond = st.selectbox("Condition", ["Clear", "Cloudy", "Rain", "Snow", "Fog", "Thunderstorm"])
            
        with col3:
            st.subheader("🚦 Road Features")
            c1, c2 = st.columns(2)
            crossing = c1.checkbox("Crossing")
            junction = c1.checkbox("Junction")
            signal = c1.checkbox("Traffic Signal")
            stop = c1.checkbox("Stop Sign")
            station = c2.checkbox("Station")
            railway = c2.checkbox("Railway")
            amenity = c2.checkbox("Amenity")
            sunrise = st.selectbox("Day/Night", ["Day", "Night"])

        submit = st.form_submit_button("Predict Severity")
        
        if submit:
            payload = {
                "Start_Time": f"{date} {time}",
                "Start_Lat": lat,
                "Start_Lng": lng,
                "Temperature_F": temp,
                "Humidity_Percent": humid,
                "Pressure_in": 29.92, # Default
                "Visibility_mi": vis,
                "Wind_Speed_mph": wind,
                "Precipitation_in": precip,
                "Weather_Condition": cond,
                "Amenity": amenity,
                "Bump": False,
                "Crossing": crossing,
                "Give_Way": False,
                "Junction": junction,
                "No_Exit": False,
                "Railway": railway,
                "Roundabout": False,
                "Station": station,
                "Stop": stop,
                "Traffic_Calming": False,
                "Traffic_Signal": signal,
                "Turning_Loop": False,
                "Sunrise_Sunset": sunrise,
                "Wind_Direction": "Calm",
                "Side": side
            }
            
            with st.spinner("Analyzing..."):
                try:
                    res = requests.post(f"{API_URL}/predict", json=payload)
                    if res.status_code == 200:
                        pred = res.json()["predictions"][0]
                        
                        sev_map = {
                            1: ("Low Impact", "sev-1"),
                            2: ("Minor Impact", "sev-2"),
                            3: ("Major Impact", "sev-3"),
                            4: ("Severe Impact", "sev-4")
                        }
                        # Handle 0-indexed models just in case (map 0->1, etc if needed, but usually 1-4)
                        # Assumed output is 1-4. If 0-3, we might need mapping.
                        # For now assume model output matches dataset labels (int).
                        
                        label, css = sev_map.get(pred, (f"Level {pred}", "sev-2"))
                        
                        st.markdown(f"""
                        <div class="prediction-card {css}">
                            Severity Level: {pred}<br>
                            <span style="font-size: 0.8em">{label}</span>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.error(res.text)
                except Exception as e:
                    st.error(f"Error: {e}")

elif page == "📂 Batch Analysis":
    st.title("Batch Prediction")
    f = st.file_uploader("Upload US Accidents CSV", type="csv")
    if f:
        if st.button("Process"):
            with st.spinner("Processing..."):
                files = {"file": (f.name, f, "text/csv")}
                res = requests.post(f"{API_URL}/predictCSV", files=files)
                if res.status_code == 200:
                    st.success("Done!")
                    st.download_button("Download Predictions", res.content, "predictions.csv")
                else:
                    st.error("Failed")