# app.py
import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import altair as alt

# --- App Configuration ---
st.set_page_config(
    page_title="Trail Condition Analytics Mockup",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Load Data ---
@st.cache_data
def load_data(path):
    df = pd.read_csv(path).fillna('') # Handle empty image URLs
    # Create a mock 'Predicted_Risk' score for the demo
    # This simulates the output of the ML model
    df['Predicted_Risk'] = (df['Severity_Score'] * 1.5) + (df['Slope'] / 5) + (df['Past_7_Day_Rainfall'] / 20)
    return df

df = load_data("mock_trail_data_real.csv")

# --- Navigation Bar ---
st.markdown("""
<style>
/* Hide Streamlit header */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.stApp > header {display: none;}

.navbar {
    position: fixed;
    top: 0;
    width: 100%;
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(10px);
    z-index: 1000;
    padding: 1rem 0;
    transition: all 0.3s ease;
    border-bottom: 1px solid #e2e8f0;
}

.nav-container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 0 20px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.nav-logo h2 {
    color: #2d3748;
    font-weight: 600;
    margin: 0;
}

.nav-menu {
    display: flex;
    list-style: none;
    gap: 2rem;
    margin: 0;
    padding: 0;
}

.nav-menu li {
    margin: 0;
}

.nav-menu a {
    text-decoration: none;
    color: #4a5568;
    font-weight: 500;
    transition: color 0.3s ease;
}

.nav-menu a:hover {
    color: #3182ce;
}

.hamburger {
    display: none;
    flex-direction: column;
    cursor: pointer;
}

.hamburger span {
    width: 25px;
    height: 3px;
    background: #4a5568;
    margin: 3px 0;
    transition: 0.3s;
}

@media (max-width: 768px) {
    .nav-menu {
        display: none;
    }
    .hamburger {
        display: flex;
    }
}
</style>

<nav class="navbar">
    <div class="nav-container">
        <div class="nav-logo">
            <h2>Trail AI Project</h2>
        </div>
        <ul class="nav-menu">
            <li><a href="https://krasunalex.github.io/te-araroa-project/#vision" target="_blank">Vision</a></li>
            <li><a href="https://krasunalex.github.io/te-araroa-project/#about" target="_blank">About</a></li>
            <li><a href="https://krasunalex.github.io/te-araroa-project/#partnerships" target="_blank">Partnerships</a></li>
            <li><a href="https://krasunalex.github.io/te-araroa-project/#budget" target="_blank">Budget</a></li>
            <li><a href="https://krasunalex.github.io/te-araroa-project/#contact" target="_blank">Contact</a></li>
        </ul>
        <div class="hamburger">
            <span></span>
            <span></span>
            <span></span>
        </div>
    </div>
</nav>
""", unsafe_allow_html=True)

# Add top margin to account for fixed navbar
st.markdown('<div style="margin-top: 80px;"></div>', unsafe_allow_html=True)

# --- App Header ---
st.title("🏔️ Predictive Trail Condition Analytics: A Mockup")
st.markdown("This interactive application demonstrates the key deliverables for the Trail AI Intelligence Project. The data shown here is a synthesized sample.")

# --- Deliverable 1: The Dataset ---
st.header("Deliverable 1: The Raw & Enriched Dataset")
st.dataframe(df)
st.markdown("_This table includes a 'Predicted_Risk' column to simulate the ML model's output._")

st.markdown("---")

# --- Deliverable 2: The Interactive 'Risk Model' Map ---
st.header("Deliverable 2: Interactive 'Risk Model' Map")

def get_marker_color(severity):
    """Get color for observation markers"""
    if severity >= 4: return 'red'
    elif severity == 3: return 'orange'
    else: return 'green'

def get_line_color(risk_score):
    """Get color for trail line segments based on predicted risk"""
    if risk_score > 10: return 'red'
    elif risk_score > 6: return 'orange'
    else: return 'green'

def get_icon_for_issue(issue_category, severity):
    """Get appropriate icon for each issue type"""
    color = get_marker_color(severity)
    
    icon_mapping = {
        'Erosion': 'exclamation-triangle',  # Warning triangle for erosion
        'Hazard (Treefall)': 'tree-conifer',  # Tree icon for treefall
        'Infrastructure Damage': 'wrench',  # Wrench for infrastructure
        'Drainage Failure': 'tint',  # Water drop for drainage
        'Nominal': 'check-circle'  # Check mark for nominal conditions
    }
    
    icon = icon_mapping.get(issue_category, 'info-sign')
    return folium.Icon(color=color, icon=icon)

m = folium.Map(location=[df['Latitude'].mean(), df['Longitude'].mean()], zoom_start=8)

# Draw the trail as colored segments based on predicted risk
# We loop through the points to create segments between them
for i in range(len(df) - 1):
    start_point = (df['Latitude'][i], df['Longitude'][i])
    end_point = (df['Latitude'][i+1], df['Longitude'][i+1])
    segment_risk = df['Predicted_Risk'][i]
    
    folium.PolyLine(
        locations=[start_point, end_point],
        color=get_line_color(segment_risk),
        weight=7,
        opacity=0.8,
        tooltip=f"Predicted Risk Score: {segment_risk:.2f}"
    ).add_to(m)

# Add a marker for each data point with a photo in the popup
for idx, row in df.iterrows():
    # Handle image display for local files
    image_html = ""
    if row["Image_URL"]:
        # For local images, we'll use base64 encoding
        import base64
        import os
        
        image_path = row["Image_URL"]
        if os.path.exists(image_path):
            try:
                with open(image_path, "rb") as img_file:
                    img_data = base64.b64encode(img_file.read()).decode()
                    image_html = f'<img src="data:image/jpeg;base64,{img_data}" width="200" style="border-radius: 5px; margin-top: 5px;">'
            except:
                image_html = f'<p style="color: #666; font-style: italic;">Image: {os.path.basename(image_path)}</p>'
    
    # Add image indicator text
    image_indicator = ""
    if row["Image_URL"]:
        image_indicator = '<p style="margin: 5px 0; color: #0066cc; font-weight: bold;">📷 Photo documentation available below</p>'
    
    popup_text = f"""
    <div style="font-family: Arial, sans-serif;">
        <h4 style="margin: 0 0 10px 0; color: #333;">{row['Issue_Category']}</h4>
        <p style="margin: 5px 0;"><b>Severity Score (Observed):</b> {row['Severity_Score']}/5</p>
        <p style="margin: 5px 0;"><b>Predicted Risk Score:</b> {row['Predicted_Risk']:.2f}</p>
        <p style="margin: 5px 0;"><b>Slope:</b> {row['Slope']}°</p>
        <p style="margin: 5px 0;"><b>Elevation:</b> {row['Elevation']}m</p>
        <p style="margin: 5px 0;"><b>Aspect:</b> {row['Aspect']}</p>
        <p style="margin: 5px 0;"><b>Rain (7d):</b> {row['Past_7_Day_Rainfall']}mm</p>
        {image_indicator}
        {image_html}
    </div>
    """
    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        popup=folium.Popup(popup_text, max_width=300),
        icon=get_icon_for_issue(row['Issue_Category'], row['Severity_Score'])
    ).add_to(m)

# Create columns for map and legend
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st_folium(m, width=600, height=500)

with col2:
    st.markdown('<div style="border-left: 3px solid #cccccc; padding-left: 20px; margin-left: 10px;">', unsafe_allow_html=True)
    st.subheader("Map Legend")
    st.markdown("""
    **Trail Line (The Prediction)**: The color of the trail itself represents the AI model's *predicted risk* for that segment.
    - 🟢 **Low Risk** (Risk Score ≤ 6)
    - 🟠 **Medium Risk** (Risk Score 6-10)
    - 🔴 **High Risk** (Risk Score > 10)

    **Markers (The Observation)**: The dots represent actual issues documented on the trail.
    - 🟢 **Green Pin**: Nominal/Minor (Severity 0-2)
    - 🟠 **Orange Pin**: Requires Monitoring (Severity 3)
    - 🔴 **Red Pin**: Hazardous/Impassable (Severity 4-5)

    **Icon Types**:
    - **⚠** Erosion (Warning Triangle)
    - **🌲** Treefall Hazard (Tree Icon)
    - **🔧** Infrastructure Damage (Wrench)
    - **💧** Drainage Failure (Water Drop)
    - **✓** Nominal (Check Mark)
    """)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# --- Deliverable 3: Key Findings & Report Excerpt ---
st.header("Deliverable 3: Key Findings & Recommendations")
st.markdown("The final deliverable includes a comprehensive report with actionable insights. Below is a sample finding.")

# Create columns for content and report image
col1, col2 = st.columns([1, 1], gap="small")

with col1:
    st.subheader("Sample Finding 3.2:")
    st.markdown("""
    #### South-facing slopes are disproportionately at risk for severe erosion.
    The model's analysis reveals that **slope** and **aspect** are the two biggest predictors of severe erosion events (Severity 4-5). Over 70% of severe erosion occurs on south-facing slopes with an incline greater than 15 degrees, which are more susceptible to soil saturation.

    **Recommendation:** Proactively schedule assessments of all south-facing trail sections with a slope >15° immediately following any MetService rainfall warning.
    """)

    st.markdown("---")

    st.subheader("Sample Chart: Erosion Events by Aspect")
    chart_data = pd.DataFrame({
        'Aspect': ['North-facing', 'South-facing', 'East-facing', 'West-facing'],
        'Severe Erosion Events': [12, 41, 20, 25]
    })
    c = alt.Chart(chart_data).mark_bar().encode(
        x=alt.X('Aspect', sort=None),
        y='Severe Erosion Events',
        tooltip=['Aspect', 'Severe Erosion Events']
    ).interactive()
    st.altair_chart(c, use_container_width=True)

with col2:
    st.markdown('<div style="border-left: 3px solid #cccccc; padding-left: 20px; margin-left: 10px;">', unsafe_allow_html=True)
    st.subheader("Final Report Preview")
    
    with open("Final Report - Example.pdf", "rb") as file:
        st.download_button(
            label="📥 **Download an Example of Full Report**",
            data=file.read(),
            file_name="Final Report - Example.pdf",
            mime="application/pdf"
        )
    
    st.image("final-report-example.png", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

