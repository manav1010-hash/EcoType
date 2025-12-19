# ==================================================================================
# STREAMLIT APP: EcoType - Forest Cover Type Prediction
# ==================================================================================
# Run with: streamlit run app.py
# ==================================================================================

import streamlit as st
import numpy as np
import pandas as pd
import joblib
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px

# ==================================================================================
# PAGE CONFIGURATION
# ==================================================================================

st.set_page_config(
    page_title="EcoType - Forest Cover Prediction",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .header-title {
        font-size: 3em;
        font-weight: bold;
        color: #1f6b4a;
        text-align: center;
        margin-bottom: 10px;
    }
    .subtitle {
        font-size: 1.2em;
        color: #555;
        text-align: center;
        margin-bottom: 30px;
    }
    .prediction-box {
        background-color: #d4edda;
        border: 2px solid #1f6b4a;
        border-radius: 10px;
        padding: 20px;
        margin: 20px 0;
        font-size: 1.1em;
        font-weight: bold;
        color: #0c5b3e;
    }
    .info-box {
        background-color: #e7f3ff;
        border-left: 5px solid #2196F3;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .section-title {
        font-size: 1.8em;
        font-weight: bold;
        color: #1f6b4a;
        margin-top: 30px;
        margin-bottom: 15px;
    }
    </style>
    """, unsafe_allow_html=True)

# ==================================================================================
# LOAD MODEL & ARTIFACTS
# ==================================================================================

@st.cache_resource
def load_model():
    """Load the trained model and associated artifacts"""
    try:
        model_package = joblib.load('best_model_package.pkl')
        return model_package
    except FileNotFoundError:
        st.error("⚠ Model file 'best_model_package.pkl' not found!")
        st.info("Please ensure the model has been trained and saved using the ML notebook.")
        return None

# ==================================================================================
# HELPER FUNCTIONS
# ==================================================================================

def apply_log_transformation(df, columns):
    """Apply log1p transformation to numerical features"""
    df_transformed = df.copy()
    for col in columns:
        if df_transformed[col].min() >= 0:
            df_transformed[col] = np.log1p(df_transformed[col])
    return df_transformed

def prepare_input(input_data, feature_names, selected_features, scaler, skewed_features):
    """Prepare user input for model prediction"""
    # Create dataframe with same structure as training data
    df_input = pd.DataFrame([input_data], columns=feature_names)
    
    # Apply log transformation to skewed features
    if skewed_features:
        df_input = apply_log_transformation(df_input, skewed_features)
    
    # Scale the features
    df_input_scaled = scaler.transform(df_input)
    
    # Select only the features used for training
    selected_indices = [feature_names.index(f) for f in selected_features]
    df_input_selected = df_input_scaled[:, selected_indices]
    
    return df_input_selected

def get_forest_type_description(forest_type):
    """Get description for each forest cover type"""
    descriptions = {
        1: {
            'name': 'Spruce-Fir Forest',
            'description': 'Dense coniferous forest dominated by spruce and fir trees',
            'characteristics': ['High elevation', 'Cold climate', 'Dense canopy', 'Poor understory'],
            'icon': '🌲'
        },
        2: {
            'name': 'Lodgepole Pine Forest',
            'description': 'Forest dominated by lodgepole pine trees',
            'characteristics': ['Medium-high elevation', 'Moderate to dense canopy', 'Fire-adapted', 'Sparser understory'],
            'icon': '🌲'
        },
        3: {
            'name': 'Ponderosa Pine Forest',
            'description': 'Open forest with ponderosa pine as dominant species',
            'characteristics': ['Lower elevation', 'Open canopy', 'Diverse understory', 'Fire-resistant'],
            'icon': '🌳'
        },
        4: {
            'name': 'Cottonwood/Willow Forest',
            'description': 'Riparian forest near water bodies',
            'characteristics': ['Near water', 'Deciduous trees', 'High moisture', 'Riparian vegetation'],
            'icon': '🌳'
        },
        5: {
            'name': 'Aspen Forest',
            'description': 'Forest dominated by aspen trees',
            'characteristics': ['Medium elevation', 'Deciduous', 'Quick regeneration', 'Post-fire succession'],
            'icon': '🌳'
        },
        6: {
            'name': 'Douglas-fir Forest',
            'description': 'Mixed coniferous forest with Douglas fir dominance',
            'characteristics': ['Lower-medium elevation', 'Mixed species', 'Diverse structure', 'Variable canopy'],
            'icon': '🌲'
        },
        7: {
            'name': 'Krummholz/Alpine Vegetation',
            'description': 'Stunted vegetation at high elevation or harsh conditions',
            'characteristics': ['Very high elevation', 'Stunted growth', 'Harsh weather', 'Sparse coverage'],
            'icon': '🌿'
        }
    }
    return descriptions.get(forest_type, descriptions[1])

# ==================================================================================
# MAIN APP
# ==================================================================================

def main():
    # Header
    st.markdown("<div class='header-title'>🌲 EcoType</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Forest Cover Type Prediction Using Machine Learning</div>", 
                unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.markdown("## 📋 Navigation")
    page = st.sidebar.radio("Select a page:", 
                           ["🔮 Predictions", "📊 Model Information", "📚 Data Guide", "ℹ️ About"])
    
    # Load model
    model_package = load_model()
    
    if model_package is None:
        st.error("Cannot proceed without model. Please train the model first.")
        return
    
    # Extract model artifacts
    model = model_package['model']
    scaler = model_package['scaler']
    feature_names = model_package['feature_names']
    selected_features = model_package['selected_features']
    class_names = model_package['class_names']
    numerical_cols = model_package['numerical_cols']
    
    # Identify skewed features for transformation
    skewed_features = ['Horizontal_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways', 
                      'Horizontal_Distance_To_Fire_Points']
    
    # ==================================================================================
    # PAGE: PREDICTIONS
    # ==================================================================================
    
    if page == "🔮 Predictions":
        st.markdown("<div class='section-title'>Make a Prediction</div>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class='info-box'>
        <strong>ℹ️ Instructions:</strong> Enter the geographical and environmental parameters 
        of a forest area to predict its cover type.
        </div>
        """, unsafe_allow_html=True)
        
        # Create tabs for input organization
        tab1, tab2 = st.tabs(["🏔️ Geographical Features", "🌍 Area Classification"])
        
        with tab1:
            st.markdown("#### Topographical Features")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                elevation = st.number_input(
                    "Elevation (meters)",
                    min_value=1000,
                    max_value=4000,
                    value=2500,
                    step=50,
                    help="Height above sea level in meters"
                )
            
            with col2:
                aspect = st.number_input(
                    "Aspect (degrees)",
                    min_value=0,
                    max_value=360,
                    value=180,
                    step=10,
                    help="Slope direction (0-360°, 0=North)"
                )
            
            with col3:
                slope = st.number_input(
                    "Slope (degrees)",
                    min_value=0,
                    max_value=90,
                    value=15,
                    step=1,
                    help="Slope steepness in degrees"
                )
            
            st.markdown("#### Proximity Features")
            col4, col5, col6 = st.columns(3)
            
            with col4:
                h_hydrology = st.number_input(
                    "Horizontal Distance to Hydrology (meters)",
                    min_value=0,
                    max_value=3500,
                    value=500,
                    step=50,
                    help="Distance to nearest water body"
                )
            
            with col5:
                v_hydrology = st.number_input(
                    "Vertical Distance to Hydrology (meters)",
                    min_value=-200,
                    max_value=600,
                    value=100,
                    step=10,
                    help="Elevation difference to nearest water"
                )
            
            with col6:
                h_roadways = st.number_input(
                    "Horizontal Distance to Roadways (meters)",
                    min_value=0,
                    max_value=7500,
                    value=1000,
                    step=100,
                    help="Distance to nearest road"
                )
            
            st.markdown("#### Lighting & Fire Features")
            col7, col8, col9 = st.columns(3)
            
            with col7:
                hillshade_9am = st.number_input(
                    "Hillshade at 9:00 AM",
                    min_value=0,
                    max_value=255,
                    value=128,
                    step=5,
                    help="Illumination index at 9 AM"
                )
            
            with col8:
                hillshade_noon = st.number_input(
                    "Hillshade at Noon",
                    min_value=0,
                    max_value=255,
                    value=175,
                    step=5,
                    help="Illumination index at noon"
                )
            
            with col9:
                hillshade_3pm = st.number_input(
                    "Hillshade at 3:00 PM",
                    min_value=0,
                    max_value=255,
                    value=120,
                    step=5,
                    help="Illumination index at 3 PM"
                )
            
            col10, col11 = st.columns(2)
            with col10:
                h_fire = st.number_input(
                    "Distance to Fire Points (meters)",
                    min_value=0,
                    max_value=7000,
                    value=1500,
                    step=100,
                    help="Distance to nearest wildfire point"
                )
        
        with tab2:
            st.markdown("#### Wilderness Area Classification")
            wilderness_options = {
                'Rawah Wilderness Area': 0,
                'Neota Wilderness Area': 1,
                'Comanche Peak Wilderness Area': 2,
                'Cache la Poudre Wilderness Area': 3
            }
            
            selected_wilderness = st.selectbox(
                "Select Wilderness Area",
                list(wilderness_options.keys())
            )
            wilderness_area = wilderness_options[selected_wilderness]
            
            st.markdown("#### Soil Type Selection")
            soil_type = st.slider(
                "Soil Type (1-40)",
                min_value=1,
                max_value=40,
                value=20,
                step=1,
                help="Forest soil type classification"
            )
        
        # Prepare input for prediction
        input_dict = {}
        
        # Numerical features
        input_dict['Elevation'] = elevation
        input_dict['Aspect'] = aspect
        input_dict['Slope'] = slope
        input_dict['Horizontal_Distance_To_Hydrology'] = h_hydrology
        input_dict['Vertical_Distance_To_Hydrology'] = v_hydrology
        input_dict['Horizontal_Distance_To_Roadways'] = h_roadways
        input_dict['Hillshade_9am'] = hillshade_9am
        input_dict['Hillshade_Noon'] = hillshade_noon
        input_dict['Hillshade_3pm'] = hillshade_3pm
        input_dict['Horizontal_Distance_To_Fire_Points'] = h_fire
        
        # Wilderness area one-hot encoding
        for i in range(4):
            input_dict[f'Wilderness_Area_{i+1}'] = 1 if i == wilderness_area else 0
        
        # Soil type one-hot encoding
        for i in range(40):
            input_dict[f'Soil_Type_{i+1}'] = 1 if i == (soil_type - 1) else 0
        
        # Derived features
        input_dict['Elevation_Slope_Interaction'] = elevation * slope
        input_dict['Distance_to_Hydrology'] = np.sqrt(h_hydrology**2 + v_hydrology**2)
        input_dict['Hillshade_Variance'] = np.var([hillshade_9am, hillshade_noon, hillshade_3pm])
        input_dict['Average_Hillshade'] = np.mean([hillshade_9am, hillshade_noon, hillshade_3pm])
        
        # Make prediction
        st.markdown("<div class='section-title'>Prediction Result</div>", unsafe_allow_html=True)
        
        col_pred1, col_pred2 = st.columns([2, 1])
        
        with col_pred1:
            if st.button("🔮 Predict Forest Cover Type", key="predict_button", use_container_width=True):
                try:
                    # Prepare input
                    X_input = prepare_input(
                        list(input_dict.values()),
                        feature_names,
                        selected_features,
                        scaler,
                        skewed_features
                    )
                    
                    # Make prediction
                    prediction = model.predict(X_input)[0]
                    forest_type = prediction + 1
                    
                    # Get probability estimates if available
                    try:
                        probabilities = model.predict_proba(X_input)[0]
                        confidence = np.max(probabilities) * 100
                    except:
                        confidence = None
                    
                    # Display prediction
                    forest_info = get_forest_type_description(forest_type)
                    
                    st.markdown(f"""
                    <div class='prediction-box'>
                    {forest_info['icon']} <strong>{forest_info['name']}</strong> (Type {forest_type})
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"**Description:** {forest_info['description']}")
                    
                    if confidence:
                        st.markdown(f"**Model Confidence:** {confidence:.1f}%")
                    
                    st.markdown("**Key Characteristics:**")
                    for char in forest_info['characteristics']:
                        st.write(f"• {char}")
                    
                    # Show probability distribution
                    if confidence is not None:
                        col_chart1, col_chart2 = st.columns(2)
                        
                        with col_chart1:
                            st.markdown("**Prediction Probabilities:**")
                            prob_df = pd.DataFrame({
                                'Forest Type': range(1, 8),
                                'Probability': probabilities
                            })
                            fig = px.bar(prob_df, x='Forest Type', y='Probability',
                                        labels={'Probability': 'Probability (%)'})
                            fig.update_yaxes(range=[0, 1])
                            st.plotly_chart(fig, use_container_width=True)
                
                except Exception as e:
                    st.error(f"❌ Error making prediction: {str(e)}")
    
    # ==================================================================================
    # PAGE: MODEL INFORMATION
    # ==================================================================================
    
    elif page == "📊 Model Information":
        st.markdown("<div class='section-title'>Model Details</div>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Model Architecture")
            st.info(f"""
            **Model Type:** {type(model).__name__}
            
            **Number of Features:** {len(selected_features)}
            
            **Number of Classes:** 7
            
            **Random State:** 42 (reproducible)
            """)
        
        with col2:
            st.markdown("#### Feature Information")
            st.info(f"""
            **Total Features Engineered:** {len(feature_names)}
            
            **Features Selected:** {len(selected_features)}
            
            **Numerical Features:** {len(numerical_cols)}
            
            **Categorical Features:** 44 (one-hot encoded)
            """)
        
        st.markdown("#### Selected Features for Model")
        features_df = pd.DataFrame({'Selected Features': selected_features})
        st.dataframe(features_df, use_container_width=True)
        
        st.markdown("#### Forest Cover Types")
        forest_types_data = {
            'Type': list(range(1, 8)),
            'Name': [
                'Spruce-Fir Forest',
                'Lodgepole Pine Forest',
                'Ponderosa Pine Forest',
                'Cottonwood/Willow Forest',
                'Aspen Forest',
                'Douglas-fir Forest',
                'Krummholz/Alpine'
            ]
        }
        forest_types_df = pd.DataFrame(forest_types_data)
        st.dataframe(forest_types_df, use_container_width=True)
    
    # ==================================================================================
    # PAGE: DATA GUIDE
    # ==================================================================================
    
    elif page == "📚 Data Guide":
        st.markdown("<div class='section-title'>Feature Descriptions</div>", unsafe_allow_html=True)
        
        st.markdown("### 🏔️ Topographical Features")
        st.markdown("""
        - **Elevation:** Height above sea level (meters). Range: 1,000 - 4,000m
        - **Aspect:** Slope direction/azimuth (degrees). Range: 0-360°
        - **Slope:** Steepness of terrain (degrees). Range: 0-90°
        """)
        
        st.markdown("### 💧 Hydrological Features")
        st.markdown("""
        - **Horizontal Distance to Hydrology:** Direct distance to nearest water body (meters)
        - **Vertical Distance to Hydrology:** Elevation difference to nearest water (meters)
        - **Distance to Hydrology (derived):** Euclidean distance combining horizontal and vertical
        """)
        
        st.markdown("### 🛣️ Infrastructure Features")
        st.markdown("""
        - **Horizontal Distance to Roadways:** Distance to nearest road access point (meters)
        - **Horizontal Distance to Fire Points:** Distance to known wildfire ignition points (meters)
        """)
        
        st.markdown("### ☀️ Illumination Features (Hillshade)")
        st.markdown("""
        - **Hillshade at 9:00 AM:** Solar illumination index in morning
        - **Hillshade at Noon:** Solar illumination index at midday
        - **Hillshade at 3:00 PM:** Solar illumination index in afternoon
        - **Average Hillshade (derived):** Mean illumination across three time points
        - **Hillshade Variance (derived):** Variation in illumination throughout day
        """)
        
        st.markdown("### 🌍 Geographical Classification")
        st.markdown("""
        - **Wilderness Area:** One of four protected areas (one-hot encoded)
          - Rawah, Neota, Comanche Peak, Cache la Poudre
        - **Soil Type:** One of 40 soil classifications (one-hot encoded)
        """)
    
    # ==================================================================================
    # PAGE: ABOUT
    # ==================================================================================
    
    elif page == "ℹ️ About":
        st.markdown("<div class='section-title'>About EcoType</div>", unsafe_allow_html=True)
        
        st.markdown("""
        ### 🌿 Project Overview
        
        **EcoType** is a machine learning application designed to predict forest cover types 
        based on geographical and environmental features. The system assists in environmental 
        monitoring, forestry management, and land use planning.
        
        ### 📊 Dataset
        - **Source:** UCI Forest Cover Type Dataset
        - **Samples:** 145,891 forest areas
        - **Features:** 54 (after encoding)
        - **Target Classes:** 7 forest cover types
        
        ### 🎯 Real-World Applications
        - **Forest Resource Management:** Identify and classify forest areas for planning
        - **Wildfire Risk Assessment:** Combine with fire risk models to predict high-risk zones
        - **Land Cover Mapping:** Monitor land usage patterns for environmental scientists
        - **Ecological Research:** Support biodiversity and habitat analysis studies
        
        ### 🔧 Technical Stack
        - **Languages:** Python
        - **Libraries:** Scikit-learn, XGBoost, Pandas, NumPy
        - **Frontend:** Streamlit
        - **Model Selection:** Ensemble & Gradient Boosting methods
        - **Data Processing:** Feature engineering, scaling, SMOTE for imbalance
        
        ### ✨ Key Features
        - Interactive user interface for predictions
        - Real-time prediction with confidence scores
        - Comprehensive model information and feature descriptions
        - Data-driven insights for forest management
        
        ### 📈 Model Performance
        The model was trained on a balanced dataset using multiple classification algorithms:
        - Random Forest
        - XGBoost
        - Decision Tree
        - Logistic Regression
        - K-Nearest Neighbors
        
        The best-performing model was selected based on accuracy, precision, recall, and F1-score.
        
        ---
        
        **Version:** 1.0  
        **Last Updated:** December 2024  
        **Status:** Production Ready ✓
        """)

if __name__ == "__main__":
    main()