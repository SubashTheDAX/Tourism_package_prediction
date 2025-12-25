"""
Wellness Tourism Package Prediction App
Production-grade Streamlit application for predicting customer purchase likelihood
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from huggingface_hub import hf_hub_download
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os

# Page configuration
st.set_page_config(
    page_title="Wellness Tourism Predictor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 2rem 0;
    }
    .prediction-positive {
        background-color: #d4edda;
        color: #155724;
        border: 2px solid #c3e6cb;
    }
    .prediction-negative {
        background-color: #f8d7da;
        color: #721c24;
        border: 2px solid #f5c6cb;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .stDownloadButton button {
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """
    Load the trained model from Hugging Face Hub
    Uses caching to avoid reloading on every interaction
    """
    try:
        model_path = hf_hub_download(
            repo_id="TheHumanAgent/tour_pkg_pred_model",
            filename="final_tour_pkg_pred_model_v1.joblib",
            repo_type="model"
        )
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.error("Please ensure the model is uploaded to Hugging Face Hub")
        st.stop()

def create_input_features():
    """
    Create input form for all features required by the model
    Returns a dictionary with user inputs based on actual data ranges
    """
    st.sidebar.header("📋 Customer Information")
    
    # Initialize session state for form
    if 'prediction_made' not in st.session_state:
        st.session_state.prediction_made = False
    
    with st.sidebar:
        st.subheader("👤 Personal Details")
        
        # Age: Range from 18-61 based on data
        age = st.slider("Age", 
                       min_value=18, 
                       max_value=61, 
                       value=36,  # median
                       help="Customer's age (18-61 years)")
        
        # Gender: Male, Female, Fe Male (as seen in data)
        gender = st.selectbox("Gender", 
                             ["Female", "Male", "Fe Male"],
                             help="Customer's gender")
        
        # MaritalStatus: Single, Married, Divorced, Unmarried
        marital_status = st.selectbox("Marital Status", 
                                     ["Single", "Divorced", "Married", "Unmarried"],
                                     help="Customer's marital status")
        
        # CityTier: 1, 2, 3
        city_tier = st.selectbox("City Tier", 
                                [1, 2, 3],
                                index=0,  # median is 1
                                help="City development level (1=Most developed, 3=Least developed)")
        
        st.markdown("---")
        st.subheader("💼 Professional Details")
        
        # Occupation: Salaried, Small Business, Large Business, Free Lancer
        occupation = st.selectbox("Occupation", 
                                 ["Salaried", "Free Lancer", "Small Business", "Large Business"],
                                 help="Customer's occupation type")
        
        # Designation: Executive, Manager, Senior Manager, AVP, VP
        designation = st.selectbox("Designation",
                                  ["Manager", "Executive", "Senior Manager", "AVP", "VP"],
                                  help="Customer's job designation")
        
        # MonthlyIncome: Range from 1000 to 98678
        monthly_income = st.number_input("Monthly Income (₹)", 
                                        min_value=1000, 
                                        max_value=100000, 
                                        value=22418,  # median
                                        step=1000,
                                        help="Gross monthly income in Rupees (₹1,000 - ₹98,678)")
        
        st.markdown("---")
        st.subheader("✈️ Travel Preferences")
        
        # NumberOfTrips: Range from 1-22
        num_trips = st.slider("Number of Trips (Annually)", 
                             min_value=1, 
                             max_value=22, 
                             value=3,  # median
                             help="Average annual trips taken (1-22)")
        
        # Passport: 0 or 1
        passport = st.selectbox("Valid Passport", 
                               [0, 1],
                               format_func=lambda x: "Yes" if x == 1 else "No",
                               index=0,  # median is 0
                               help="Does customer have a valid passport?")
        
        # OwnCar: 0 or 1
        own_car = st.selectbox("Own Car", 
                              [0, 1],
                              format_func=lambda x: "Yes" if x == 1 else "No",
                              index=1,  # median is 1
                              help="Does customer own a car?")
        
        # PreferredPropertyStar: 3, 4, 5
        preferred_property_star = st.selectbox("Preferred Hotel Rating", 
                                              [3, 4, 5],
                                              index=0,  # median is 3
                                              help="Preferred hotel star rating (3-5 stars)")
        
        st.markdown("---")
        st.subheader("👨‍👩‍👧‍👦 Trip Details")
        
        # NumberOfPersonVisiting: Range from 1-5
        num_persons = st.slider("Number of Persons Visiting", 
                               min_value=1, 
                               max_value=5, 
                               value=3,  # median
                               help="Total people in the group (1-5)")
        
        # NumberOfChildrenVisiting: Range from 0-3
        num_children = st.slider("Number of Children (<5 years)", 
                                min_value=0, 
                                max_value=3, 
                                value=1,  # median
                                help="Number of children under 5 years (0-3)")
        
        st.markdown("---")
        st.subheader("📞 Interaction Details")
        
        # TypeofContact: Company Invited, Self Enquiry
        type_of_contact = st.selectbox("Type of Contact", 
                                      ["Self Enquiry", "Company Invited"],
                                      help="How was the customer contacted?")
        
        # ProductPitched: Basic, Standard, Deluxe, Super Deluxe, King
        product_pitched = st.selectbox("Product Pitched",
                                      ["Deluxe", "Basic", "Standard", "Super Deluxe", "King"],
                                      help="Type of package pitched to the customer")
        
        # DurationOfPitch: Range from 5-127 minutes
        duration_of_pitch = st.slider("Duration of Pitch (minutes)", 
                                     min_value=5, 
                                     max_value=127, 
                                     value=14,  # median
                                     help="Sales pitch duration in minutes (5-127)")
        
        # NumberOfFollowups: Range from 1-6
        num_followups = st.slider("Number of Follow-ups", 
                                 min_value=1, 
                                 max_value=6, 
                                 value=4,  # median
                                 help="Total follow-ups after initial pitch (1-6)")
        
        # PitchSatisfactionScore: Range from 1-5
        pitch_satisfaction = st.slider("Pitch Satisfaction Score", 
                                      min_value=1, 
                                      max_value=5, 
                                      value=3,  # median
                                      help="Customer satisfaction with the pitch (1=Very Low, 5=Very High)")
    
    # Create feature dictionary matching exact column names from training data
    features = {
        'Age': age,
        'CityTier': city_tier,
        'DurationOfPitch': duration_of_pitch,
        'NumberOfPersonVisiting': num_persons,
        'NumberOfFollowups': num_followups,
        'PreferredPropertyStar': preferred_property_star,
        'NumberOfTrips': num_trips,
        'Passport': passport,
        'PitchSatisfactionScore': pitch_satisfaction,
        'NumberOfChildrenVisiting': num_children,
        'MonthlyIncome': monthly_income,
        'TypeofContact': type_of_contact,
        'Occupation': occupation,
        'Gender': gender,
        'OwnCar': own_car,
        'ProductPitched': product_pitched,
        'MaritalStatus': marital_status,
        'Designation': designation
    }
    
    return features

def create_gauge_chart(probability):
    """
    Create a gauge chart to visualize purchase probability
    """
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Purchase Probability (%)", 'font': {'size': 24}},
        delta = {'reference': 45, 'increasing': {'color': "green"}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': '#ffcccc'},
                {'range': [30, 70], 'color': '#ffffcc'},
                {'range': [70, 100], 'color': '#ccffcc'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 45
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig

def create_feature_importance_chart(features_df):
    """
    Create a bar chart showing key customer metrics
    """
    # Select key features for visualization
    key_features = {
        'Monthly Income (₹K)': features_df['MonthlyIncome'].values[0] / 1000,
        'Age': features_df['Age'].values[0],
        'Annual Trips': features_df['NumberOfTrips'].values[0],
        'Pitch Duration (min)': features_df['DurationOfPitch'].values[0],
        'Follow-ups': features_df['NumberOfFollowups'].values[0],
        'Satisfaction': features_df['PitchSatisfactionScore'].values[0],
        'Hotel Rating': features_df['PreferredPropertyStar'].values[0],
        'Group Size': features_df['NumberOfPersonVisiting'].values[0]
    }
    
    fig = px.bar(
        x=list(key_features.values()),
        y=list(key_features.keys()),
        orientation='h',
        title='Key Customer Metrics Overview',
        labels={'x': 'Value', 'y': 'Feature'},
        color=list(key_features.values()),
        color_continuous_scale='Blues'
    )
    
    fig.update_layout(
        height=400,
        showlegend=False,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig

def get_recommendation(probability, features):
    """
    Generate actionable recommendations based on prediction and customer profile
    """
    recommendations = []
    
    # Priority level based on probability
    if probability >= 0.7:
        recommendations.append("✅ **HIGH PRIORITY LEAD** - Strong purchase likelihood")
        recommendations.append("🎯 **Action**: Schedule immediate follow-up call within 24 hours")
        recommendations.append("💎 **Strategy**: Offer premium package options and exclusive benefits")
    elif probability >= 0.45:
        recommendations.append("⚠️ **MEDIUM PRIORITY LEAD** - Moderate purchase likelihood")
        recommendations.append("📧 **Action**: Send personalized email highlighting package benefits")
        recommendations.append("🎁 **Strategy**: Consider offering limited-time discount (5-10%)")
    else:
        recommendations.append("❌ **LOW PRIORITY LEAD** - Lower purchase likelihood")
        recommendations.append("📬 **Action**: Add to nurture email campaign")
        recommendations.append("🔄 **Strategy**: Re-engage after 2-3 months with seasonal offers")
    
    recommendations.append("")  # Spacing
    
    # Additional contextual recommendations based on specific features
    if features['NumberOfFollowups'] <= 2:
        recommendations.append("📌 **Insight**: Low follow-up count - Increase engagement frequency")
    
    if features['PitchSatisfactionScore'] <= 2:
        recommendations.append("⚠️ **Alert**: Low satisfaction score - Review and improve pitch approach")
    elif features['PitchSatisfactionScore'] >= 4:
        recommendations.append("⭐ **Positive**: High satisfaction - Customer is engaged, act quickly!")
    
    if features['MonthlyIncome'] >= 30000:
        recommendations.append("💰 **Insight**: High-income customer - Emphasize luxury and premium features")
    
    if features['NumberOfTrips'] >= 5:
        recommendations.append("✈️ **Insight**: Frequent traveler - Highlight loyalty benefits and travel perks")
    
    if features['Passport'] == 0:
        recommendations.append("🛂 **Note**: No passport - Consider domestic package options")
    
    if features['NumberOfChildrenVisiting'] >= 2:
        recommendations.append("👨‍👩‍👧‍👦 **Insight**: Family with children - Emphasize family-friendly amenities")
    
    if features['DurationOfPitch'] < 10:
        recommendations.append("⏱️ **Note**: Short pitch duration - May need more detailed product information")
    
    return recommendations

def display_customer_summary(features):
    """
    Display a formatted summary of customer information
    """
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("👤 Age", f"{features['Age']} years")
        st.metric("🏙️ City Tier", f"Tier {features['CityTier']}")
    
    with col2:
        st.metric("💰 Income", f"₹{features['MonthlyIncome']:,}")
        st.metric("✈️ Annual Trips", features['NumberOfTrips'])
    
    with col3:
        st.metric("📞 Follow-ups", features['NumberOfFollowups'])
        st.metric("⭐ Satisfaction", f"{features['PitchSatisfactionScore']}/5")
    
    with col4:
        st.metric("👥 Group Size", features['NumberOfPersonVisiting'])
        st.metric("🏨 Hotel Pref", f"{features['PreferredPropertyStar']} Star")

def main():
    """
    Main application function
    """
    # Header
    st.markdown('<p class="main-header">✈️ Wellness Tourism Package Predictor</p>', 
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Customer Purchase Prediction System</p>', 
                unsafe_allow_html=True)
    
    # Load model
    with st.spinner("🔄 Loading ML model..."):
        model = load_model()
    
    st.success("✅ Model loaded successfully!")
    
    # Create input form
    features = create_input_features()
    
    # Main content area
    st.markdown("---")
    st.subheader("📊 Customer Profile Summary")
    
    display_customer_summary(features)
    
    # Show detailed information in expandable section
    with st.expander("📋 View Complete Customer Details"):
        df_display = pd.DataFrame([features]).T
        df_display.columns = ['Value']
        st.dataframe(df_display, use_container_width=True, height=600)
    
    st.markdown("---")
    
    # Prediction section
    col_left, col_right = st.columns([2, 1])
    
    with col_right:
        st.subheader("🎯 Make Prediction")
        predict_button = st.button("🔮 Predict Purchase Likelihood", 
                                   type="primary", 
                                   use_container_width=True)
        
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.prediction_made = False
            st.rerun()
    
    with col_left:
        if predict_button:
            with st.spinner("🤖 Analyzing customer data..."):
                # Create DataFrame with exact feature order
                input_df = pd.DataFrame([features])
                
                # Make prediction
                try:
                    prediction_proba = model.predict_proba(input_df)[0, 1]
                    prediction = 1 if prediction_proba >= 0.45 else 0
                    
                    # Store in session state
                    st.session_state.prediction_made = True
                    st.session_state.prediction = prediction
                    st.session_state.probability = prediction_proba
                    st.session_state.features = features
                    st.session_state.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                except Exception as e:
                    st.error(f"❌ Prediction Error: {str(e)}")
                    st.error("Please check that all input values are valid.")
                    st.stop()
    
    # Display prediction results
    if st.session_state.prediction_made:
        st.markdown("---")
        st.subheader("📈 Prediction Results")
        
        prediction = st.session_state.prediction
        probability = st.session_state.probability
        
        # Prediction box with color coding
        if prediction == 1:
            st.markdown(f"""
                <div class="prediction-box prediction-positive">
                    ✅ LIKELY TO PURCHASE<br>
                    <span style="font-size: 2rem;">{probability*100:.1f}%</span><br>
                    Confidence Level
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="prediction-box prediction-negative">
                    ❌ UNLIKELY TO PURCHASE<br>
                    <span style="font-size: 2rem;">{(1-probability)*100:.1f}%</span><br>
                    Confidence Level (Not Buying)
                </div>
            """, unsafe_allow_html=True)
        
        # Visualization section
        st.markdown("---")
        st.subheader("📊 Visual Analysis")
        
        viz_col1, viz_col2 = st.columns([1, 1])
        
        with viz_col1:
            st.plotly_chart(create_gauge_chart(probability), 
                          use_container_width=True)
        
        with viz_col2:
            input_df = pd.DataFrame([st.session_state.features])
            st.plotly_chart(create_feature_importance_chart(input_df), 
                          use_container_width=True)
        
        # Recommendations section
        st.markdown("---")
        st.subheader("💡 Actionable Recommendations")
        
        recommendations = get_recommendation(probability, st.session_state.features)
        
        for rec in recommendations:
            if rec:  # Skip empty strings
                st.markdown(f"{rec}")
        
        # Model explanation
        with st.expander("🤔 How does the model work?"):
            st.markdown("""
            **Model Details:**
            - **Algorithm**: XGBoost (Extreme Gradient Boosting)
            - **Classification Threshold**: 45%
            - **Training Data**: 4,128 customer records
            - **Features**: 18 input variables including demographics, travel preferences, and interaction history
            
            **Prediction Logic:**
            - Probability ≥ 45% → Customer likely to purchase
            - Probability < 45% → Customer unlikely to purchase
            
            **Key Factors Considered:**
            - Customer demographics (age, income, occupation)
            - Travel behavior (past trips, preferences)
            - Sales interaction (pitch satisfaction, follow-ups)
            - Family situation (marital status, children)
            
            The model has been trained to identify patterns that indicate purchase likelihood based on historical customer data.
            """)
        
        # Export functionality
        st.markdown("---")
        st.subheader("📥 Export Prediction Report")
        
        report_col1, report_col2 = st.columns([2, 1])
        
        with report_col1:
            st.info("💾 Download a detailed report with all customer information and prediction results")
        
        with report_col2:
            # Create comprehensive report
            report_data = {
                'Timestamp': [st.session_state.timestamp],
                'Prediction': ['Will Purchase' if prediction == 1 else 'Will Not Purchase'],
                'Purchase_Probability': [f"{probability*100:.2f}%"],
                'Confidence_Level': ['High' if abs(probability - 0.5) > 0.2 else 'Medium'],
                **st.session_state.features
            }
            
            report_df = pd.DataFrame(report_data)
            csv = report_df.to_csv(index=False)
            
            st.download_button(
                label="📄 Download CSV Report",
                data=csv,
                file_name=f"customer_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #888; padding: 1rem;'>
            <p><b>🏢 Visit with Us</b> - Wellness Tourism Package Prediction System</p>
            <p>Powered by XGBoost ML Model | Classification Threshold: 45% | Trained on 4,128 customers</p>
            <p style='font-size: 0.85rem;'>Model Version: v1.0 | Last Updated: December 2024</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Sidebar footer with statistics
    with st.sidebar:
        st.markdown("---")
        st.info("""
        **ℹ️ About This Application**
        
        This ML-powered system predicts whether a customer will purchase 
        the Wellness Tourism Package based on their profile and interaction history.
        
        **📊 Model Statistics:**
        - **Training Data**: 4,128 customers
        - **Purchase Rate**: 19.3%
        - **Algorithm**: XGBoost Classifier
        - **Threshold**: 45%
        - **Features**: 18 variables
        
        **🎯 How to Use:**
        1. Enter customer details in the form
        2. Click 'Predict Purchase Likelihood'
        3. Review prediction and recommendations
        4. Download detailed report (optional)
        
        **📈 Prediction Accuracy:**
        The model considers demographics, travel preferences, 
        and sales interaction history to make accurate predictions.
        """)
        
        st.warning("""
        **⚠️ Important Notes:**
        - Ensure all fields are filled accurately
        - Income should be in Indian Rupees (₹)
        - Follow-ups range from 1-6
        - Pitch duration in minutes (5-127)
        """)

if __name__ == "__main__":
    main()
