"""
Streamlit Web Interface for Weather Prediction
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from weather_model import WeatherPredictor

# Page configuration
st.set_page_config(
    page_title="Seattle Weather Predictor",
    page_icon="🌤️",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #E3F2FD;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1E88E5;
        margin: 10px 0;
    }
    .metric-box {
        background-color: #F5F5F5;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

def load_data():
    """Load and prepare the data"""
    df = pd.read_csv('seattle-weather.csv')
    df['date'] = pd.to_datetime(df['date'])
    return df

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🌤️ Seattle Weather Prediction</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Model Configuration")
        
        model_type = st.selectbox(
            "Select Model",
            ["Random Forest", "Decision Tree", "Logistic Regression"],
            index=0
        )
        
        st.header("📊 Data Insights")
        if st.button("Show Dataset Info"):
            df = load_data()
            st.write(f"**Dataset Shape:** {df.shape}")
            st.write(f"**Date Range:** {df['date'].min().date()} to {df['date'].max().date()}")
        
        st.header("ℹ️ About")
        st.info("""
        This app predicts weather conditions in Seattle using historical data.
        
        **Features Used:**
        - Precipitation
        - Maximum Temperature
        - Minimum Temperature
        - Wind Speed
        - Month
        - Day of Year
        """)
    
    # Main content area - Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏠 Home", 
        "🤖 Train Model", 
        "🔮 Predict", 
        "📈 Insights"
    ])
    
    with tab1:
        st.header("Welcome to Seattle Weather Predictor")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 📋 Project Overview
            This machine learning model predicts weather conditions based on:
            - Historical weather patterns
            - Temperature ranges
            - Precipitation levels
            - Wind speeds
            
            ### 🎯 Objectives
            1. Predict weather conditions (sun, rain, snow, drizzle, fog)
            2. Provide probabilistic predictions
            3. Visualize model performance
            4. Enable real-time predictions
            """)
        
        with col2:
            st.markdown("""
            ### 🛠️ Technologies Used
            - **Scikit-learn**: Machine learning models
            - **Pandas**: Data manipulation
            - **Streamlit**: Web interface
            - **Matplotlib/Seaborn**: Visualizations
            
            ### 📊 Model Types
            1. **Random Forest** (Recommended)
            2. **Decision Tree**
            3. **Logistic Regression**
            """)
        
        # Load sample data
        df = load_data()
        
        st.subheader("📅 Recent Weather Data")
        st.dataframe(df.tail(10), use_container_width=True)
    
    with tab2:
        st.header("Train Machine Learning Model")
        
        if st.button("🚀 Train Model", type="primary"):
            with st.spinner("Training model... This may take a moment."):
                # Initialize and train model
                model_type_map = {
                    "Random Forest": "random_forest",
                    "Decision Tree": "decision_tree",
                    "Logistic Regression": "logistic"
                }
                
                predictor = WeatherPredictor(
                    model_type=model_type_map[model_type]
                )
                
                # Train model
                X_train, X_test, y_train, y_test, df = predictor.load_and_preprocess_data()
                predictor.train_model(X_train, y_train)
                metrics = predictor.evaluate_model(X_test, y_test)
                
                # Save model
                predictor.save_model('weather_model.pkl')
                
                # Display results
                st.success("✅ Model trained successfully!")
                
                # Metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Model Type", model_type)
                with col2:
                    st.metric("Accuracy", f"{metrics['accuracy']:.2%}")
                with col3:
                    st.metric("Testing Samples", len(y_test))
                
                # Classification report
                st.subheader("📋 Detailed Performance")
                report_df = pd.DataFrame(metrics['report']).transpose()
                st.dataframe(report_df, use_container_width=True)
                
                # Confusion matrix visualization
                st.subheader("🎯 Confusion Matrix")
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(
                    metrics['confusion_matrix'],
                    annot=True,
                    fmt='d',
                    cmap='Blues',
                    xticklabels=predictor.label_encoder.classes_,
                    yticklabels=predictor.label_encoder.classes_,
                    ax=ax
                )
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                ax.set_title('Confusion Matrix')
                st.pyplot(fig)
        
        # Load pre-trained model
        st.subheader("📂 Load Existing Model")
        if st.button("Load Saved Model"):
            try:
                predictor = WeatherPredictor()
                predictor.load_model('weather_model.pkl')
                st.success("✅ Model loaded successfully!")
                st.session_state['predictor'] = predictor
            except:
                st.error("❌ No saved model found. Please train a model first.")
    
    with tab3:
        st.header("Make Weather Prediction")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🌡️ Input Parameters")
            
            precipitation = st.slider(
                "Precipitation (mm)",
                min_value=0.0,
                max_value=50.0,
                value=5.0,
                step=0.1
            )
            
            temp_max = st.slider(
                "Maximum Temperature (°C)",
                min_value=-10.0,
                max_value=40.0,
                value=15.0,
                step=0.1
            )
            
            temp_min = st.slider(
                "Minimum Temperature (°C)",
                min_value=-10.0,
                max_value=30.0,
                value=8.0,
                step=0.1
            )
        
        with col2:
            st.subheader("🌬️ Additional Parameters")
            
            wind = st.slider(
                "Wind Speed (km/h)",
                min_value=0.0,
                max_value=20.0,
                value=3.0,
                step=0.1
            )
            
            month = st.selectbox(
                "Month",
                range(1, 13),
                format_func=lambda x: [
                    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
                ][x-1]
            )
            
            day_of_year = st.slider(
                "Day of Year",
                min_value=1,
                max_value=365,
                value=150
            )
        
        # Prediction button
        if st.button("🔮 Predict Weather", type="primary"):
            try:
                # Load model
                if 'predictor' not in st.session_state:
                    predictor = WeatherPredictor()
                    predictor.load_model('weather_model.pkl')
                    st.session_state['predictor'] = predictor
                
                predictor = st.session_state['predictor']
                
                # Prepare features
                features = [
                    precipitation,
                    temp_max,
                    temp_min,
                    wind,
                    month,
                    day_of_year
                ]
                
                # Make prediction
                prediction, probabilities = predictor.predict_weather(features)
                
                # Display results
                st.markdown(f"""
                <div class="prediction-box">
                    <h3>🎯 Prediction Result</h3>
                    <p><strong>Predicted Weather:</strong> {prediction.upper()}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Display probabilities
                if probabilities:
                    st.subheader("📊 Prediction Probabilities")
                    
                    # Create bar chart
                    prob_df = pd.DataFrame(
                        list(probabilities.items()),
                        columns=['Weather', 'Probability']
                    )
                    prob_df = prob_df.sort_values('Probability', ascending=False)
                    
                    fig, ax = plt.subplots(figsize=(10, 4))
                    bars = ax.barh(
                        prob_df['Weather'],
                        prob_df['Probability'],
                        color=sns.color_palette("Blues_r", len(prob_df))
                    )
                    
                    # Add percentage labels
                    for bar, prob in zip(bars, prob_df['Probability']):
                        width = bar.get_width()
                        ax.text(
                            width + 0.01,
                            bar.get_y() + bar.get_height()/2,
                            f'{prob:.1%}',
                            va='center'
                        )
                    
                    ax.set_xlim(0, 1)
                    ax.set_xlabel('Probability')
                    ax.set_title('Weather Prediction Probabilities')
                    st.pyplot(fig)
                    
                    # Display in columns
                    st.subheader("🌧️ Detailed Probabilities")
                    cols = st.columns(len(probabilities))
                    for idx, (weather, prob) in enumerate(probabilities.items()):
                        with cols[idx]:
                            st.metric(
                                weather.capitalize(),
                                f"{prob:.1%}"
                            )
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.info("💡 Please train or load a model first in the 'Train Model' tab.")
    
    with tab4:
        st.header("Data Insights & Visualizations")
        
        df = load_data()
        
        # Weather distribution
        st.subheader("📊 Weather Distribution")
        weather_counts = df['weather'].value_counts()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 6))
            weather_counts.plot(kind='pie', autopct='%1.1f%%', ax=ax)
            ax.set_ylabel('')
            ax.set_title('Weather Condition Distribution')
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 6))
            weather_counts.plot(kind='bar', color='skyblue', ax=ax)
            ax.set_xlabel('Weather Type')
            ax.set_ylabel('Count')
            ax.set_title('Weather Frequency')
            ax.tick_params(axis='x', rotation=45)
            st.pyplot(fig)
        
        # Monthly patterns
        st.subheader("📅 Monthly Weather Patterns")
        df['month'] = df['date'].dt.month
        monthly_weather = pd.crosstab(df['month'], df['weather'])
        
        fig, ax = plt.subplots(figsize=(12, 6))
        monthly_weather.plot(kind='bar', stacked=True, ax=ax)
        ax.set_xlabel('Month')
        ax.set_ylabel('Count')
        ax.set_title('Weather Distribution by Month')
        ax.legend(title='Weather')
        st.pyplot(fig)
        
        # Temperature trends
        st.subheader("🌡️ Temperature Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 4))
            df.groupby('weather')['temp_max'].mean().sort_values().plot(
                kind='barh', color='lightcoral', ax=ax
            )
            ax.set_xlabel('Average Max Temperature (°C)')
            ax.set_title('Average Max Temperature by Weather')
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 4))
            df.groupby('weather')['precipitation'].mean().sort_values().plot(
                kind='barh', color='lightblue', ax=ax
            )
            ax.set_xlabel('Average Precipitation (mm)')
            ax.set_title('Average Precipitation by Weather')
            st.pyplot(fig)
        
        # Correlation heatmap
        st.subheader("🔗 Feature Correlations")
        
        # Select numerical columns for correlation
        numeric_cols = ['precipitation', 'temp_max', 'temp_min', 'wind']
        correlation_matrix = df[numeric_cols].corr()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            correlation_matrix,
            annot=True,
            cmap='coolwarm',
            center=0,
            square=True,
            ax=ax
        )
        ax.set_title('Feature Correlation Matrix')
        st.pyplot(fig)

if __name__ == "__main__":
    main()