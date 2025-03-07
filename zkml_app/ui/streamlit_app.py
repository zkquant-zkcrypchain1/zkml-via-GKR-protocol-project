import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import json
from typing import Tuple, Dict

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from src import ZKMLModelGKR, DataHandler

class ZKMLApp:
    def __init__(self):
        """Initialize the Streamlit ZKML application"""
        st.set_page_config(
            page_title="ZKML Demo",
            page_icon="🔒",
            layout="wide"
        )
        
        self.data_handler = DataHandler()
        self.model = ZKMLModelGKR()
        
        # Initialize session state
        if 'trained_model' not in st.session_state:
            st.session_state.trained_model = False
        if 'proof' not in st.session_state:
            st.session_state.proof = None
        if 'data' not in st.session_state:
            st.session_state.data = None

    def render_sidebar(self) -> None:
        """Render the sidebar with configuration options"""
        st.sidebar.title("ZKML Configuration")
        
        data_option = st.sidebar.selectbox(
            "Select Data Source",
            ["Upload Data", "Generate Synthetic Data"]
        )
        
        if data_option == "Generate Synthetic Data":
            n_samples = st.sidebar.slider("Number of Samples", 100, 1000, 500)
            noise = st.sidebar.slider("Noise Level", 0.0, 1.0, 0.1)
            
            if st.sidebar.button("Generate Data"):
                X, y = self.data_handler.generate_synthetic_data(
                    n_samples=n_samples,
                    noise=noise
                )
                st.session_state.data = self.data_handler.preprocess_data(X, y)
                st.session_state.trained_model = False
                
        else:
            uploaded_file = st.sidebar.file_uploader(
                "Upload CSV file",
                type=['csv']
            )
            
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                target_col = st.sidebar.selectbox(
                    "Select Target Column",
                    df.columns.tolist()
                )
                
                if st.sidebar.button("Process Data"):
                    X = df.drop(columns=[target_col]).values
                    y = df[target_col].values
                    st.session_state.data = self.data_handler.preprocess_data(X, y)
                    st.session_state.trained_model = False

    def plot_data(self, X: np.ndarray, y: np.ndarray, title: str) -> None:
        """Create scatter plot of data"""
        if X.shape[1] == 1:
            fig = px.scatter(
                x=X.squeeze(),
                y=y,
                title=title,
                labels={'x': 'Feature', 'y': 'Target'}
            )
            st.plotly_chart(fig)
        else:
            st.warning("Data visualization is only available for 1D features")

    def plot_predictions(self, X: np.ndarray, y_true: np.ndarray, 
                        y_pred: np.ndarray, title: str) -> None:
        """Create scatter plot comparing true values and predictions"""
        if X.shape[1] == 1:
            fig = go.Figure()
            
            # Plot true values
            fig.add_trace(go.Scatter(
                x=X.squeeze(),
                y=y_true,
                mode='markers',
                name='True Values',
                marker=dict(color='blue')
            ))
            
            # Plot predictions
            fig.add_trace(go.Scatter(
                x=X.squeeze(),
                y=y_pred,
                mode='markers',
                name='Predictions',
                marker=dict(color='red')
            ))
            
            fig.update_layout(
                title=title,
                xaxis_title='Feature',
                yaxis_title='Target'
            )
            
            st.plotly_chart(fig)
        else:
            st.warning("Prediction visualization is only available for 1D features")

    def display_proof_details(self, proof: Dict) -> None:
        """Display the details of the zero-knowledge proof"""
        st.subheader("Zero-Knowledge Proof Details")
        
        # Create expandable sections for proof components
        with st.expander("Layer Commitments"):
            for layer_id, commitment in proof['layer_commitments'].items():
                st.write(f"Layer {layer_id}:")
                st.json(commitment)
        
        with st.expander("Transition Verifications"):
            for verification in proof['transition_verifications']:
                st.write(f"Transition from Layer {verification['layer1']} "
                        f"to Layer {verification['layer2']}:")
                st.write(f"Valid: {verification['valid']}")
        
        with st.expander("Sumcheck Proofs"):
            for layer_id, sumcheck in proof['sumcheck_proofs'].items():
                st.write(f"Layer {layer_id} Sumcheck:")
                st.write(f"Linear Combination: {sumcheck['linear_combination']:.4f}")

    def render_main_content(self) -> None:
        """Render the main content area"""
        st.title("Zero-Knowledge Machine Learning Demo")
        
        if st.session_state.data is not None:
            data = st.session_state.data
            
            # Display data visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Training Data")
                self.plot_data(
                    data['X_train'], 
                    data['y_train'], 
                    "Training Data Distribution"
                )
                
            with col2:
                st.subheader("Test Data")
                self.plot_data(
                    data['X_test'], 
                    data['y_test'], 
                    "Test Data Distribution"
                )
            
            # Model training section
            if not st.session_state.trained_model:
                if st.button("Train Model"):
                    with st.spinner("Training model and generating proof..."):
                        st.session_state.proof = self.model.train(
                            data['X_train'],
                            data['y_train']
                        )
                        st.session_state.trained_model = True
            
            # Model predictions and proof verification
            if st.session_state.trained_model:
                st.subheader("Model Predictions")
                
                # Make predictions
                train_predictions = self.model.predict(data['X_train'])
                test_predictions = self.model.predict(data['X_test'])
                
                # Display predictions
                col3, col4 = st.columns(2)
                
                with col3:
                    self.plot_predictions(
                        data['X_train'],
                        data['y_train'],
                        train_predictions,
                        "Training Predictions"
                    )
                    
                with col4:
                    self.plot_predictions(
                        data['X_test'],
                        data['y_test'],
                        test_predictions,
                        "Test Predictions"
                    )
                
                # Display proof details
                if st.session_state.proof is not None:
                    self.display_proof_details(st.session_state.proof)
        
        else:
            st.info("Please select a data source from the sidebar to begin.")

    def run(self) -> None:
        """Run the Streamlit application"""
        self.render_sidebar()
        self.render_main_content()

if __name__ == "__main__":
    app = ZKMLApp()
    app.run()
