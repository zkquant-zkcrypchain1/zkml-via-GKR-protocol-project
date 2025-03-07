import numpy as np
import pandas as pd
from typing import Tuple, Optional, Union, List, Dict
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import json
import pickle
from pathlib import Path
import logging
from datetime import datetime

class DataHandler:
    """
    Handles data preprocessing, loading, and saving operations for ZKML models.
    """
    
    def __init__(self, data_dir: Optional[str] = "./data"):
        """
        Initialize DataHandler with data directory.
        
        Args:
            data_dir (str): Directory for storing/loading data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Initialize scalers
        self.feature_scaler = StandardScaler()
        self.target_scaler = MinMaxScaler()
        
        # Store data statistics
        self.data_stats = {}

    def setup_logging(self) -> None:
        """Configure logging for the data handler"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.data_dir / 'data_processing.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('DataHandler')

    def load_data(self, 
                 file_path: Union[str, Path], 
                 target_column: str,
                 feature_columns: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and preprocess data from file.
        
        Args:
            file_path (Union[str, Path]): Path to data file
            target_column (str): Name of target column
            feature_columns (Optional[List[str]]): List of feature columns to use
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Features (X) and target (y) arrays
        """
        try:
            file_path = Path(file_path)
            self.logger.info(f"Loading data from {file_path}")
            
            if file_path.suffix == '.csv':
                df = pd.read_csv(file_path)
            elif file_path.suffix in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
            
            # Select features
            if feature_columns is None:
                feature_columns = [col for col in df.columns if col != target_column]
            
            X = df[feature_columns].values
            y = df[target_column].values
            
            # Store data statistics
            self.data_stats = {
                'n_samples': len(df),
                'n_features': len(feature_columns),
                'feature_names': feature_columns,
                'target_name': target_column,
                'timestamp': datetime.now().isoformat()
            }
            
            return X, y
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise

    def preprocess_data(self, 
                       X: np.ndarray, 
                       y: np.ndarray,
                       test_size: float = 0.2,
                       random_state: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Preprocess data and split into train/test sets.
        
        Args:
            X (np.ndarray): Feature array
            y (np.ndarray): Target array
            test_size (float): Proportion of test set
            random_state (Optional[int]): Random seed for splitting
            
        Returns:
            Dict[str, np.ndarray]: Dictionary containing split and scaled datasets
        """
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            # Scale features
            X_train_scaled = self.feature_scaler.fit_transform(X_train)
            X_test_scaled = self.feature_scaler.transform(X_test)
            
            # Scale target if needed
            y_train_reshaped = y_train.reshape(-1, 1)
            y_test_reshaped = y_test.reshape(-1, 1)
            
            y_train_scaled = self.target_scaler.fit_transform(y_train_reshaped).squeeze()
            y_test_scaled = self.target_scaler.transform(y_test_reshaped).squeeze()
            
            # Update data statistics
            self.data_stats.update({
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'feature_mean': self.feature_scaler.mean_.tolist(),
                'feature_scale': self.feature_scaler.scale_.tolist()
            })
            
            return {
                'X_train': X_train_scaled,
                'X_test': X_test_scaled,
                'y_train': y_train_scaled,
                'y_test': y_test_scaled,
                'X_train_raw': X_train,
                'X_test_raw': X_test,
                'y_train_raw': y_train,
                'y_test_raw': y_test
            }
            
        except Exception as e:
            self.logger.error(f"Error preprocessing data: {str(e)}")
            raise

    def save_processed_data(self, 
                          data_dict: Dict[str, np.ndarray], 
                          filename: str) -> None:
        """
        Save processed data and preprocessing parameters.
        
        Args:
            data_dict (Dict[str, np.ndarray]): Dictionary of processed data
            filename (str): Base filename for saving
        """
        try:
            save_path = self.data_dir / filename
            
            # Save numpy arrays
            np.savez(
                save_path.with_suffix('.npz'),
                **{k: v for k, v in data_dict.items()}
            )
            
            # Save scalers
            with open(save_path.with_suffix('.pkl'), 'wb') as f:
                pickle.dump({
                    'feature_scaler': self.feature_scaler,
                    'target_scaler': self.target_scaler
                }, f)
            
            # Save statistics
            with open(save_path.with_suffix('.json'), 'w') as f:
                json.dump(self.data_stats, f, indent=4)
                
            self.logger.info(f"Saved processed data to {save_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving processed data: {str(e)}")
            raise

    def load_processed_data(self, filename: str) -> Dict[str, np.ndarray]:
        """
        Load processed data and preprocessing parameters.
        
        Args:
            filename (str): Base filename for loading
            
        Returns:
            Dict[str, np.ndarray]: Dictionary containing loaded data
        """
        try:
            load_path = self.data_dir / filename
            
            # Load numpy arrays
            data_dict = dict(np.load(load_path.with_suffix('.npz')))
            
            # Load scalers
            with open(load_path.with_suffix('.pkl'), 'rb') as f:
                scalers = pickle.load(f)
                self.feature_scaler = scalers['feature_scaler']
                self.target_scaler = scalers['target_scaler']
            
            # Load statistics
            with open(load_path.with_suffix('.json'), 'r') as f:
                self.data_stats = json.load(f)
                
            self.logger.info(f"Loaded processed data from {load_path}")
            
            return data_dict
            
        except Exception as e:
            self.logger.error(f"Error loading processed data: {str(e)}")
            raise

    def generate_synthetic_data(self, 
                              n_samples: int = 1000, 
                              n_features: int = 1, 
                              noise: float = 0.1, 
                              random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic data for testing.
        
        Args:
            n_samples (int): Number of samples to generate
            n_features (int): Number of features
            noise (float): Noise level
            random_state (Optional[int]): Random seed
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Features (X) and target (y) arrays
        """
        try:
            if random_state is not None:
                np.random.seed(random_state)
            
            # Generate features
            X = np.random.randn(n_samples, n_features)
            
            # Generate target with some non-linear relationship
            y = np.sum(2 * X + X**2, axis=1) + np.random.randn(n_samples) * noise
            
            self.data_stats = {
                'n_samples': n_samples,
                'n_features': n_features,
                'noise_level': noise,
                'data_type': 'synthetic',
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"Generated synthetic data with {n_samples} samples and {n_features} features")
            
            return X, y
            
        except Exception as e:
            self.logger.error(f"Error generating synthetic data: {str(e)}")
            raise

    def get_data_statistics(self) -> Dict:
        """
        Get statistics about the processed data.
        
        Returns:
            Dict: Dictionary containing data statistics
        """
        return self.data_stats

    def inverse_transform_predictions(self, y_pred: np.ndarray) -> np.ndarray:
        """
        Transform predictions back to original scale.
        
        Args:
            y_pred (np.ndarray): Scaled predictions
            
        Returns:
            np.ndarray: Predictions in original scale
        """
        try:
            y_pred_reshaped = y_pred.reshape(-1, 1)
            return self.target_scaler.inverse_transform(y_pred_reshaped).squeeze()
        except Exception as e:
            self.logger.error(f"Error inverse transforming predictions: {str(e)}")
            raise
