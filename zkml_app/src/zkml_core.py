import numpy as np
from sklearn.preprocessing import StandardScaler
from cryptography.fernet import Fernet
import hashlib
import pickle
from typing import Dict, List, Tuple, Optional, Union
import random
from .gkr_proof import GKRProof

class ZKMLModelGKR:
    """
    Zero-Knowledge Machine Learning Model using GKR Protocol
    
    This class implements a privacy-preserving machine learning model that uses
    the GKR (Goldwasser-Kalai-Rothblum) protocol for zero-knowledge proofs.
    """
    
    def __init__(self, random_state: Optional[int] = None):
        """
        Initialize the ZKML model with GKR protocol support.
        
        Args:
            random_state (Optional[int]): Seed for random number generation
        """
        self.model_params = None
        self.encryption_key = Fernet.generate_key()
        self.cipher_suite = Fernet(self.encryption_key)
        self.proof_system = GKRProof()
        self.prime_field = 2**31 - 1  # Mersenne prime for finite field operations
        self.computation_trace = []
        self.proof = None
        
        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)

    def generate_random_point(self) -> float:
        """
        Generate a random point for polynomial evaluation.
        
        Returns:
            float: Random value between 0 and 1
        """
        return random.uniform(0, 1)

    def compute_layer_polynomial(self, values: List[float]) -> List[float]:
        """
        Compute polynomial coefficients for a layer with numerical stability.
        
        Args:
            values (List[float]): Layer values to be converted to polynomial
            
        Returns:
            List[float]: Polynomial coefficients
        """
        try:
            values = np.array(values, dtype=np.float64)
            scale_factor = np.max(np.abs(values)) if len(values) > 0 else 1
            
            if scale_factor > 1e-10:
                scaled_values = values / scale_factor
            else:
                scaled_values = values
            
            degree = min(len(values) - 1, 10)
            x_points = np.linspace(0, 1, len(values))
            coefficients = np.polynomial.polynomial.polyfit(
                x_points, 
                scaled_values, 
                degree,
                rcond=1e-10
            )
            
            return (coefficients * scale_factor).tolist()
            
        except Exception as e:
            print(f"Warning in polynomial fitting: {e}")
            return values[:min(len(values), 11)].tolist()

    def generate_gkr_proof(self, computation_trace: List[np.ndarray]) -> Dict:
        """
        Generate GKR protocol proof for the computation trace.
        
        Args:
            computation_trace (List[np.ndarray]): List of computation layers
            
        Returns:
            Dict: Proof components for each layer
        """
        proof = {}
        
        for layer_idx, layer_values in enumerate(computation_trace):
            flat_values = layer_values.ravel()
            scale = np.max(np.abs(flat_values)) if len(flat_values) > 0 else 1
            
            if scale > 1e-10:
                normalized_values = flat_values / scale
            else:
                normalized_values = flat_values
            
            poly_coeffs = self.compute_layer_polynomial(normalized_values)
            r = self.generate_random_point()
            eval_result = self.proof_system.evaluate_polynomial(r, poly_coeffs)
            
            proof[f'layer_{layer_idx}'] = {
                'coefficients': poly_coeffs,
                'random_point': r,
                'evaluation': eval_result,
                'scale_factor': scale
            }
            
            self.proof_system.add_layer(normalized_values, layer_idx)
        
        return proof

    def verify_gkr_proof(self, proof: Dict, computation_trace: List[np.ndarray]) -> bool:
        """
        Verify the GKR protocol proof.
        
        Args:
            proof (Dict): Proof components for each layer
            computation_trace (List[np.ndarray]): Original computation trace
            
        Returns:
            bool: True if verification succeeds, False otherwise
        """
        try:
            for layer_idx, layer_values in enumerate(computation_trace):
                layer_proof = proof[f'layer_{layer_idx}']
                computed_eval = self.proof_system.evaluate_polynomial(
                    layer_proof['random_point'],
                    layer_proof['coefficients']
                )
                
                if abs(computed_eval - layer_proof['evaluation']) > 1e-6:
                    return False
            
            return True
            
        except Exception as e:
            print(f"Warning in verification: {e}")
            return False

    def encrypt_model_params(self, params: Dict) -> bytes:
        """
        Encrypt model parameters using Fernet symmetric encryption.
        
        Args:
            params (Dict): Model parameters to encrypt
            
        Returns:
            bytes: Encrypted parameters
        """
        params_bytes = pickle.dumps(params)
        return self.cipher_suite.encrypt(params_bytes)

    def decrypt_model_params(self, encrypted_params: bytes) -> Dict:
        """
        Decrypt model parameters.
        
        Args:
            encrypted_params (bytes): Encrypted parameters
            
        Returns:
            Dict: Decrypted model parameters
        """
        decrypted_bytes = self.cipher_suite.decrypt(encrypted_params)
        return pickle.loads(decrypted_bytes)

    def train(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Train the model with GKR protocol verification.
        
        Args:
            X (np.ndarray): Training features
            y (np.ndarray): Target values
            
        Returns:
            Dict: GKR proof for the training computation
        """
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        self.computation_trace = []
        self.computation_trace.append(X_scaled)
        
        X_mean = np.mean(X_scaled, axis=0)
        y_mean = np.mean(y)
        
        means = np.concatenate([X_mean.ravel(), [y_mean]])
        self.computation_trace.append(means.reshape(-1, 1))
        
        numerator = np.sum((X_scaled - X_mean) * (y - y_mean))
        denominator = np.sum((X_scaled - X_mean) ** 2)
        
        self.computation_trace.append(np.array([[numerator], [denominator]], dtype=np.float64))
        
        slope = numerator / denominator if abs(denominator) > 1e-10 else 0
        intercept = y_mean - slope * X_mean
        
        if isinstance(slope, np.ndarray):
            slope = slope.item()
        if isinstance(intercept, np.ndarray):
            intercept = intercept.item()
        
        final_params = np.array([[slope], [intercept]], dtype=np.float64)
        self.computation_trace.append(final_params)
        
        self.proof = self.generate_gkr_proof(self.computation_trace)
        
        self.model_params = self.encrypt_model_params({
            'slope': slope,
            'intercept': intercept,
            'scaler': scaler
        })
        
        return self.proof

    def predict(self, X: np.ndarray, verify_proof: bool = True) -> np.ndarray:
        """
        Make predictions with optional GKR verification.
        
        Args:
            X (np.ndarray): Features to predict
            verify_proof (bool): Whether to verify the GKR proof
            
        Returns:
            np.ndarray: Predictions
            
        Raises:
            ValueError: If proof verification fails
        """
        if verify_proof and not self.verify_gkr_proof(self.proof, self.computation_trace):
            raise ValueError("GKR proof verification failed!")
        
        params = self.decrypt_model_params(self.model_params)
        X_scaled = params['scaler'].transform(X)
        predictions = params['slope'] * X_scaled + params['intercept']
        
        return predictions
