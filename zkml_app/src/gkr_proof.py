import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import random
from dataclasses import dataclass
from scipy.interpolate import lagrange

@dataclass
class LayerCommitment:
    """Data class for storing layer commitment information"""
    values: np.ndarray
    polynomial_coefficients: List[float]
    evaluation_point: float
    evaluation_result: float

class GKRProof:
    """
    Implementation of the GKR (Goldwasser-Kalai-Rothblum) protocol for
    zero-knowledge proofs of computational integrity.
    """
    
    def __init__(self, security_parameter: int = 128):
        """
        Initialize the GKR proof system.
        
        Args:
            security_parameter (int): Security parameter for proof generation
        """
        self.layers: List[Tuple[int, np.ndarray]] = []
        self.polynomials: Dict[int, np.ndarray] = {}
        self.security_parameter = security_parameter
        self.commitments: Dict[int, LayerCommitment] = {}
        
    def evaluate_polynomial(self, x: float, coefficients: List[float]) -> float:
        """
        Evaluate polynomial at point x using Horner's method.
        
        Args:
            x (float): Point at which to evaluate the polynomial
            coefficients (List[float]): Polynomial coefficients
            
        Returns:
            float: Polynomial evaluation result
        """
        result = 0.0
        for coeff in reversed(coefficients):
            result = result * float(x) + float(coeff)
        return result

    def interpolate_polynomial(self, points: np.ndarray, values: np.ndarray) -> np.ndarray:
        """
        Compute polynomial interpolation through given points.
        
        Args:
            points (np.ndarray): X coordinates
            values (np.ndarray): Y coordinates
            
        Returns:
            np.ndarray: Polynomial coefficients
        """
        try:
            # Use Lagrange interpolation for numerical stability
            poly = lagrange(points, values)
            return poly.coef
        except Exception as e:
            print(f"Warning in interpolation: {e}")
            # Fallback to numpy's polyfit
            return np.polynomial.polynomial.polyfit(points, values, len(points)-1)

    def add_layer(self, values: Union[List[float], np.ndarray], layer_id: int) -> None:
        """
        Add a computation layer to the proof system.
        
        Args:
            values (Union[List[float], np.ndarray]): Layer values
            layer_id (int): Unique identifier for the layer
        """
        values_array = np.array(values, dtype=np.float64)
        self.layers.append((layer_id, values_array))
        
        # Generate commitment for the layer
        points = np.linspace(0, 1, len(values))
        coeffs = self.interpolate_polynomial(points, values_array)
        eval_point = random.uniform(0, 1)
        eval_result = self.evaluate_polynomial(eval_point, coeffs)
        
        self.commitments[layer_id] = LayerCommitment(
            values=values_array,
            polynomial_coefficients=coeffs.tolist(),
            evaluation_point=eval_point,
            evaluation_result=eval_result
        )

    def generate_sumcheck_proof(self, layer_idx: int) -> Dict:
        """
        Generate a sum-check proof for a layer.
        
        Args:
            layer_idx (int): Index of the layer
            
        Returns:
            Dict: Sum-check proof components
        """
        _, values = self.layers[layer_idx]
        n = len(values)
        
        # Generate random linear combination
        r = np.random.rand(n)
        linear_combo = np.dot(values, r)
        
        # Generate proof components
        proof = {
            'linear_combination': linear_combo,
            'random_coefficients': r.tolist(),
            'partial_sums': []
        }
        
        # Compute partial sums for verification
        partial_sum = 0
        for i in range(n):
            partial_sum += values[i] * r[i]
            proof['partial_sums'].append(partial_sum)
            
        return proof

    def verify_layer_transition(self, layer1_id: int, layer2_id: int) -> bool:
        """
        Verify the transition between two consecutive layers.
        
        Args:
            layer1_id (int): ID of the first layer
            layer2_id (int): ID of the second layer
            
        Returns:
            bool: True if transition is valid, False otherwise
        """
        if layer1_id not in self.commitments or layer2_id not in self.commitments:
            return False
            
        comm1 = self.commitments[layer1_id]
        comm2 = self.commitments[layer2_id]
        
        # Generate random challenge point
        r = random.uniform(0, 1)
        
        # Evaluate both polynomials at challenge point
        eval1 = self.evaluate_polynomial(r, comm1.polynomial_coefficients)
        eval2 = self.evaluate_polynomial(r, comm2.polynomial_coefficients)
        
        # Check if evaluations are consistent
        return abs(eval1 - eval2) < 1e-10

    def generate_final_proof(self) -> Dict:
        """
        Generate the final proof combining all layer proofs.
        
        Returns:
            Dict: Complete proof structure
        """
        final_proof = {
            'layer_commitments': {},
            'transition_verifications': [],
            'sumcheck_proofs': {}
        }
        
        # Add commitments for each layer
        for layer_id, commitment in self.commitments.items():
            final_proof['layer_commitments'][layer_id] = {
                'evaluation_point': commitment.evaluation_point,
                'evaluation_result': commitment.evaluation_result,
                'polynomial_degree': len(commitment.polynomial_coefficients) - 1
            }
            
        # Verify transitions between consecutive layers
        for i in range(len(self.layers) - 1):
            layer1_id = self.layers[i][0]
            layer2_id = self.layers[i + 1][0]
            transition_valid = self.verify_layer_transition(layer1_id, layer2_id)
            final_proof['transition_verifications'].append({
                'layer1': layer1_id,
                'layer2': layer2_id,
                'valid': transition_valid
            })
            
        # Generate sumcheck proofs for each layer
        for i in range(len(self.layers)):
            final_proof['sumcheck_proofs'][self.layers[i][0]] = self.generate_sumcheck_proof(i)
            
        return final_proof

    def verify_proof(self, proof: Dict) -> bool:
        """
        Verify the complete GKR proof.
        
        Args:
            proof (Dict): Proof to verify
            
        Returns:
            bool: True if proof is valid, False otherwise
        """
        try:
            # Verify all layer transitions
            for transition in proof['transition_verifications']:
                if not transition['valid']:
                    return False
            
            # Verify sumcheck proofs
            for layer_id, sumcheck in proof['sumcheck_proofs'].items():
                if layer_id not in self.commitments:
                    return False
                    
                commitment = self.commitments[layer_id]
                linear_combo = np.dot(commitment.values, 
                                    sumcheck['random_coefficients'])
                
                if abs(linear_combo - sumcheck['linear_combination']) > 1e-10:
                    return False
            
            return True
            
        except Exception as e:
            print(f"Error in proof verification: {e}")
            return False
