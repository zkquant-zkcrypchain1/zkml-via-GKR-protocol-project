"""
ZKML (Zero-Knowledge Machine Learning) Package
Implements zero-knowledge proofs for machine learning using the GKR protocol.
"""

from .zkml_core import ZKMLModelGKR
from .gkr_proof import GKRProof
from .data_handler import DataHandler

__version__ = "0.1.0"
__author__ = "Your Name"

# Define what should be imported with "from zkml import *"
__all__ = [
    'ZKMLModelGKR',
    'GKRProof',
    'DataHandler'
]

# Optional: Add package metadata
package_info = {
    'name': 'zkml',
    'version': __version__,
    'description': 'Zero-Knowledge Machine Learning implementation using GKR protocol',
    'author': __author__,
    'components': __all__
}
