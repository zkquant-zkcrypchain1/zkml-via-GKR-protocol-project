"""
ZKML UI Package
Provides the web interface components for the ZKML application.
"""

from .streamlit_app import ZKMLApp

__version__ = "0.1.0"
__author__ = "Your Name"

# Define what should be imported with "from zkml.ui import *"
__all__ = ['ZKMLApp']

# UI-specific configurations
ui_config = {
    'name': 'ZKML Demo',
    'description': 'Zero-Knowledge Machine Learning Interactive Demo',
    'version': __version__,
    'author': __author__,
    'theme': {
        'primaryColor': '#FF4B4B',
        'backgroundColor': '#FFFFFF',
        'secondaryBackgroundColor': '#F0F2F6',
        'textColor': '#262730',
        'font': 'sans-serif'
    },
    'pages': {
        'main': 'ZKML Demo',
        'about': 'About ZKML',
        'documentation': 'Documentation'
    }
}

def get_ui_config():
    """Return the UI configuration"""
    return ui_config

def launch_app():
    """Launch the Streamlit application"""
    app = ZKMLApp()
    app.run()

# Optional: Add any UI initialization code here
def initialize_ui():
    """Initialize UI components"""
    pass
