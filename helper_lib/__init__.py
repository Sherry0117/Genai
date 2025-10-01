"""
Helper Library for Neural Network Projects

This library provides common functionalities for data loading, model training, 
and evaluation to reduce duplication across various neural network projects.

Modules:
- data_loader: Data loading and preprocessing utilities
- trainer: Model training functionality
- evaluator: Model evaluation and metrics
- model: Neural network model definitions
- utils: General utility functions
"""

from .data_loader import get_data_loader
from .trainer import train_model
from .evaluator import evaluate_model
from .model import FCNN, CNN, EnhancedCNN, get_model
from .utils import (
    count_parameters, 
    get_optimizer, 
    get_criterion, 
    plot_training_history, 
    save_model, 
    load_model
)

__version__ = "0.1.0"
__author__ = "SPS GenAI"
