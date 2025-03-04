#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Differential Privacy Module
Implement differential privacy computation functions to protect data privacy
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from diffprivlib import mechanisms
from diffprivlib.models import LogisticRegression, GaussianNB
from diffprivlib.tools import mean, std, var, count, sum

logger = logging.getLogger(__name__)

class DifferentialPrivacy:
    """Differential Privacy class, implementing differential privacy computation functions"""
    
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5):
        """
        Initialize Differential Privacy
        
        Args:
            epsilon: Privacy budget, the smaller the value, the stronger the privacy protection
            delta: Privacy failure probability
        """
        self.epsilon = epsilon
        self.delta = delta
        logger.info(f"Differential Privacy initialized, epsilon={epsilon}, delta={delta}")
    
    def add_noise_to_value(self, value: float, sensitivity: float, 
                          mechanism: str = 'laplace') -> float:
        """
        Add differential privacy noise to a single value
        
        Args:
            value: Original value
            sensitivity: Sensitivity
            mechanism: Noise mechanism (laplace, gaussian)
            
        Returns:
            float: Value after adding noise
        """
        if mechanism == 'laplace':
            # Laplace mechanism
            mech = mechanisms.Laplace(epsilon=self.epsilon, sensitivity=sensitivity)
            noisy_value = mech.randomise(value)
            
        elif mechanism == 'gaussian':
            # Gaussian mechanism
            mech = mechanisms.GaussianAnalytic(epsilon=self.epsilon, delta=self.delta, 
                                             sensitivity=sensitivity)
            noisy_value = mech.randomise(value)
            
        else:
            logger.error(f"Unsupported noise mechanism: {mechanism}")
            raise ValueError(f"Unsupported noise mechanism: {mechanism}")
        
        logger.info(f"Added {mechanism} noise to value {value}, result: {noisy_value}")
        return noisy_value
    
    def add_noise_to_array(self, array: np.ndarray, sensitivity: float, 
                          mechanism: str = 'laplace') -> np.ndarray:
        """
        Add differential privacy noise to an array
        
        Args:
            array: Original array
            sensitivity: Sensitivity
            mechanism: Noise mechanism (laplace, gaussian)
            
        Returns:
            np.ndarray: Array after adding noise
        """
        if mechanism == 'laplace':
            # Laplace mechanism
            mech = mechanisms.Laplace(epsilon=self.epsilon, sensitivity=sensitivity)
            noisy_array = np.array([mech.randomise(x) for x in array])
            
        elif mechanism == 'gaussian':
            # Gaussian mechanism
            mech = mechanisms.GaussianAnalytic(epsilon=self.epsilon, delta=self.delta, 
                                             sensitivity=sensitivity)
            noisy_array = np.array([mech.randomise(x) for x in array])
            
        else:
            logger.error(f"Unsupported noise mechanism: {mechanism}")
            raise ValueError(f"Unsupported noise mechanism: {mechanism}")
        
        logger.info(f"Added {mechanism} noise to array, shape: {array.shape}")
        return noisy_array
    
    def privatize_dataframe(self, df: pd.DataFrame, 
                           numeric_columns: List[str],
                           sensitivities: Dict[str, float],
                           mechanism: str = 'laplace') -> pd.DataFrame:
        """
        Add differential privacy noise to numeric columns of a dataframe
        
        Args:
            df: Original dataframe
            numeric_columns: Numeric columns to which noise should be added
            sensitivities: Sensitivity of each column
            mechanism: Noise mechanism (laplace, gaussian)
            
        Returns:
            pd.DataFrame: Dataframe after adding noise
        """
        # Copy dataframe to avoid modifying the original data
        private_df = df.copy()
        
        for col in numeric_columns:
            if col not in df.columns:
                logger.warning(f"Column {col} does not exist, skipping")
                continue
            
            if col not in sensitivities:
                logger.warning(f"Sensitivity for column {col} not specified, using default value 1.0")
                sensitivity = 1.0
            else:
                sensitivity = sensitivities[col]
            
            # Add noise
            private_df[col] = self.add_noise_to_array(
                df[col].values, 
                sensitivity=sensitivity,
                mechanism=mechanism
            )
        
        logger.info(f"Differential privacy processing of dataframe completed, processed columns: {numeric_columns}")
        return private_df
    
    def private_mean(self, values: np.ndarray, sensitivity: Optional[float] = None) -> float:
        """
        Calculate the differentially private mean
        
        Args:
            values: Numeric array
            sensitivity: Sensitivity, if None it will be calculated automatically
            
        Returns:
            float: Differentially private mean
        """
        if sensitivity is None:
            # Automatically calculate sensitivity, for mean, sensitivity is (max - min) / n
            sensitivity = (np.max(values) - np.min(values)) / len(values)
        
        # Use diffprivlib's mean function
        private_mean_value = mean(values, epsilon=self.epsilon, sensitivity=sensitivity)
        
        logger.info(f"Calculated differentially private mean: {private_mean_value}")
        return private_mean_value
    
    def private_std(self, values: np.ndarray, sensitivity: Optional[float] = None) -> float:
        """
        Calculate the differentially private standard deviation
        
        Args:
            values: Numeric array
            sensitivity: Sensitivity, if None it will be calculated automatically
            
        Returns:
            float: Differentially private standard deviation
        """
        if sensitivity is None:
            # Automatically calculate sensitivity
            sensitivity = (np.max(values) - np.min(values)) / np.sqrt(len(values))
        
        # Use diffprivlib's standard deviation function
        private_std_value = std(values, epsilon=self.epsilon, sensitivity=sensitivity)
        
        logger.info(f"Calculated differentially private standard deviation: {private_std_value}")
        return private_std_value
    
    def private_count(self, values: np.ndarray) -> int:
        """
        Calculate the differentially private count
        
        Args:
            values: Numeric array
            
        Returns:
            int: Differentially private count
        """
        # Use diffprivlib's count function
        private_count_value = count(values, epsilon=self.epsilon)
        
        logger.info(f"Calculated differentially private count: {private_count_value}")
        return private_count_value
    
    def private_sum(self, values: np.ndarray, sensitivity: Optional[float] = None) -> float:
        """
        Calculate the differentially private sum
        
        Args:
            values: Numeric array
            sensitivity: Sensitivity, if None it will be calculated automatically
            
        Returns:
            float: Differentially private sum
        """
        if sensitivity is None:
            # Automatically calculate sensitivity, for sum, sensitivity is the maximum possible value
            sensitivity = np.max(np.abs(values))
        
        # Use diffprivlib's sum function
        private_sum_value = sum(values, epsilon=self.epsilon, sensitivity=sensitivity)
        
        logger.info(f"Calculated differentially private sum: {private_sum_value}")
        return private_sum_value
    
    def private_histogram(self, values: np.ndarray, bins: int = 10, 
                         range: Optional[Tuple[float, float]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the differentially private histogram
        
        Args:
            values: Numeric array
            bins: Number of bins in the histogram
            range: Range of the histogram
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (Counts, Bin edges)
        """
        # Use diffprivlib's histogram function
        from diffprivlib.tools import histogram
        
        counts, bin_edges = histogram(values, epsilon=self.epsilon, bins=bins, range=range)
        
        logger.info(f"Calculated differentially private histogram, number of bins: {bins}")
        return counts, bin_edges
    
    def train_private_logistic_regression(self, X: np.ndarray, y: np.ndarray, 
                                         **kwargs) -> Any:
        """
        Train differentially private logistic regression model
        
        Args:
            X: Feature matrix
            y: Label vector
            **kwargs: Other parameters
            
        Returns:
            Any: Trained model
        """
        # Create differentially private logistic regression model
        model = LogisticRegression(epsilon=self.epsilon, data_norm=kwargs.get('data_norm', 1.0),
                                  **{k: v for k, v in kwargs.items() if k != 'data_norm'})
        
        # Train the model
        model.fit(X, y)
        
        logger.info(f"Training of differentially private logistic regression model completed, number of features: {X.shape[1]}")
        return model
    
    def train_private_naive_bayes(self, X: np.ndarray, y: np.ndarray, 
                                **kwargs) -> Any:
        """
        Train differentially private naive bayes model
        
        Args:
            X: Feature matrix
            y: Label vector
            **kwargs: Other parameters
            
        Returns:
            Any: Trained model
        """
        # Create differentially private naive bayes model
        model = GaussianNB(epsilon=self.epsilon, bounds=kwargs.get('bounds', None))
        
        # Train the model
        model.fit(X, y)
        
        logger.info(f"Training of differentially private naive bayes model completed, number of features: {X.shape[1]}")
        return model
    
    def privatize_gradient(self, gradient: np.ndarray, clip_norm: float = 1.0, 
                          mechanism: str = 'gaussian') -> np.ndarray:
        """
        Add differentially private noise to gradient for federated learning
        
        Args:
            gradient: Original gradient
            clip_norm: Gradient clipping norm
            mechanism: Noise mechanism (laplace, gaussian)
            
        Returns:
            np.ndarray: Gradient after adding noise
        """
        # Calculate gradient norm
        grad_norm = np.linalg.norm(gradient)
        
        # Clip gradient
        if grad_norm > clip_norm:
            gradient = gradient * (clip_norm / grad_norm)
        
        # Add noise
        private_gradient = self.add_noise_to_array(
            gradient, 
            sensitivity=clip_norm,
            mechanism=mechanism
        )
        
        logger.info(f"Gradient differential privacy processing completed, original norm: {grad_norm}, clipped norm: {clip_norm}")
        return private_gradient
