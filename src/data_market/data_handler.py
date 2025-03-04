#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data Processing Module
For handling dataset loading, preprocessing, encryption and other functionalities
"""

import os
import json
import logging
import hashlib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path

logger = logging.getLogger(__name__)

class DataHandler:
    """Data Handler class for processing datasets including loading, preprocessing, encryption etc."""
    
    def __init__(self, data_dir: str = "./data"):
        """
        Initialize data handler
        
        Args:
            data_dir: Data directory
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.datasets = {}  # Cache loaded datasets
        logger.info(f"Data handler initialized, data directory: {self.data_dir}")
    
    def load_dataset(self, dataset_id: str, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load dataset
        
        Args:
            dataset_id: Dataset ID
            file_path: Dataset file path, if None load from default location
            
        Returns:
            pd.DataFrame: Loaded dataset
        """
        # Return from cache if exists
        if dataset_id in self.datasets:
            logger.info(f"Loading dataset {dataset_id} from cache")
            return self.datasets[dataset_id]
        
        # Determine file path
        if file_path is None:
            file_path = self.data_dir / f"{dataset_id}.csv"
        else:
            file_path = Path(file_path)
        
        # Check if file exists
        if not file_path.exists():
            logger.error(f"Dataset file {file_path} does not exist")
            raise FileNotFoundError(f"Dataset file {file_path} does not exist")
        
        # Load data based on file extension
        ext = file_path.suffix.lower()
        try:
            if ext == '.csv':
                df = pd.read_csv(file_path)
            elif ext == '.json':
                df = pd.read_json(file_path)
            elif ext == '.parquet':
                df = pd.read_parquet(file_path)
            elif ext in ['.xls', '.xlsx']:
                df = pd.read_excel(file_path)
            else:
                logger.error(f"Unsupported file format: {ext}")
                raise ValueError(f"Unsupported file format: {ext}")
            
            # Cache dataset
            self.datasets[dataset_id] = df
            logger.info(f"Successfully loaded dataset {dataset_id}, shape: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_id}: {str(e)}")
            raise
    
    def save_dataset(self, df: pd.DataFrame, dataset_id: str, 
                    format: str = 'csv', metadata: Optional[Dict] = None) -> str:
        """
        Save dataset
        
        Args:
            df: Dataset
            dataset_id: Dataset ID
            format: Save format (csv, json, parquet)
            metadata: Metadata
            
        Returns:
            str: Saved file path
        """
        # Determine file path
        file_path = self.data_dir / f"{dataset_id}.{format}"
        
        # Save dataset
        try:
            if format == 'csv':
                df.to_csv(file_path, index=False)
            elif format == 'json':
                df.to_json(file_path, orient='records')
            elif format == 'parquet':
                df.to_parquet(file_path, index=False)
            else:
                logger.error(f"Unsupported save format: {format}")
                raise ValueError(f"Unsupported save format: {format}")
            
            # Save metadata
            if metadata:
                metadata_path = self.data_dir / f"{dataset_id}_metadata.json"
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
            
            # Update cache
            self.datasets[dataset_id] = df
            
            logger.info(f"Successfully saved dataset {dataset_id}, shape: {df.shape}, path: {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Failed to save dataset {dataset_id}: {str(e)}")
            raise
    
    def preprocess_dataset(self, df: pd.DataFrame, 
                          normalize: bool = False,
                          fill_na: bool = False,
                          drop_columns: Optional[List[str]] = None,
                          categorical_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Preprocess dataset
        
        Args:
            df: Dataset
            normalize: Whether to normalize numeric columns
            fill_na: Whether to fill missing values
            drop_columns: Columns to drop
            categorical_columns: Categorical columns for one-hot encoding
            
        Returns:
            pd.DataFrame: Preprocessed dataset
        """
        # Copy dataset to avoid modifying original data
        processed_df = df.copy()
        
        # Drop specified columns
        if drop_columns:
            processed_df = processed_df.drop(columns=[col for col in drop_columns if col in processed_df.columns])
            logger.info(f"Dropped columns: {drop_columns}")
        
        # Fill missing values
        if fill_na:
            # Fill numeric columns with mean
            num_cols = processed_df.select_dtypes(include=['number']).columns
            for col in num_cols:
                processed_df[col] = processed_df[col].fillna(processed_df[col].mean())
            
            # Fill categorical columns with mode
            cat_cols = processed_df.select_dtypes(include=['object', 'category']).columns
            for col in cat_cols:
                processed_df[col] = processed_df[col].fillna(processed_df[col].mode()[0])
            
            logger.info("Missing values filled")
        
        # Process categorical columns
        if categorical_columns:
            for col in categorical_columns:
                if col in processed_df.columns:
                    # One-hot encoding
                    dummies = pd.get_dummies(processed_df[col], prefix=col)
                    processed_df = pd.concat([processed_df.drop(columns=[col]), dummies], axis=1)
            
            logger.info(f"Categorical columns encoded: {categorical_columns}")
        
        # Normalize numeric columns
        if normalize:
            num_cols = processed_df.select_dtypes(include=['number']).columns
            for col in num_cols:
                min_val = processed_df[col].min()
                max_val = processed_df[col].max()
                if max_val > min_val:  # Avoid division by zero
                    processed_df[col] = (processed_df[col] - min_val) / (max_val - min_val)
            
            logger.info("Numeric columns normalized")
        
        logger.info(f"Data preprocessing completed, original shape: {df.shape}, processed shape: {processed_df.shape}")
        return processed_df
    
    def split_dataset(self, df: pd.DataFrame, 
                     test_size: float = 0.2, 
                     random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split dataset into training and testing sets
        
        Args:
            df: Dataset
            test_size: Proportion of the dataset to include in the test split
            random_state: Random seed
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (Training set, Test set)
        """
        # Set random seed
        np.random.seed(random_state)
        
        # Shuffle indices
        indices = np.random.permutation(len(df))
        test_count = int(len(df) * test_size)
        
        test_indices = indices[:test_count]
        train_indices = indices[test_count:]
        
        train_df = df.iloc[train_indices].reset_index(drop=True)
        test_df = df.iloc[test_indices].reset_index(drop=True)
        
        logger.info(f"Dataset split completed, training set: {train_df.shape}, test set: {test_df.shape}")
        return train_df, test_df
    
    def anonymize_dataset(self, df: pd.DataFrame, 
                         sensitive_columns: List[str],
                         method: str = 'hash') -> pd.DataFrame:
        """
        Anonymize sensitive columns in the dataset
        
        Args:
            df: Dataset
            sensitive_columns: Sensitive columns
            method: Anonymization method (hash, mask, remove)
            
        Returns:
            pd.DataFrame: Anonymized dataset
        """
        # Copy dataset to avoid modifying original data
        anon_df = df.copy()
        
        for col in sensitive_columns:
            if col not in anon_df.columns:
                logger.warning(f"Column {col} does not exist, skipping")
                continue
            
            if method == 'hash':
                # Hash processing
                anon_df[col] = anon_df[col].apply(
                    lambda x: hashlib.sha256(str(x).encode()).hexdigest()[:16] if pd.notna(x) else x
                )
                logger.info(f"Column {col} anonymized using hash method")
                
            elif method == 'mask':
                # Mask processing
                if anon_df[col].dtype == 'object':
                    # String type, retain the first two characters
                    anon_df[col] = anon_df[col].apply(
                        lambda x: str(x)[:2] + '*' * (len(str(x)) - 2) if pd.notna(x) else x
                    )
                else:
                    # Numeric type, add random noise
                    std = anon_df[col].std() * 0.1  # Noise standard deviation is 10% of the original standard deviation
                    anon_df[col] = anon_df[col] + np.random.normal(0, std, size=len(anon_df))
                
                logger.info(f"Column {col} anonymized using mask method")
                
            elif method == 'remove':
                # Directly remove
                anon_df = anon_df.drop(columns=[col])
                logger.info(f"Column {col} removed")
                
            else:
                logger.error(f"Unsupported anonymization method: {method}")
                raise ValueError(f"Unsupported anonymization method: {method}")
        
        logger.info(f"Dataset anonymization completed, processed sensitive columns: {sensitive_columns}")
        return anon_df
    
    def compute_dataset_hash(self, df: pd.DataFrame) -> str:
        """
        Compute the hash value of the dataset to verify data integrity
        
        Args:
            df: Dataset
            
        Returns:
            str: Dataset hash value
        """
        # Convert dataset to string
        data_str = df.to_csv(index=False)
        
        # Compute SHA-256 hash
        hash_obj = hashlib.sha256(data_str.encode())
        hash_value = hash_obj.hexdigest()
        
        logger.info(f"Computed dataset hash value: {hash_value}")
        return hash_value
    
    def generate_sample_dataset(self, dataset_id: str, 
                               rows: int = 1000, 
                               columns: List[Dict] = None) -> pd.DataFrame:
        """
        Generate sample dataset
        
        Args:
            dataset_id: Dataset ID
            rows: Number of rows
            columns: Column definitions, format is [{"name": "col1", "type": "int", "min": 0, "max": 100}, ...]
            
        Returns:
            pd.DataFrame: Generated dataset
        """
        if columns is None:
            columns = [
                {"name": "id", "type": "int", "min": 1, "max": rows},
                {"name": "name", "type": "str", "length": 10},
                {"name": "age", "type": "int", "min": 18, "max": 80},
                {"name": "income", "type": "float", "min": 1000, "max": 10000},
                {"name": "is_customer", "type": "bool"}
            ]
        
        data = {}
        
        for col in columns:
            col_name = col["name"]
            col_type = col["type"]
            
            if col_type == "int":
                min_val = col.get("min", 0)
                max_val = col.get("max", 100)
                data[col_name] = np.random.randint(min_val, max_val + 1, size=rows)
                
            elif col_type == "float":
                min_val = col.get("min", 0.0)
                max_val = col.get("max", 1.0)
                data[col_name] = np.random.uniform(min_val, max_val, size=rows)
                
            elif col_type == "bool":
                data[col_name] = np.random.choice([True, False], size=rows)
                
            elif col_type == "str":
                length = col.get("length", 8)
                chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
                data[col_name] = [
                    ''.join(np.random.choice(list(chars), size=length)) 
                    for _ in range(rows)
                ]
                
            elif col_type == "category":
                categories = col.get("categories", ["A", "B", "C"])
                data[col_name] = np.random.choice(categories, size=rows)
                
            elif col_type == "date":
                start_date = pd.to_datetime(col.get("start", "2020-01-01"))
                end_date = pd.to_datetime(col.get("end", "2023-12-31"))
                date_range = (end_date - start_date).days
                random_days = np.random.randint(0, date_range, size=rows)
                data[col_name] = [start_date + pd.Timedelta(days=days) for days in random_days]
                
            else:
                logger.error(f"Unsupported column type: {col_type}")
                raise ValueError(f"Unsupported column type: {col_type}")
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Save dataset
        self.save_dataset(df, dataset_id)
        
        logger.info(f"Generated sample dataset {dataset_id}, shape: {df.shape}")
        return df
