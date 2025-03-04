#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Federated Learning Demo
Demonstrates basic federated learning functionality with blockchain and privacy computing
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Add project root directory to system path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.blockchain.blockchain import Blockchain
from src.blockchain.smart_contract import DataMarketContract
from src.federated_learning.server import FederatedServer
from src.federated_learning.client import FederatedClient
from src.federated_learning.model import LogisticRegressionModel
from src.data_market.data_handler import DataHandler
from src.privacy.differential_privacy import DifferentialPrivacy
from src.privacy.homomorphic_encryption import HomomorphicEncryption

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def prepare_data(data_handler):
    """Prepare dataset"""
    # Load medical dataset
    medical_df = data_handler.load_dataset("medical")
    
    # Add target variable (diabetes risk prediction)
    np.random.seed(42)
    
    # Generate diabetes risk based on blood pressure, heart rate and cholesterol
    medical_df['diabetes_risk'] = (
        0.02 * (medical_df['blood_pressure'] - 120) + 
        0.01 * (medical_df['heart_rate'] - 70) + 
        0.005 * (medical_df['cholesterol'] - 200) + 
        np.random.normal(0, 0.5, size=len(medical_df))
    )
    
    # Convert risk to binary labels
    medical_df['diabetes_risk_label'] = (medical_df['diabetes_risk'] > 0.5).astype(int)
    
    # Select features and target
    features = ['age', 'blood_pressure', 'heart_rate', 'cholesterol']
    target = 'diabetes_risk_label'
    
    # Split dataset
    X = medical_df[features].values
    y = medical_df[target].values
    
    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    logger.info(f"Data preparation completed, features: {features}, samples: {len(X)}")
    return X, y, features

def split_data_for_clients(X, y, num_clients=3, iid=True):
    """Split data for multiple clients"""
    if iid:
        # IID data split (Independent and Identically Distributed)
        client_data = []
        for i in range(num_clients):
            # Random sampling
            indices = np.random.choice(len(X), size=len(X) // num_clients, replace=False)
            client_X = X[indices]
            client_y = y[indices]
            client_data.append((client_X, client_y))
            logger.info(f"Client {i} data: {len(client_X)} samples")
    else:
        # Non-IID data split (Non-Independent and Identically Distributed)
        # Sort by labels
        sorted_indices = np.argsort(y)
        sorted_X = X[sorted_indices]
        sorted_y = y[sorted_indices]
        
        # Assign different proportions of classes to each client
        client_data = []
        samples_per_client = len(X) // num_clients
        
        for i in range(num_clients):
            start_idx = i * samples_per_client
            end_idx = (i + 1) * samples_per_client if i < num_clients - 1 else len(X)
            client_X = sorted_X[start_idx:end_idx]
            client_y = sorted_y[start_idx:end_idx]
            client_data.append((client_X, client_y))
            
            # Calculate class distribution
            unique, counts = np.unique(client_y, return_counts=True)
            class_dist = dict(zip(unique, counts))
            logger.info(f"Client {i} data: {len(client_X)} samples, class distribution: {class_dist}")
    
    return client_data

def setup_federated_learning(client_data, use_privacy=False, use_encryption=False):
    """Setup federated learning environment"""
    # Initialize blockchain
    blockchain = Blockchain()
    logger.info(f"Blockchain initialized, current blocks: {len(blockchain.chain)}")
    
    # Initialize smart contract
    contract = DataMarketContract(blockchain)
    logger.info(f"Smart contract initialized, contract address: {contract.address}")
    
    # Initialize federated learning server
    input_dim = client_data[0][0].shape[1]  # Feature dimension
    server = FederatedServer(
        model_type='logistic_regression',
        input_dim=input_dim,
        learning_rate=0.01,
        contract=contract
    )
    logger.info(f"Federated learning server initialized, model type: logistic_regression, input dimension: {input_dim}")
    
    # Initialize differential privacy (if enabled)
    differential_privacy = None
    if use_privacy:
        differential_privacy = DifferentialPrivacy(epsilon=1.0, delta=1e-5)
        logger.info(f"Differential privacy initialized, epsilon={differential_privacy.epsilon}")
    
    # Initialize homomorphic encryption (if enabled)
    homomorphic_encryption = None
    if use_encryption:
        homomorphic_encryption = HomomorphicEncryption(scheme='ckks')
        logger.info(f"Homomorphic encryption initialized, scheme: {homomorphic_encryption.scheme}")
    
    # Initialize federated learning clients
    clients = []
    for i, (client_X, client_y) in enumerate(client_data):
        client = FederatedClient(
            client_id=f"client_{i}",
            model_type='logistic_regression',
            input_dim=input_dim,
            learning_rate=0.01,
            differential_privacy=differential_privacy,
            homomorphic_encryption=homomorphic_encryption,
            contract=contract
        )
        # Set client data
        client.set_data(client_X, client_y)
        clients.append(client)
        logger.info(f"Federated learning client {client.client_id} initialized, data size: {len(client_X)}")
    
    return server, clients, blockchain, contract

def train_federated_model(server, clients, num_rounds=10, local_epochs=5):
    """Train federated learning model"""
    # Record training history
    history = {
        'server_loss': [],
        'client_losses': [[] for _ in clients],
        'client_accuracies': [[] for _ in clients]
    }
    
    # Federated learning training process
    for round_idx in range(num_rounds):
        logger.info(f"Starting round {round_idx + 1}/{num_rounds} of federated learning")
        
        # Server sends global model
        global_weights = server.get_model_weights()
        for client in clients:
            client.set_model_weights(global_weights)
        
        # Client local training
        client_weights = []
        client_sample_counts = []
        
        for i, client in enumerate(clients):
            # Local training
            local_history = client.train_local_model(epochs=local_epochs)
            
            # Record training history
            history['client_losses'][i].extend(local_history['loss'])
            history['client_accuracies'][i].extend(local_history['accuracy'])
            
            # Get updated weights
            weights = client.get_model_weights()
            client_weights.append(weights)
            client_sample_counts.append(len(client.X))
            
            logger.info(f"Client {client.client_id} local training completed, final loss: {local_history['loss'][-1]:.4f}, accuracy: {local_history['accuracy'][-1]:.4f}")
        
        # Server aggregates model
        server.aggregate_weights(client_weights, client_sample_counts)
        
        # Evaluate global model
        server_loss = server.evaluate_model()
        history['server_loss'].append(server_loss)
        
        logger.info(f"Round {round_idx + 1} of federated learning completed, global model loss: {server_loss:.4f}")
        
        # Record global model to blockchain
        if server.contract:
            model_hash = server.save_model_to_blockchain(round_idx + 1)
            logger.info(f"Global model recorded to blockchain, round: {round_idx + 1}, hash: {model_hash}")
    
    return history

def evaluate_model(server, X_test, y_test):
    """Evaluate global model"""
    # Get global model
    global_model = server.model
    
    # Predict
    y_pred_prob = global_model.predict_proba(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_prob)
    
    logger.info(f"Global model evaluation results:")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    logger.info(f"AUC: {auc:.4f}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }

def compare_with_centralized(X, y, X_test, y_test):
    """Compare with centralized model"""
    # Train centralized model
    centralized_model = LogisticRegressionModel(input_dim=X.shape[1])
    
    # Training history
    history = centralized_model.fit(X, y, epochs=50)
    
    # Predict
    y_pred_prob = centralized_model.predict_proba(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_prob)
    
    logger.info(f"Centralized model evaluation results:")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    logger.info(f"AUC: {auc:.4f}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'history': history
    }

def visualize_results(federated_history, centralized_results, federated_metrics, centralized_metrics):
    """Visualize results"""
    # Create charts directory
    charts_dir = ROOT_DIR / "examples" / "charts"
    charts_dir.mkdir(exist_ok=True)
    
    # 1. Federated learning training history
    plt.figure(figsize=(12, 5))
    
    # Client loss
    plt.subplot(1, 2, 1)
    for i, client_loss in enumerate(federated_history['client_losses']):
        plt.plot(client_loss, alpha=0.5, label=f'Client {i}')
    plt.plot(np.arange(0, len(federated_history['server_loss']) * 5, 5), 
             federated_history['server_loss'], 'r-', linewidth=2, label='Global Model')
    plt.title('Federated Learning Loss Curves')
    plt.xlabel('Training Rounds')
    plt.ylabel('Loss')
    plt.legend()
    
    # Client accuracy
    plt.subplot(1, 2, 2)
    for i, client_acc in enumerate(federated_history['client_accuracies']):
        plt.plot(client_acc, alpha=0.5, label=f'Client {i}')
    plt.title('Federated Learning Accuracy Curves')
    plt.xlabel('Training Rounds')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(charts_dir / "federated_learning_history.png")
    
    # 2. Centralized vs Federated Learning Performance Comparison
    plt.figure(figsize=(10, 6))
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    x = np.arange(len(metrics))
    width = 0.35
    
    centralized_values = [centralized_metrics[m] for m in metrics]
    federated_values = [federated_metrics[m] for m in metrics]
    
    plt.bar(x - width/2, centralized_values, width, label='Centralized Model')
    plt.bar(x + width/2, federated_values, width, label='Federated Learning Model')
    
    plt.title('Centralized vs Federated Learning Performance')
    plt.xticks(x, metrics)
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(charts_dir / "centralized_vs_federated.png")
    
    # 3. Centralized model training history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(centralized_results['history']['loss'])
    plt.title('Centralized Model Loss Curve')
    plt.xlabel('Training Epochs')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(centralized_results['history']['accuracy'])
    plt.title('Centralized Model Accuracy Curve')
    plt.xlabel('Training Epochs')
    plt.ylabel('Accuracy')
    
    plt.tight_layout()
    plt.savefig(charts_dir / "centralized_history.png")
    
    logger.info(f"Results visualization completed, charts saved to {charts_dir}")

def main():
    """Main function"""
    logger.info("Starting federated learning demo...")
    
    # Initialize data handler
    data_handler = DataHandler(data_dir=ROOT_DIR / "data")
    
    # Prepare data
    X, y, features = prepare_data(data_handler)
    
    # Split test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logger.info(f"Data split completed, training set: {X_train.shape}, test set: {X_test.shape}")
    
    # Split data for clients
    client_data = split_data_for_clients(X_train, y_train, num_clients=3, iid=False)
    
    # Setup federated learning environment
    server, clients, blockchain, contract = setup_federated_learning(
        client_data, 
        use_privacy=True,
        use_encryption=False
    )
    
    # Train federated learning model
    federated_history = train_federated_model(server, clients, num_rounds=10, local_epochs=5)
    
    # Evaluate federated learning model
    federated_metrics = evaluate_model(server, X_test, y_test)
    
    # Compare with centralized model
    centralized_results = compare_with_centralized(X_train, y_train, X_test, y_test)
    
    # Visualize results
    visualize_results(federated_history, centralized_results, federated_metrics, centralized_results)
    
    logger.info("Federated learning demo completed!")

if __name__ == "__main__":
    main() 