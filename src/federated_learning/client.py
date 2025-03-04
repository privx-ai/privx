"""
Federated Learning Client Module
Implements client functionality for federated learning
"""

import torch
import torch.utils.data
import numpy as np
import json
import time
import requests
from typing import Dict, List, Any, Optional, Tuple
import os
import logging
from .model import FederatedModel


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("FederatedClient")


class FederatedClient:
    """Federated Learning Client"""
    
    def __init__(self, client_id: str, server_url: str, data_loader=None, 
                 model: Optional[FederatedModel] = None):
        """
        Initialize federated learning client
        
        Args:
            client_id: Client ID
            server_url: Server URL
            data_loader: Data loader
            model: Federated learning model
        """
        self.client_id = client_id
        self.server_url = server_url
        self.data_loader = data_loader
        self.model = model
        self.training_config = {}
        self.is_registered = False
        self.current_round_id = None
        self.metadata = {
            "device": "cpu" if not torch.cuda.is_available() else "cuda",
            "created_at": time.time(),
            "last_update": time.time(),
            "total_samples": 0 if data_loader is None else len(data_loader.dataset),
            "data_description": "Not specified"
        }
    
    def register(self, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Register client with server
        
        Args:
            metadata: Client metadata
            
        Returns:
            bool: Registration success
        """
        if metadata:
            self.metadata.update(metadata)
        
        try:
            response = requests.post(
                f"{self.server_url}/register",
                json={
                    "client_id": self.client_id,
                    "metadata": self.metadata
                }
            )
            
            if response.status_code == 200:
                self.is_registered = True
                logger.info(f"Client {self.client_id} registered successfully")
                return True
            else:
                logger.error(f"Client registration failed: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error during registration: {str(e)}")
            return False
    
    def get_model_from_server(self, model_id: str) -> bool:
        """
        Get model from server
        
        Args:
            model_id: Model ID
            
        Returns:
            bool: Success status
        """
        try:
            response = requests.get(
                f"{self.server_url}/models/{model_id}"
            )
            
            if response.status_code == 200:
                model_data = response.json()
                
                if self.model is None:
                    logger.error("Client model not initialized")
                    return False
                
                # Set model parameters
                serialized_params = model_data.get("parameters")
                if serialized_params:
                    params = self.model.deserialize_parameters(serialized_params)
                    self.model.set_parameters(params)
                    self.model.version = model_data.get("version", 1)
                    logger.info(f"Successfully retrieved model {model_id} from server, version: {self.model.version}")
                    return True
                else:
                    logger.error("No parameters in model data from server")
                    return False
            else:
                logger.error(f"Failed to get model: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error while getting model: {str(e)}")
            return False
    
    def join_training_round(self, round_id: str) -> bool:
        """
        Join training round
        
        Args:
            round_id: Round ID
            
        Returns:
            bool: Join success
        """
        if not self.is_registered:
            logger.error("Client is not registered, cannot join training round")
            return False
        
        try:
            response = requests.post(
                f"{self.server_url}/rounds/{round_id}/join",
                json={
                    "client_id": self.client_id
                }
            )
            
            if response.status_code == 200:
                round_data = response.json()
                self.current_round_id = round_id
                self.training_config = round_data.get("config", {})
                logger.info(f"Successfully joined training round {round_id}")
                
                # Get latest model
                model_id = round_data.get("model_id")
                if model_id:
                    return self.get_model_from_server(model_id)
                return True
            else:
                logger.error(f"Failed to join training round: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error during joining training round: {str(e)}")
            return False
    
    def train_local_model(self) -> Dict[str, Any]:
        """
        Train local model
        
        Returns:
            Training result statistics
        """
        if self.model is None or self.data_loader is None:
            logger.error("Model or data loader not initialized")
            return {"error": "Model or data loader not initialized"}
        
        # Save original parameters for computing update
        original_parameters = self.model.get_parameters()
        
        # Get parameters from training config
        epochs = self.training_config.get("epochs", 1)
        lr = self.training_config.get("learning_rate", 0.01)
        
        # Train model
        logger.info(f"Start local training, epochs: {epochs}, learning rate: {lr}")
        stats = self.model.train(self.data_loader, epochs, lr)
        
        # Compute update
        update = self.model.compute_update(original_parameters)
        
        # Restore original parameters, waiting for aggregation
        self.model.set_parameters(original_parameters)
        
        return {
            "stats": stats,
            "update": update
        }
    
    def submit_update(self, training_result: Dict[str, Any]) -> bool:
        """
        Submit model update
        
        Args:
            training_result: Training result
            
        Returns:
            bool: Submission success
        """
        if not self.current_round_id:
            logger.error("Not joined training round, cannot submit update")
            return False
        
        try:
            # Serialize update
            update = training_result.get("update", {})
            serialized_update = {}
            
            for name, param in update.items():
                serialized_update[name] = param.tolist()
            
            # Prepare submission data
            submission = {
                "client_id": self.client_id,
                "round_id": self.current_round_id,
                "update": json.dumps(serialized_update),
                "metadata": {
                    "stats": training_result.get("stats", {}),
                    "timestamp": time.time()
                }
            }
            
            response = requests.post(
                f"{self.server_url}/rounds/{self.current_round_id}/submit",
                json=submission
            )
            
            if response.status_code == 200:
                logger.info(f"Successfully submitted update to round {self.current_round_id}")
                return True
            else:
                logger.error(f"Failed to submit update: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error during submitting update: {str(e)}")
            return False
    
    def get_aggregated_model(self) -> bool:
        """
        Get aggregated model
        
        Returns:
            bool: Get success
        """
        if not self.current_round_id:
            logger.error("Not joined training round, cannot get aggregated model")
            return False
        
        try:
            response = requests.get(
                f"{self.server_url}/rounds/{self.current_round_id}/result"
            )
            
            if response.status_code == 200:
                result = response.json()
                
                if result.get("status") != "completed":
                    logger.info(f"Round {self.current_round_id} is not completed yet")
                    return False
                
                # Get aggregated model
                model_id = result.get("model_id")
                if model_id:
                    return self.get_model_from_server(model_id)
                return False
            else:
                logger.error(f"Failed to get aggregated model: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error during getting aggregated model: {str(e)}")
            return False
    
    def evaluate_model(self, test_data_loader=None) -> Dict[str, float]:
        """
        Evaluate model
        
        Args:
            test_data_loader: Test data loader, if None then use client's data loader
            
        Returns:
            Evaluation result statistics
        """
        if self.model is None:
            logger.error("Model not initialized")
            return {"error": "Model not initialized"}
        
        data_loader = test_data_loader if test_data_loader is not None else self.data_loader
        
        if data_loader is None:
            logger.error("Data loader not initialized")
            return {"error": "Data loader not initialized"}
        
        logger.info("Start evaluating model")
        return self.model.evaluate(data_loader)
    
    def save_model(self, path: str) -> bool:
        """
        Save model
        
        Args:
            path: Save path
            
        Returns:
            bool: Save success
        """
        if self.model is None:
            logger.error("Model not initialized")
            return False
        
        try:
            directory = os.path.dirname(path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
                
            self.model.save_model(path)
            logger.info(f"Model saved to {path}")
            return True
        except Exception as e:
            logger.error(f"Error during saving model: {str(e)}")
            return False
    
    def load_model(self, path: str) -> bool:
        """
        Load model
        
        Args:
            path: Load path
            
        Returns:
            bool: Load success
        """
        if self.model is None:
            logger.error("Model not initialized")
            return False
        
        try:
            self.model.load_model(path)
            logger.info(f"Model loaded from {path}")
            return True
        except Exception as e:
            logger.error(f"Error during loading model: {str(e)}")
            return False
    
    def set_data_loader(self, data_loader) -> None:
        """
        Set data loader
        
        Args:
            data_loader: Data loader
        """
        self.data_loader = data_loader
        if data_loader is not None:
            self.metadata["total_samples"] = len(data_loader.dataset)
            self.metadata["last_update"] = time.time()
    
    def set_model(self, model: FederatedModel) -> None:
        """
        Set model
        
        Args:
            model: Federated learning model
        """
        self.model = model


# Mock data loader
class MockDataLoader:
    """Mock data loader, for testing"""
    
    def __init__(self, num_samples: int = 100, batch_size: int = 10):
        """
        Initialize mock data loader
        
        Args:
            num_samples: Number of samples
            batch_size: Batch size
        """
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.dataset = list(range(num_samples))
    
    def __iter__(self):
        """Iterator"""
        for i in range(0, self.num_samples, self.batch_size):
            batch = self.dataset[i:i+self.batch_size]
            # Simulate inputs and targets
            inputs = torch.randn(len(batch), 1, 28, 28)
            targets = torch.randint(0, 10, (len(batch),))
            yield inputs, targets
    
    def __len__(self):
        """Data loader length"""
        return (self.num_samples + self.batch_size - 1) // self.batch_size


# Example usage
if __name__ == "__main__":
    from model import create_model
    
    # Create model
    model = create_model("cnn", "mnist_cnn")
    
    # Create mock data loader
    data_loader = MockDataLoader(1000, 32)
    
    # Create client
    client = FederatedClient(
        client_id="client1",
        server_url="http://localhost:5000",
        data_loader=data_loader,
        model=model
    )
    
    # Register client
    client.register({
        "data_description": "Part of MNIST dataset",
        "organization": "Test Organization"
    })
    
    # Simulate local training
    result = client.train_local_model()
    print(f"Training result: {result['stats']}")
    
    # Save model
    client.save_model("client_model.pt") 
