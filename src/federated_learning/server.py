"""
Federated Learning Server Module
Implements server functionality for federated learning
"""

import torch
import numpy as np
import json
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
import os
from flask import Flask, request, jsonify
from .model import FederatedModel, create_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("FederatedServer")


class FederatedServer:
    """Federated Learning Server"""
    
    def __init__(self, server_id: str, model: Optional[FederatedModel] = None):
        """
        Initialize federated learning server
        
        Args:
            server_id: Server ID
            model: Federated learning model
        """
        self.server_id = server_id
        self.model = model
        self.clients: Dict[str, Dict[str, Any]] = {}
        self.models: Dict[str, Dict[str, Any]] = {}
        self.training_rounds: Dict[str, Dict[str, Any]] = {}
        self.current_round_id: Optional[str] = None
        
        # Register model if provided
        if model:
            self.register_model(model)
    
    def register_client(self, client_id: str, metadata: Dict[str, Any]) -> bool:
        """
        Register client
        
        Args:
            client_id: Client ID
            metadata: Client metadata
            
        Returns:
            bool: Registration success
        """
        if client_id in self.clients:
            # Update existing client
            self.clients[client_id].update({
                "metadata": metadata,
                "last_seen": time.time()
            })
            logger.info(f"Updated client {client_id}")
        else:
            # Register new client
            self.clients[client_id] = {
                "client_id": client_id,
                "metadata": metadata,
                "registered_at": time.time(),
                "last_seen": time.time(),
                "rounds_participated": [],
                "total_contributions": 0
            }
            logger.info(f"Registered new client {client_id}")
        
        return True
    
    def register_model(self, model: FederatedModel) -> str:
        """
        Register model
        
        Args:
            model: Federated learning model
            
        Returns:
            Model ID
        """
        model_id = model.model_id
        
        if model_id in self.models:
            logger.warning(f"Model {model_id} already exists, will be overwritten")
        
        # Get model parameters
        parameters = model.get_parameters()
        serialized_params = model.serialize_parameters()
        
        # Register model
        self.models[model_id] = {
            "model_id": model_id,
            "parameters": serialized_params,
            "version": model.version,
            "created_at": time.time(),
            "updated_at": time.time(),
            "total_rounds": 0
        }
        
        logger.info(f"Registered model {model_id}, version: {model.version}")
        
        # Set current model
        self.model = model
        
        return model_id
    
    def create_training_round(self, model_id: str, round_id: str, config: Dict[str, Any]) -> str:
        """
        Create training round
        
        Args:
            model_id: Model ID
            round_id: Round ID
            config: Training configuration
            
        Returns:
            Round ID
            
        Raises:
            ValueError: If model does not exist or round ID already exists
        """
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} does not exist")
        
        if round_id in self.training_rounds:
            raise ValueError(f"Round ID {round_id} already exists")
        
        # Create training round
        self.training_rounds[round_id] = {
            "round_id": round_id,
            "model_id": model_id,
            "config": config,
            "status": "created",
            "created_at": time.time(),
            "started_at": None,
            "completed_at": None,
            "participants": [],
            "updates": {},
            "results": None
        }
        
        logger.info(f"Created training round {round_id} for model {model_id}")
        
        # Update model information
        self.models[model_id]["total_rounds"] += 1
        self.models[model_id]["updated_at"] = time.time()
        
        # Set current round
        self.current_round_id = round_id
        
        return round_id
    
    def start_training_round(self, round_id: str) -> bool:
        """
        Start training round
        
        Args:
            round_id: Round ID
            
        Returns:
            Start success
            
        Raises:
            ValueError: If round does not exist or status is incorrect
        """
        if round_id not in self.training_rounds:
            raise ValueError(f"Round {round_id} does not exist")
        
        training_round = self.training_rounds[round_id]
        
        if training_round["status"] != "created":
            raise ValueError(f"Round {round_id} status is incorrect: {training_round['status']}")
        
        # Update round status
        training_round["status"] = "active"
        training_round["started_at"] = time.time()
        
        logger.info(f"Started training round {round_id}")
        
        return True
    
    def add_client_to_round(self, round_id: str, client_id: str) -> bool:
        """
        Add client to training round
        
        Args:
            round_id: Round ID
            client_id: Client ID
            
        Returns:
            Add success
            
        Raises:
            ValueError: If round or client does not exist, or round status is incorrect
        """
        if round_id not in self.training_rounds:
            raise ValueError(f"Round {round_id} does not exist")
        
        if client_id not in self.clients:
            raise ValueError(f"Client {client_id} does not exist")
        
        training_round = self.training_rounds[round_id]
        
        if training_round["status"] != "active":
            raise ValueError(f"Round {round_id} status is incorrect: {training_round['status']}")
        
        # Add client to participant list
        if client_id not in training_round["participants"]:
            training_round["participants"].append(client_id)
            
            # Update client information
            self.clients[client_id]["rounds_participated"].append(round_id)
            self.clients[client_id]["last_seen"] = time.time()
            
            logger.info(f"Client {client_id} joined round {round_id}")
        
        return True
    
    def submit_update(self, round_id: str, client_id: str, 
                     update: Dict[str, Any], metadata: Dict[str, Any]) -> bool:
        """
        Submit model update
        
        Args:
            round_id: Round ID
            client_id: Client ID
            update: Model update
            metadata: Update metadata
            
        Returns:
            Submit success
            
        Raises:
            ValueError: If round or client does not exist, or round status is incorrect
        """
        if round_id not in self.training_rounds:
            raise ValueError(f"Round {round_id} does not exist")
        
        if client_id not in self.clients:
            raise ValueError(f"Client {client_id} does not exist")
        
        training_round = self.training_rounds[round_id]
        
        if training_round["status"] != "active":
            raise ValueError(f"Round {round_id} status is incorrect: {training_round['status']}")
        
        # Ensure client is a participant
        if client_id not in training_round["participants"]:
            self.add_client_to_round(round_id, client_id)
        
        # Record update
        training_round["updates"][client_id] = {
            "update": update,
            "metadata": metadata,
            "timestamp": time.time()
        }
        
        # Update client information
        self.clients[client_id]["total_contributions"] += 1
        self.clients[client_id]["last_seen"] = time.time()
        
        logger.info(f"Client {client_id} submitted update to round {round_id}")
        
        return True
    
    def aggregate_updates(self, round_id: str, aggregation_method: str = "fedavg") -> Dict[str, np.ndarray]:
        """
        Aggregate model updates
        
        Args:
            round_id: Round ID
            aggregation_method: Aggregation method
            
        Returns:
            Aggregated update
            
        Raises:
            ValueError: If round does not exist or has no updates
        """
        if round_id not in self.training_rounds:
            raise ValueError(f"Round {round_id} does not exist")
        
        training_round = self.training_rounds[round_id]
        updates = training_round["updates"]
        
        if not updates:
            raise ValueError(f"Round {round_id} has no updates to aggregate")
        
        # Get model ID and parameters
        model_id = training_round["model_id"]
        model_params = self.models[model_id]["parameters"]
        
        # Deserialize parameters
        if self.model and self.model.model_id == model_id:
            current_params = self.model.deserialize_parameters(model_params)
        else:
            logger.error(f"Current model is not {model_id}, cannot aggregate updates")
            return {}
        
        # Aggregate updates
        if aggregation_method == "fedavg":
            # Federated averaging
            aggregated_update = self._federated_averaging(updates)
        else:
            raise ValueError(f"Unsupported aggregation method: {aggregation_method}")
        
        logger.info(f"Aggregated {len(updates)} updates for round {round_id}")
        
        return aggregated_update
    
    def _federated_averaging(self, updates: Dict[str, Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """
        Federated averaging aggregation method
        
        Args:
            updates: Client updates
            
        Returns:
            Aggregated update
        """
        if not updates:
            return {}
        
        # Extract updates and sample counts
        client_updates = {}
        sample_counts = {}
        
        for client_id, update_data in updates.items():
            # Deserialize update
            serialized_update = update_data["update"]
            if isinstance(serialized_update, str):
                update_dict = json.loads(serialized_update)
            else:
                update_dict = serialized_update
            
            # Convert to numpy array
            update = {}
            for name, param in update_dict.items():
                if isinstance(param, list):
                    update[name] = np.array(param)
                else:
                    update[name] = param
            
            client_updates[client_id] = update
            
            # Get sample count
            metadata = update_data.get("metadata", {})
            stats = metadata.get("stats", {})
            sample_counts[client_id] = stats.get("samples", 1)
        
        # Calculate total samples
        total_samples = sum(sample_counts.values())
        
        # Initialize aggregated update
        aggregated_update = {}
        
        # Weighted average for each parameter
        for client_id, update in client_updates.items():
            weight = sample_counts[client_id] / total_samples
            
            for name, param in update.items():
                if name not in aggregated_update:
                    aggregated_update[name] = param * weight
                else:
                    aggregated_update[name] += param * weight
        
        return aggregated_update
    
    def complete_round(self, round_id: str) -> bool:
        """
        Complete training round
        
        Args:
            round_id: Round ID
            
        Returns:
            Complete success
            
        Raises:
            ValueError: If round does not exist or status is incorrect
        """
        if round_id not in self.training_rounds:
            raise ValueError(f"Round {round_id} does not exist")
        
        training_round = self.training_rounds[round_id]
        
        if training_round["status"] != "active":
            raise ValueError(f"Round {round_id} status is incorrect: {training_round['status']}")
        
        # Aggregate updates
        try:
            aggregated_update = self.aggregate_updates(round_id)
            
            # Get model
            model_id = training_round["model_id"]
            
            if self.model and self.model.model_id == model_id:
                # Apply update
                self.model.apply_update(aggregated_update)
                
                # Update model information
                self.models[model_id]["parameters"] = self.model.serialize_parameters()
                self.models[model_id]["version"] = self.model.version
                self.models[model_id]["updated_at"] = time.time()
                
                # Record result
                training_round["results"] = {
                    "model_id": model_id,
                    "model_version": self.model.version,
                    "participants": len(training_round["participants"]),
                    "updates": len(training_round["updates"]),
                    "timestamp": time.time()
                }
                
                # Update round status
                training_round["status"] = "completed"
                training_round["completed_at"] = time.time()
                
                logger.info(f"Completed training round {round_id}, model version: {self.model.version}")
                
                return True
            else:
                logger.error(f"Current model is not {model_id}, cannot complete round")
                return False
                
        except Exception as e:
            logger.error(f"Error while completing round: {str(e)}")
            training_round["status"] = "failed"
            return False
    
    def get_model(self, model_id: str) -> Dict[str, Any]:
        """
        Get model information
        
        Args:
            model_id: Model ID
            
        Returns:
            Model information
            
        Raises:
            ValueError: If model does not exist
        """
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} does not exist")
        
        return self.models[model_id]
    
    def get_round(self, round_id: str) -> Dict[str, Any]:
        """
        Get round information
        
        Args:
            round_id: Round ID
            
        Returns:
            Round information
            
        Raises:
            ValueError: If round does not exist
        """
        if round_id not in self.training_rounds:
            raise ValueError(f"Round {round_id} does not exist")
        
        return self.training_rounds[round_id]
    
    def get_client(self, client_id: str) -> Dict[str, Any]:
        """
        Get client information
        
        Args:
            client_id: Client ID
            
        Returns:
            Client information
            
        Raises:
            ValueError: If client does not exist
        """
        if client_id not in self.clients:
            raise ValueError(f"Client {client_id} does not exist")
        
        return self.clients[client_id]
    
    def save_model(self, model_id: str, path: str) -> bool:
        """
        Save model
        
        Args:
            model_id: Model ID
            path: Save path
            
        Returns:
            Save success
            
        Raises:
            ValueError: If model does not exist
        """
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} does not exist")
        
        if not self.model or self.model.model_id != model_id:
            logger.error(f"Current model is not {model_id}, cannot save")
            return False
        
        try:
            directory = os.path.dirname(path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
                
            self.model.save_model(path)
            logger.info(f"Model {model_id} saved to {path}")
            return True
        except Exception as e:
            logger.error(f"Error while saving model: {str(e)}")
            return False
    
    def load_model(self, path: str) -> str:
        """
        Load model
        
        Args:
            path: Load path
            
        Returns:
            str: Model ID
            
        Raises:
            ValueError: If model loading fails
        """
        if not self.model:
            logger.error("Server model not initialized")
            raise ValueError("Server model not initialized")
        
        try:
            self.model.load_model(path)
            model_id = self.model.model_id
            
            # Update or register model
            return self.register_model(self.model)
        except Exception as e:
            logger.error(f"Error while loading model: {str(e)}")
            raise ValueError(f"Failed to load model: {str(e)}")


class FederatedServerAPI:
    """Federated Learning Server API"""
    
    def __init
