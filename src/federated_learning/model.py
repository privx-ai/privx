"""
Federated Learning Model Module
Defines model structures for federated learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import json


class FederatedModel:
    """Federated Learning Base Model"""
    
    def __init__(self, model_id: str, model_architecture: nn.Module):
        """
        Initialize federated learning model
        
        Args:
            model_id: Model ID
            model_architecture: PyTorch model architecture
        """
        self.model_id = model_id
        self.model = model_architecture
        self.version = 1
        self.updates_history: List[Dict[str, Any]] = []
    
    def get_parameters(self) -> Dict[str, np.ndarray]:
        """
        Get model parameters
        
        Returns:
            Dict of model parameters
        """
        params = {}
        for name, param in self.model.named_parameters():
            params[name] = param.detach().cpu().numpy()
        return params
    
    def set_parameters(self, parameters: Dict[str, np.ndarray]) -> None:
        """
        Set model parameters
        
        Args:
            parameters: Dict of model parameters
        """
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in parameters:
                    param.copy_(torch.tensor(parameters[name]))
    
    def train(self, data_loader, epochs: int, lr: float = 0.01) -> Dict[str, float]:
        """
        Train model
        
        Args:
            data_loader: Data loader
            epochs: Number of training epochs
            lr: Learning rate
            
        Returns:
            Training result statistics
        """
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        self.model.train()
        
        stats = {
            "loss": 0.0,
            "accuracy": 0.0,
            "samples": 0
        }
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0
            
            for inputs, targets in data_loader:
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            
            epoch_loss /= len(data_loader)
            accuracy = correct / total
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}")
        
        stats["loss"] = epoch_loss
        stats["accuracy"] = accuracy
        stats["samples"] = total
        
        return stats
    
    def evaluate(self, data_loader) -> Dict[str, float]:
        """
        Evaluate model
        
        Args:
            data_loader: Data loader
            
        Returns:
            Evaluation result statistics
        """
        criterion = nn.CrossEntropyLoss()
        
        self.model.eval()
        
        test_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in data_loader:
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        test_loss /= len(data_loader)
        accuracy = correct / total
        
        stats = {
            "loss": test_loss,
            "accuracy": accuracy,
            "samples": total
        }
        
        return stats
    
    def compute_update(self, original_parameters: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Compute model update
        
        Args:
            original_parameters: Original parameters
            
        Returns:
            Model update
        """
        current_parameters = self.get_parameters()
        update = {}
        
        for name, param in current_parameters.items():
            if name in original_parameters:
                # Compute difference
                update[name] = param - original_parameters[name]
        
        return update
    
    def apply_update(self, update: Dict[str, np.ndarray]) -> None:
        """
        Apply model update
        
        Args:
            update: Model update
        """
        current_parameters = self.get_parameters()
        new_parameters = {}
        
        for name, param in current_parameters.items():
            if name in update:
                new_parameters[name] = param + update[name]
            else:
                new_parameters[name] = param
        
        self.set_parameters(new_parameters)
        self.version += 1
    
    def save_model(self, path: str) -> None:
        """
        Save model
        
        Args:
            path: Save path
        """
        state = {
            "model_id": self.model_id,
            "version": self.version,
            "state_dict": self.model.state_dict()
        }
        torch.save(state, path)
    
    def load_model(self, path: str) -> None:
        """
        Load model
        
        Args:
            path: Load path
        """
        state = torch.load(path)
        self.model_id = state["model_id"]
        self.version = state["version"]
        self.model.load_state_dict(state["state_dict"])
    
    def serialize_parameters(self) -> str:
        """
        Serialize model parameters
        
        Returns:
            str: Serialized parameter string
        """
        params = self.get_parameters()
        serialized_params = {}
        
        for name, param in params.items():
            serialized_params[name] = param.tolist()
        
        return json.dumps(serialized_params)
    
    def deserialize_parameters(self, serialized_params: str) -> Dict[str, np.ndarray]:
        """
        Deserialize model parameters
        
        Args:
            serialized_params: Serialized parameter string
            
        Returns:
            Dict: Deserialized parameter dictionary
        """
        params_dict = json.loads(serialized_params)
        params = {}
        
        for name, param in params_dict.items():
            params[name] = np.array(param)
        
        return params


class SimpleCNN(nn.Module):
    """Simple CNN Model"""
    
    def __init__(self, input_channels: int = 1, num_classes: int = 10):
        """
        Initialize CNN model
        
        Args:
            input_channels: Number of input channels
            num_classes: Number of classes
        """
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input data
            
        Returns:
            Model output
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class SimpleNN(nn.Module):
    """Simple Fully Connected Neural Network"""
    
    def __init__(self, input_size: int, hidden_size: int, num_classes: int):
        """
        Initialize fully connected neural network
        
        Args:
            input_size: Input size
            hidden_size: Hidden layer size
            num_classes: Number of classes
        """
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input data
            
        Returns:
            Model output
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def create_model(model_type: str, model_id: str, **kwargs) -> FederatedModel:
    """
    Create federated learning model
    
    Args:
        model_type: Model type
        model_id: Model ID
        **kwargs: Other parameters
        
    Returns:
        Federated learning model
        
    Raises:
        ValueError: If the model type is not supported
    """
    if model_type == "cnn":
        input_channels = kwargs.get("input_channels", 1)
        num_classes = kwargs.get("num_classes", 10)
        model_architecture = SimpleCNN(input_channels, num_classes)
        return FederatedModel(model_id, model_architecture)
    
    elif model_type == "nn":
        input_size = kwargs.get("input_size", 784)
        hidden_size = kwargs.get("hidden_size", 128)
        num_classes = kwargs.get("num_classes", 10)
        model_architecture = SimpleNN(input_size, hidden_size, num_classes)
        return FederatedModel(model_id, model_architecture)
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


# Example usage
if __name__ == "__main__":
    # Create CNN model
    cnn_model = create_model("cnn", "mnist_cnn", input_channels=1, num_classes=10)
    
    # Get parameters
    params = cnn_model.get_parameters()
    print(f"Number of model parameters: {len(params)}")
    
    # Serialize parameters
    serialized = cnn_model.serialize_parameters()
    print(f"Serialized parameter length: {len(serialized)}")
    
    # Deserialize parameters
    deserialized = cnn_model.deserialize_parameters(serialized)
    print(f"Number of deserialized parameters: {len(deserialized)}")
    
    # Set parameters
    cnn_model.set_parameters(deserialized)
    print("Parameters set")
