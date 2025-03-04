"""
Smart Contract Simulation Module
For simulating smart contract functionality on blockchain, especially for data trading and privacy computing
"""

import json
import time
from typing import Dict, List, Any, Callable, Optional
from .blockchain import Blockchain


class SmartContract:
    """Smart Contract Base Class"""
    
    def __init__(self, contract_id: str, owner: str, blockchain: Blockchain):
        """
        Initialize Smart Contract
        
        Args:
            contract_id: Contract ID
            owner: Contract owner address
            blockchain: Blockchain instance
        """
        self.contract_id = contract_id
        self.owner = owner
        self.blockchain = blockchain
        self.state: Dict[str, Any] = {}
        self.functions: Dict[str, Callable] = {}
        self.events: List[Dict[str, Any]] = []
        self.created_at = time.time()
    
    def register_function(self, name: str, function: Callable) -> None:
        """
        Register contract function
        
        Args:
            name: Function name
            function: Function object
        """
        self.functions[name] = function
    
    def call(self, function_name: str, caller: str, *args, **kwargs) -> Any:
        """
        Call contract function
        
        Args:
            function_name: Name of function to call
            caller: Caller address
            *args, **kwargs: Arguments passed to function
            
        Returns:
            Function call result
            
        Raises:
            ValueError: If function does not exist
        """
        if function_name not in self.functions:
            raise ValueError(f"Function {function_name} does not exist")
        
        # Record function call
        self.events.append({
            "type": "function_call",
            "function": function_name,
            "caller": caller,
            "timestamp": time.time(),
            "args": args,
            "kwargs": kwargs
        })
        
        # Call function
        return self.functions[function_name](caller, *args, **kwargs)
    
    def emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Emit event
        
        Args:
            event_type: Event type
            data: Event data
        """
        event = {
            "type": event_type,
            "timestamp": time.time(),
            "data": data
        }
        self.events.append(event)
        print(f"Event emitted: {event_type}")
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get contract state
        
        Returns:
            Contract state
        """
        return self.state
    
    def set_state(self, key: str, value: Any) -> None:
        """
        Set contract state
        
        Args:
            key: State key
            value: State value
        """
        self.state[key] = value
        
        # Record state change
        self.events.append({
            "type": "state_change",
            "key": key,
            "timestamp": time.time()
        })
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert contract to dictionary
        
        Returns:
            Dictionary representing contract
        """
        return {
            "contract_id": self.contract_id,
            "owner": self.owner,
            "state": self.state,
            "events": self.events,
            "created_at": self.created_at
        }
    
    def __str__(self) -> str:
        """String representation"""
        return json.dumps(self.to_dict(), indent=2)


class DataMarketContract(SmartContract):
    """Data Market Smart Contract"""
    
    def __init__(self, contract_id: str, owner: str, blockchain: Blockchain):
        """
        Initialize Data Market Contract
        
        Args:
            contract_id: Contract ID
            owner: Contract owner address
            blockchain: Blockchain instance
        """
        super().__init__(contract_id, owner, blockchain)
        
        # Initialize state
        self.state = {
            "data_listings": {},  # Data listings
            "purchases": {},      # Purchase records
            "fees": 0.05,         # Platform fee ratio
            "total_transactions": 0,
            "total_volume": 0.0
        }
        
        # Register contract functions
        self.register_function("list_data", self._list_data)
        self.register_function("purchase_data", self._purchase_data)
        self.register_function("update_data", self._update_data)
        self.register_function("remove_data", self._remove_data)
        self.register_function("get_data_info", self._get_data_info)
    
    def _list_data(self, caller: str, data_id: str, metadata: Dict[str, Any], 
                  price: float, access_control: Dict[str, Any]) -> bool:
        """
        List data for sale
        
        Args:
            caller: Caller address
            data_id: Data ID
            metadata: Data metadata
            price: Data price
            access_control: Access control information
            
        Returns:
            Operation success
        """
        if data_id in self.state["data_listings"]:
            raise ValueError(f"Data ID {data_id} already exists")
        
        # Create data listing
        self.state["data_listings"][data_id] = {
            "owner": caller,
            "metadata": metadata,
            "price": price,
            "access_control": access_control,
            "created_at": time.time(),
            "updated_at": time.time(),
            "active": True
        }
        
        # Emit event
        self.emit_event("DataListed", {
            "data_id": data_id,
            "owner": caller,
            "price": price
        })
        
        return True
    
    def _purchase_data(self, caller: str, data_id: str) -> Dict[str, Any]:
        """
        Purchase data
        
        Args:
            caller: Caller address
            data_id: Data ID to purchase
            
        Returns:
            Purchase info
        """
        if data_id not in self.state["data_listings"]:
            raise ValueError(f"Data ID {data_id} does not exist")
        
        data_listing = self.state["data_listings"][data_id]
        
        if not data_listing["active"]:
            raise ValueError(f"Data ID {data_id} is not available")
        
        # Check access control
        access_control = data_listing["access_control"]
        if "whitelist" in access_control and caller not in access_control["whitelist"]:
            raise ValueError(f"Address {caller} does not have purchase permission")
        
        # Calculate fee
        price = data_listing["price"]
        fee = price * self.state["fees"]
        seller_amount = price - fee
        
        # Check balance
        if self.blockchain.get_balance(caller) < price:
            raise ValueError(f"Insufficient balance, need {price}")
        
        # Create transaction
        purchase_id = f"purchase_{int(time.time())}_{data_id}"
        
        # Transfer to seller
        self.blockchain.create_transaction(
            sender=caller,
            recipient=data_listing["owner"],
            amount=seller_amount,
            transaction_type="data_purchase",
            data={"data_id": data_id, "purchase_id": purchase_id}
        )
        
        # Transfer platform fee
        self.blockchain.create_transaction(
            sender=caller,
            recipient=self.owner,
            amount=fee,
            transaction_type="platform_fee",
            data={"data_id": data_id, "purchase_id": purchase_id}
        )
        
        # Record purchase
        purchase_info = {
            "purchase_id": purchase_id,
            "buyer": caller,
            "seller": data_listing["owner"],
            "data_id": data_id,
            "price": price,
            "fee": fee,
            "timestamp": time.time(),
            "status": "completed"
        }
        
        self.state["purchases"][purchase_id] = purchase_info
        self.state["total_transactions"] += 1
        self.state["total_volume"] += price
        
        # Emit event
        self.emit_event("DataPurchased", {
            "purchase_id": purchase_id,
            "data_id": data_id,
            "buyer": caller,
            "seller": data_listing["owner"],
            "price": price
        })
        
        return purchase_info
    
    def _update_data(self, caller: str, data_id: str, 
                    metadata: Optional[Dict[str, Any]] = None, 
                    price: Optional[float] = None,
                    access_control: Optional[Dict[str, Any]] = None,
                    active: Optional[bool] = None) -> bool:
        """
        Update data information
        
        Args:
            caller: Caller address
            data_id: Data ID
            metadata: New metadata
            price: New price
            access_control: New access control
            active: Activation
            
        Returns:
            Operation success
        """
        if data_id not in self.state["data_listings"]:
            raise ValueError(f"Data ID {data_id} does not exist")
        
        data_listing = self.state["data_listings"][data_id]
        
        # Check ownership
        if data_listing["owner"] != caller:
            raise ValueError(f"Only the owner can update data")
        
        # Update fields
        if metadata is not None:
            data_listing["metadata"] = metadata
        
        if price is not None:
            data_listing["price"] = price
        
        if access_control is not None:
            data_listing["access_control"] = access_control
        
        if active is not None:
            data_listing["active"] = active
        
        data_listing["updated_at"] = time.time()
        
        # Emit event
        self.emit_event("DataUpdated", {
            "data_id": data_id,
            "owner": caller
        })
        
        return True
    
    def _remove_data(self, caller: str, data_id: str) -> bool:
        """
        Remove data
        
        Args:
            caller: Caller address
            data_id: Data ID
            
        Returns:
            Operation success
        """
        if data_id not in self.state["data_listings"]:
            raise ValueError(f"Data ID {data_id} does not exist")
        
        data_listing = self.state["data_listings"][data_id]
        
        # Check ownership
        if data_listing["owner"] != caller and caller != self.owner:
            raise ValueError(f"Only the owner or platform can remove data")
        
        # Mark data as inactive
        data_listing["active"] = False
        data_listing["updated_at"] = time.time()
        
        # Emit event
        self.emit_event("DataRemoved", {
            "data_id": data_id,
            "owner": data_listing["owner"]
        })
        
        return True
    
    def _get_data_info(self, caller: str, data_id: str) -> Dict[str, Any]:
        """
        Get data information
        
        Args:
            caller: Caller address
            data_id: Data ID
            
        Returns:
            Data information
        """
        if data_id not in self.state["data_listings"]:
            raise ValueError(f"Data ID {data_id} does not exist")
        
        return self.state["data_listings"][data_id]


class FederatedLearningContract(SmartContract):
    """Federated Learning Smart Contract"""
    
    def __init__(self, contract_id: str, owner: str, blockchain: Blockchain):
        """
        Initialize Federated Learning Contract
        
        Args:
            contract_id: Contract ID
            owner: Contract owner address
            blockchain: Blockchain instance
        """
        super().__init__(contract_id, owner, blockchain)
        
        # Initialize state
        self.state = {
            "models": {},           # Model information
            "training_rounds": {},  # Training rounds
            "participants": {},     # Participants
            "rewards": {},          # Reward information
            "total_rewards": 0.0
        }
        
        # Register contract functions
        self.register_function("register_model", self._register_model)
        self.register_function("register_participant", self._register_participant)
        self.register_function("start_training_round", self._start_training_round)
        self.register_function("submit_update", self._submit_update)
        self.register_function("complete_round", self._complete_round)
        self.register_function("get_model_info", self._get_model_info)
    
    def _register_model(self, caller: str, model_id: str, metadata: Dict[str, Any], 
                       initial_parameters: Dict[str, Any]) -> bool:
        """
        Register model
        
        Args:
            caller: Caller address
            model_id: Model ID
            metadata: Model metadata
            initial_parameters: Initial parameters
            
        Returns:
            Operation success
        """
        if model_id in self.state["models"]:
            raise ValueError(f"Model ID {model_id} already exists")
        
        # Create model
        self.state["models"][model_id] = {
            "owner": caller,
            "metadata": metadata,
            "current_parameters": initial_parameters,
            "version": 1,
            "created_at": time.time(),
            "updated_at": time.time(),
            "active": True,
            "total_rounds": 0,
            "total_updates": 0
        }
        
        # Emit event
        self.emit_event("ModelRegistered", {
            "model_id": model_id,
            "owner": caller
        })
        
        return True
    
    def _register_participant(self, caller: str, participant_id: str, 
                             metadata: Dict[str, Any]) -> bool:
        """
        Register participant
        
        Args:
            caller: Caller address
            participant_id: Participant ID
            metadata: Participant metadata
            
        Returns:
            Operation success
        """
        if participant_id in self.state["participants"]:
            raise ValueError(f"Participant ID {participant_id} already exists")
        
        # Create participant
        self.state["participants"][participant_id] = {
            "address": caller,
            "metadata": metadata,
            "created_at": time.time(),
            "active": True,
            "total_contributions": 0,
            "reputation": 0.0
        }
        
        # Emit event
        self.emit_event("ParticipantRegistered", {
            "participant_id": participant_id,
            "address": caller
        })
        
        return True
    
    def _start_training_round(self, caller: str, model_id: str, 
                             round_id: str, config: Dict[str, Any]) -> bool:
        """
        Start training round
        
        Args:
            caller: Caller address
            model_id: Model ID
            round_id: Round ID
            config: Round configuration
            
        Returns:
            Operation success
        """
        if model_id not in self.state["models"]:
            raise ValueError(f"Model ID {model_id} does not exist")
        
        model = self.state["models"][model_id]
        
        # Check permission
        if model["owner"] != caller:
            raise ValueError(f"Only model owner can start training round")
        
        if round_id in self.state["training_rounds"]:
            raise ValueError(f"Round ID {round_id} already exists")
        
        # Create training round
        self.state["training_rounds"][round_id] = {
            "model_id": model_id,
            "config": config,
            "status": "active",
            "started_at": time.time(),
            "completed_at": None,
            "participants": [],
            "updates": {},
            "aggregated_update": None,
            "rewards": {}
        }
        
        # Update model information
        model["total_rounds"] += 1
        model["updated_at"] = time.time()
        
        # Emit event
        self.emit_event("TrainingRoundStarted", {
            "round_id": round_id,
            "model_id": model_id,
            "owner": caller
        })
        
        return True
    
    def _submit_update(self, caller: str, round_id: str, participant_id: str, 
                      update: Dict[str, Any], metadata: Dict[str, Any]) -> bool:
        """
        Submit model update
        
        Args:
            caller: Caller address
            round_id: Round ID
            participant_id: Participant ID
            update: Model update
            metadata: Update metadata
            
        Returns:
            Operation success
        """
        if round_id not in self.state["training_rounds"]:
            raise ValueError(f"Round ID {round_id} does not exist")
        
        if participant_id not in self.state["participants"]:
            raise ValueError(f"Participant ID {participant_id} does not exist")
        
        training_round = self.state["training_rounds"][round_id]
        participant = self.state["participants"][participant_id]
        
        # Check round status
        if training_round["status"] != "active":
            raise ValueError(f"Round {round_id} is not active")
        
        # Check participant address
        if participant["address"] != caller:
            raise ValueError(f"Only participant themselves can submit update")
        
        # Check if already submitted
        if participant_id in training_round["updates"]:
            raise ValueError(f"Participant {participant_id} has already submitted update")
        
        # Record update
        training_round["updates"][participant_id] = {
            "update": update,
            "metadata": metadata,
            "timestamp": time.time()
        }
        
        # Add participant to round participants list
        if participant_id not in training_round["participants"]:
            training_round["participants"].append(participant_id)
        
        # Update participant information
        participant["total_contributions"] += 1
        
        # Update model information
        model_id = training_round["model_id"]
        self.state["models"][model_id]["total_updates"] += 1
        
        # Emit event
        self.emit_event("UpdateSubmitted", {
            "round_id": round_id,
            "model_id": model_id,
            "participant_id": participant_id
        })
        
        return True
    
    def _complete_round(self, caller: str, round_id: str, 
                       aggregated_update: Dict[str, Any], 
                       rewards: Dict[str, float]) -> bool:
        """
        Complete training round
        
        Args:
            caller: Caller address
            round_id: Round ID
            aggregated_update: Aggregated update
            rewards: Participant rewards
            
        Returns:
            Operation success
        """
        if round_id not in self.state["training_rounds"]:
            raise ValueError(f"Round ID {round_id} does not exist")
        
        training_round = self.state["training_rounds"][round_id]
        model_id = training_round["model_id"]
        
        if model_id not in self.state["models"]:
            raise ValueError(f"Model ID {model_id} does not exist")
        
        model = self.state["models"][model_id]
        
        # Check permission
        if model["owner"] != caller:
            raise ValueError(f"Only model owner can complete training round")
        
        # Check round status
        if training_round["status"] != "active":
            raise ValueError(f"Round {round_id} is not active")
        
        # Update round information
        training_round["status"] = "completed"
        training_round["completed_at"] = time.time()
        training_round["aggregated_update"] = aggregated_update
        training_round["rewards"] = rewards
        
        # Update model parameters
        model["current_parameters"] = aggregated_update
        model["version"] += 1
        model["updated_at"] = time.time()
        
        # Distribute rewards
        total_rewards = 0.0
        for participant_id, reward in rewards.items():
            if participant_id not in self.state["participants"]:
                continue
            
            participant = self.state["participants"][participant_id]
            
            # Record reward
            if participant_id not in self.state["rewards"]:
                self.state["rewards"][participant_id] = 0.0
            
            self.state["rewards"][participant_id] += reward
            total_rewards += reward
            
            # Update reputation
            participant["reputation"] += reward * 0.01
            
            # Create reward transaction
            self.blockchain.create_transaction(
                sender=self.owner,
                recipient=participant["address"],
                amount=reward,
                transaction_type="fl_reward",
                data={"round_id": round_id, "model_id": model_id}
            )
        
        # Update total rewards
        self.state["total_rewards"] += total_rewards
        
        # Emit event
        self.emit_event("TrainingRoundCompleted", {
            "round_id": round_id,
            "model_id": model_id,
            "participants": training_round["participants"],
            "total_rewards": total_rewards
        })
        
        return True
    
    def _get_model_info(self, caller: str, model_id: str) -> Dict[str, Any]:
        """
        Get model information
        
        Args:
            caller: Caller address
            model_id: Model ID
            
        Returns:
            Model information
        """
        if model_id not in self.state["models"]:
            raise ValueError(f"Model ID {model_id} does not exist")
        
        return self.state["models"][model_id]


# Example usage
if __name__ == "__main__":
    from blockchain import Blockchain
    
    # Create blockchain
    blockchain = Blockchain(difficulty=2)
    
    # Create data market contract
    data_market = DataMarketContract("data_market_v1", "platform_address", blockchain)
    
    # Create federated learning contract
    fl_contract = FederatedLearningContract("fl_contract_v1", "platform_address", blockchain)
    
    # Add some initial balances
    blockchain.create_transaction("system", "user1", 1000)
    blockchain.create_transaction("system", "user2", 1000)
    blockchain.create_transaction("system", "platform_address", 5000)
    blockchain.mine_pending_transactions("miner1")
    
    # List data
    data_market.call("list_data", "user1", "dataset1", {
        "name": "Medical Dataset",
        "description": "Anonymized medical records",
        "size": "2.5GB",
        "format": "CSV"
    }, 100, {"whitelist": ["user2"]})
    
    # Purchase data
    purchase = data_market.call("purchase_data", "user2", "dataset1")
    print(f"Purchase info: {purchase}")
    
    # Register model
    fl_contract.call("register_model", "platform_address", "model1", {
        "name": "Medical Prediction Model",
        "description": "Predict patient risk",
        "architecture": "CNN"
    }, {"weights": "initial weights..."})
    
    # Register participant
    fl_contract.call("register_participant", "user1", "participant1", {
        "name": "Hospital 1",
        "data_description": "1000 patient records"
    })
    
    # Start training round
    fl_contract.call("start_training_round", "platform_address", "model1", "round1", {
        "min_participants": 1,
        "max_participants": 10,
        "epochs": 5,
        "batch_size": 32
    })
    
    # Submit update
    fl_contract.call("submit_update", "user1", "round1", "participant1", {
        "weights": "updated weights..."
    }, {"accuracy": 0.85, "loss": 0.12})
    
    # Complete round
    fl_contract.call("complete_round", "platform_address", "round1", {
        "weights": "aggregated weights..."
    }, {"participant1": 50.0})
    
    # Mine all pending transactions
    blockchain.mine_pending_transactions("miner1")
    
    # Check balances
    print(f"User1 balance: {blockchain.get_balance('user1')}")
    print(f"User2 balance: {blockchain.get_balance('user2')}")
    print(f"Platform balance: {blockchain.get_balance('platform_address')}")
    
    # Check contract states
    print("\nData market contract state:")
    print(json.dumps(data_market.get_state(), indent=2))
    
    print("\nFederated learning contract state:")
    print(json.dumps(fl_contract.get_state(), indent=2)) 