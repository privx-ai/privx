#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data Market Core Module
Implements blockchain-based data trading market functionality
"""

import json
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

class DataMarket:
    """Data Market class, implements core functionality of data trading market"""
    
    def __init__(self, contract):
        """
        Initialize data market
        
        Args:
            contract: Data market smart contract instance
        """
        self.contract = contract
        self.providers = {}  # Data provider information
        self.consumers = {}  # Data consumer information
        self.datasets = {}   # Dataset information
        self.transactions = []  # Transaction records
        logger.info("Data market initialized")
    
    def register_provider(self, provider_id: str, name: str, metadata: Optional[Dict] = None) -> bool:
        """
        Register data provider
        
        Args:
            provider_id: Provider ID
            name: Provider name
            metadata: Provider metadata
            
        Returns:
            bool: Registration success
        """
        if provider_id in self.providers:
            logger.warning(f"Provider {provider_id} already exists")
            return False
        
        provider_info = {
            "id": provider_id,
            "name": name,
            "metadata": metadata or {},
            "reputation": 5.0,  # Initial reputation score
            "datasets": [],
            "registration_time": datetime.now().isoformat(),
            "status": "active"
        }
        
        # Call smart contract to register provider
        tx_hash = self.contract.register_provider(provider_id, json.dumps(provider_info))
        
        if tx_hash:
            self.providers[provider_id] = provider_info
            logger.info(f"Provider {provider_id} registered successfully, transaction hash: {tx_hash}")
            return True
        else:
            logger.error(f"Provider {provider_id} registration failed")
            return False
    
    def register_consumer(self, consumer_id: str, name: str, metadata: Optional[Dict] = None) -> bool:
        """
        Register data consumer
        
        Args:
            consumer_id: Consumer ID
            name: Consumer name
            metadata: Consumer metadata
            
        Returns:
            bool: Registration success
        """
        if consumer_id in self.consumers:
            logger.warning(f"Consumer {consumer_id} already exists")
            return False
        
        consumer_info = {
            "id": consumer_id,
            "name": name,
            "metadata": metadata or {},
            "balance": 1000,  # Initial balance
            "purchased_datasets": [],
            "registration_time": datetime.now().isoformat(),
            "status": "active"
        }
        
        # Call smart contract to register consumer
        tx_hash = self.contract.register_consumer(consumer_id, json.dumps(consumer_info))
        
        if tx_hash:
            self.consumers[consumer_id] = consumer_info
            logger.info(f"Consumer {consumer_id} registered successfully, transaction hash: {tx_hash}")
            return True
        else:
            logger.error(f"Consumer {consumer_id} registration failed")
            return False
    
    def list_dataset(self, provider_id: str, dataset_id: str, name: str, 
                    description: str, price: float, sample_size: int,
                    metadata: Optional[Dict] = None) -> bool:
        """
        List dataset
        
        Args:
            provider_id: Provider ID
            dataset_id: Dataset ID
            name: Dataset name
            description: Dataset description
            price: Dataset price
            sample_size: Sample count
            metadata: Dataset metadata
            
        Returns:
            bool: Listing success
        """
        if provider_id not in self.providers:
            logger.error(f"Provider {provider_id} does not exist")
            return False
        
        if dataset_id in self.datasets:
            logger.warning(f"Dataset {dataset_id} already exists")
            return False
        
        dataset_info = {
            "id": dataset_id,
            "provider_id": provider_id,
            "name": name,
            "description": description,
            "price": price,
            "sample_size": sample_size,
            "metadata": metadata or {},
            "creation_time": datetime.now().isoformat(),
            "status": "active",
            "rating": 0,
            "rating_count": 0,
            "purchase_count": 0
        }
        
        # Call smart contract to list dataset
        tx_hash = self.contract.list_dataset(
            provider_id, 
            dataset_id, 
            json.dumps(dataset_info)
        )
        
        if tx_hash:
            self.datasets[dataset_id] = dataset_info
            self.providers[provider_id]["datasets"].append(dataset_id)
            logger.info(f"Dataset {dataset_id} listed successfully, transaction hash: {tx_hash}")
            return True
        else:
            logger.error(f"Dataset {dataset_id} listing failed")
            return False
    
    def purchase_dataset(self, consumer_id: str, dataset_id: str) -> bool:
        """
        Purchase dataset
        
        Args:
            consumer_id: Consumer ID
            dataset_id: Dataset ID
            
        Returns:
            bool: Purchase success
        """
        if consumer_id not in self.consumers:
            logger.error(f"Consumer {consumer_id} does not exist")
            return False
        
        if dataset_id not in self.datasets:
            logger.error(f"Dataset {dataset_id} does not exist")
            return False
        
        dataset = self.datasets[dataset_id]
        consumer = self.consumers[consumer_id]
        
        # Check balance
        if consumer["balance"] < dataset["price"]:
            logger.error(f"Consumer {consumer_id} balance insufficient")
            return False
        
        # Call smart contract to purchase dataset
        tx_hash = self.contract.purchase_dataset(
            consumer_id,
            dataset_id,
            dataset["provider_id"],
            dataset["price"]
        )
        
        if tx_hash:
            # Update consumer information
            consumer["balance"] -= dataset["price"]
            consumer["purchased_datasets"].append(dataset_id)
            
            # Update dataset information
            dataset["purchase_count"] += 1
            
            # Record transaction
            transaction = {
                "id": str(uuid.uuid4()),
                "consumer_id": consumer_id,
                "provider_id": dataset["provider_id"],
                "dataset_id": dataset_id,
                "price": dataset["price"],
                "timestamp": datetime.now().isoformat(),
                "tx_hash": tx_hash
            }
            self.transactions.append(transaction)
            
            logger.info(f"Consumer {consumer_id} purchased dataset {dataset_id} successfully, transaction hash: {tx_hash}")
            return True
        else:
            logger.error(f"Consumer {consumer_id} purchase dataset {dataset_id} failed")
            return False
    
    def rate_dataset(self, consumer_id: str, dataset_id: str, rating: float) -> bool:
        """
        Rate dataset
        
        Args:
            consumer_id: Consumer ID
            dataset_id: Dataset ID
            rating: Rating (1-5)
            
        Returns:
            bool: Rating success
        """
        if consumer_id not in self.consumers:
            logger.error(f"Consumer {consumer_id} does not exist")
            return False
        
        if dataset_id not in self.datasets:
            logger.error(f"Dataset {dataset_id} does not exist")
            return False
        
        consumer = self.consumers[consumer_id]
        
        # Check if purchased
        if dataset_id not in consumer["purchased_datasets"]:
            logger.error(f"Consumer {consumer_id} not purchased dataset {dataset_id}")
            return False
        
        # Check rating range
        if not (1 <= rating <= 5):
            logger.error(f"Rating {rating} out of range (1-5)")
            return False
        
        dataset = self.datasets[dataset_id]
        
        # Update dataset rating
        old_rating = dataset["rating"]
        old_count = dataset["rating_count"]
        new_count = old_count + 1
        new_rating = (old_rating * old_count + rating) / new_count
        
        dataset["rating"] = new_rating
        dataset["rating_count"] = new_count
        
        # Call smart contract to record rating
        tx_hash = self.contract.rate_dataset(
            consumer_id,
            dataset_id,
            rating
        )
        
        if tx_hash:
            logger.info(f"Consumer {consumer_id} rated dataset {dataset_id} successfully, rating: {rating}")
            return True
        else:
            logger.error(f"Consumer {consumer_id} rated dataset {dataset_id} failed")
            return False
    
    def get_dataset_info(self, dataset_id: str) -> Optional[Dict]:
        """
        Get dataset information
        
        Args:
            dataset_id: Dataset ID
            
        Returns:
            Dict: Dataset information
        """
        if dataset_id not in self.datasets:
            logger.warning(f"Dataset {dataset_id} does not exist")
            return None
        
        return self.datasets[dataset_id]
    
    def get_provider_datasets(self, provider_id: str) -> List[Dict]:
        """
        Get all datasets of a provider
        
        Args:
            provider_id: Provider ID
            
        Returns:
            List[Dict]: Dataset list
        """
        if provider_id not in self.providers:
            logger.warning(f"Provider {provider_id} does not exist")
            return []
        
        provider = self.providers[provider_id]
        return [self.datasets[dataset_id] for dataset_id in provider["datasets"] 
                if dataset_id in self.datasets]
    
    def get_consumer_purchased_datasets(self, consumer_id: str) -> List[Dict]:
        """
        Get all purchased datasets of a consumer
        
        Args:
            consumer_id: Consumer ID
            
        Returns:
            List[Dict]: Dataset list
        """
        if consumer_id not in self.consumers:
            logger.warning(f"Consumer {consumer_id} does not exist")
            return []
        
        consumer = self.consumers[consumer_id]
        return [self.datasets[dataset_id] for dataset_id in consumer["purchased_datasets"] 
                if dataset_id in self.datasets]
    
    def search_datasets(self, query: str, min_price: Optional[float] = None, 
                       max_price: Optional[float] = None, 
                       min_rating: Optional[float] = None) -> List[Dict]:
        """
        Search dataset
        
        Args:
            query: Search keyword
            min_price: Minimum price
            max_price: Maximum price
            min_rating: Minimum rating
            
        Returns:
            List[Dict]: Dataset list that meets the conditions
        """
        results = []
        
        for dataset in self.datasets.values():
            # Keyword matching
            if query.lower() in dataset["name"].lower() or query.lower() in dataset["description"].lower():
                # Price filtering
                if min_price is not None and dataset["price"] < min_price:
                    continue
                if max_price is not None and dataset["price"] > max_price:
                    continue
                # Rating filtering
                if min_rating is not None and dataset["rating"] < min_rating:
                    continue
                
                results.append(dataset)
        
        return results
    
    def get_market_statistics(self) -> Dict[str, Any]:
        """
        Get market statistics
        
        Returns:
            Dict: Market statistics
        """
        total_providers = len(self.providers)
        total_consumers = len(self.consumers)
        total_datasets = len(self.datasets)
        total_transactions = len(self.transactions)
        
        # Calculate total transaction value
        total_value = sum(tx["price"] for tx in self.transactions)
        
        # Calculate average rating
        rated_datasets = [d for d in self.datasets.values() if d["rating_count"] > 0]
        avg_rating = sum(d["rating"] for d in rated_datasets) / len(rated_datasets) if rated_datasets else 0
        
        return {
            "total_providers": total_providers,
            "total_consumers": total_consumers,
            "total_datasets": total_datasets,
            "total_transactions": total_transactions,
            "total_value": total_value,
            "avg_rating": avg_rating,
            "timestamp": datetime.now().isoformat()
        } 