#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data Market Demo
Demonstrates basic functionality of the data market
"""

import os
import sys
import logging
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root directory to system path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.blockchain.blockchain import Blockchain
from src.blockchain.smart_contract import DataMarketContract
from src.data_market.market import DataMarket
from src.data_market.data_handler import DataHandler
from src.privacy.differential_privacy import DifferentialPrivacy

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_data_market():
    """Setup data market"""
    # Initialize blockchain
    blockchain = Blockchain()
    logger.info(f"Blockchain initialized, current blocks: {len(blockchain.chain)}")
    
    # Initialize smart contract
    contract = DataMarketContract(blockchain)
    logger.info(f"Smart contract initialized, contract address: {contract.address}")
    
    # Initialize data market
    market = DataMarket(contract)
    logger.info(f"Data market initialized")
    
    return market, blockchain, contract

def register_participants(market):
    """Register participants"""
    # Register data providers
    providers = []
    for i in range(5):
        provider_id = f"provider_{i}"
        market.register_provider(provider_id, f"Data Provider {i}")
        providers.append(provider_id)
        logger.info(f"Data provider {provider_id} registered successfully")
    
    # Register data consumers
    consumers = []
    for i in range(3):
        consumer_id = f"consumer_{i}"
        market.register_consumer(consumer_id, f"Data Consumer {i}")
        consumers.append(consumer_id)
        logger.info(f"Data consumer {consumer_id} registered successfully")
    
    return providers, consumers

def generate_sample_datasets(data_handler):
    """Generate sample datasets"""
    datasets = []
    
    # Generate user behavior dataset
    user_behavior_columns = [
        {"name": "user_id", "type": "int", "min": 1, "max": 10000},
        {"name": "age", "type": "int", "min": 18, "max": 80},
        {"name": "gender", "type": "category", "categories": ["M", "F"]},
        {"name": "location", "type": "category", "categories": ["Beijing", "Shanghai", "Guangzhou", "Shenzhen", "Hangzhou"]},
        {"name": "purchase_amount", "type": "float", "min": 0, "max": 10000},
        {"name": "visit_frequency", "type": "int", "min": 1, "max": 100},
        {"name": "is_vip", "type": "bool"}
    ]
    user_behavior_df = data_handler.generate_sample_dataset(
        "user_behavior", 
        rows=5000, 
        columns=user_behavior_columns
    )
    datasets.append(("user_behavior", user_behavior_df, "User Behavior Dataset", 500))
    
    # Generate medical dataset
    medical_columns = [
        {"name": "patient_id", "type": "int", "min": 1, "max": 5000},
        {"name": "age", "type": "int", "min": 1, "max": 100},
        {"name": "gender", "type": "category", "categories": ["M", "F"]},
        {"name": "blood_pressure", "type": "float", "min": 80, "max": 200},
        {"name": "heart_rate", "type": "int", "min": 40, "max": 180},
        {"name": "cholesterol", "type": "float", "min": 100, "max": 300},
        {"name": "diabetes", "type": "bool"},
        {"name": "smoker", "type": "bool"}
    ]
    medical_df = data_handler.generate_sample_dataset(
        "medical", 
        rows=3000, 
        columns=medical_columns
    )
    datasets.append(("medical", medical_df, "Medical Health Dataset", 800))
    
    # Generate financial dataset
    financial_columns = [
        {"name": "customer_id", "type": "int", "min": 1, "max": 8000},
        {"name": "income", "type": "float", "min": 3000, "max": 50000},
        {"name": "credit_score", "type": "int", "min": 300, "max": 850},
        {"name": "loan_amount", "type": "float", "min": 0, "max": 1000000},
        {"name": "has_mortgage", "type": "bool"},
        {"name": "has_credit_card", "type": "bool"},
        {"name": "payment_history", "type": "int", "min": 1, "max": 10}
    ]
    financial_df = data_handler.generate_sample_dataset(
        "financial", 
        rows=4000, 
        columns=financial_columns
    )
    datasets.append(("financial", financial_df, "Financial Dataset", 1000))
    
    # Generate location dataset
    location_columns = [
        {"name": "device_id", "type": "int", "min": 1, "max": 10000},
        {"name": "latitude", "type": "float", "min": 18, "max": 53},
        {"name": "longitude", "type": "float", "min": 73, "max": 135},
        {"name": "timestamp", "type": "date", "start": "2023-01-01", "end": "2023-12-31"},
        {"name": "speed", "type": "float", "min": 0, "max": 120},
        {"name": "altitude", "type": "float", "min": 0, "max": 5000},
        {"name": "device_type", "type": "category", "categories": ["Mobile", "Tablet", "Wearable"]}
    ]
    location_df = data_handler.generate_sample_dataset(
        "location", 
        rows=6000, 
        columns=location_columns
    )
    datasets.append(("location", location_df, "Location Dataset", 600))
    
    # Generate social network dataset
    social_columns = [
        {"name": "user_id", "type": "int", "min": 1, "max": 10000},
        {"name": "friends_count", "type": "int", "min": 0, "max": 5000},
        {"name": "posts_count", "type": "int", "min": 0, "max": 1000},
        {"name": "likes_received", "type": "int", "min": 0, "max": 10000},
        {"name": "comments_received", "type": "int", "min": 0, "max": 5000},
        {"name": "active_days", "type": "int", "min": 1, "max": 365},
        {"name": "is_verified", "type": "bool"}
    ]
    social_df = data_handler.generate_sample_dataset(
        "social", 
        rows=7000, 
        columns=social_columns
    )
    datasets.append(("social", social_df, "Social Network Dataset", 700))
    
    return datasets

def list_datasets_on_market(market, providers, datasets):
    """List datasets on market"""
    dataset_ids = []
    
    for i, (dataset_id, df, description, price) in enumerate(datasets):
        # Select provider
        provider_id = providers[i % len(providers)]
        
        # List dataset
        market.list_dataset(
            provider_id=provider_id,
            dataset_id=dataset_id,
            name=f"{description}",
            description=f"This is a dataset containing {df.shape[0]} rows and {df.shape[1]} columns of {description}",
            price=price,
            sample_size=df.shape[0]
        )
        dataset_ids.append(dataset_id)
        logger.info(f"Dataset {dataset_id} listed successfully, provider: {provider_id}, price: {price}")
    
    return dataset_ids

def simulate_market_transactions(market, consumers, dataset_ids):
    """Simulate market transactions"""
    transactions = []
    
    # Each consumer purchases some datasets
    for i, consumer_id in enumerate(consumers):
        # Select datasets to purchase
        purchase_count = min(3, len(dataset_ids))
        purchase_datasets = dataset_ids[i:i+purchase_count]
        
        for dataset_id in purchase_datasets:
            # Purchase dataset
            success = market.purchase_dataset(consumer_id, dataset_id)
            
            if success:
                # Record transaction
                dataset_info = market.get_dataset_info(dataset_id)
                transactions.append({
                    "consumer_id": consumer_id,
                    "dataset_id": dataset_id,
                    "price": dataset_info["price"],
                    "provider_id": dataset_info["provider_id"]
                })
                
                # Rate dataset
                rating = 4.0 + (i % 2)  # 4.0 or 5.0
                market.rate_dataset(consumer_id, dataset_id, rating)
                logger.info(f"Consumer {consumer_id} purchased dataset {dataset_id} successfully, rating: {rating}")
            else:
                logger.warning(f"Consumer {consumer_id} failed to purchase dataset {dataset_id}")
    
    return transactions

def apply_privacy_to_dataset(data_handler, differential_privacy):
    """Apply differential privacy protection to data"""
    # Load medical dataset
    medical_df = data_handler.load_dataset("medical")
    
    # Define sensitive columns
    sensitive_columns = ["blood_pressure", "heart_rate", "cholesterol"]
    
    # Set sensitivities
    sensitivities = {
        "blood_pressure": 10.0,  # Blood pressure sensitivity
        "heart_rate": 5.0,       # Heart rate sensitivity
        "cholesterol": 20.0      # Cholesterol sensitivity
    }
    
    # Apply differential privacy
    private_df = differential_privacy.privatize_dataframe(
        medical_df,
        numeric_columns=sensitive_columns,
        sensitivities=sensitivities,
        mechanism='gaussian'
    )
    
    # Save processed dataset
    data_handler.save_dataset(private_df, "medical_private")
    
    # Compare statistics of original and private data
    logger.info("Original data statistics:")
    for col in sensitive_columns:
        logger.info(f"{col} - Mean: {medical_df[col].mean():.2f}, Std: {medical_df[col].std():.2f}")
    
    logger.info("Private data statistics:")
    for col in sensitive_columns:
        logger.info(f"{col} - Mean: {private_df[col].mean():.2f}, Std: {private_df[col].std():.2f}")
    
    return medical_df, private_df

def visualize_market_statistics(market, transactions):
    """Visualize market statistics"""
    # Get market statistics
    stats = market.get_market_statistics()
    
    # Create charts directory
    charts_dir = ROOT_DIR / "examples" / "charts"
    charts_dir.mkdir(exist_ok=True)
    
    # 1. Transaction volume bar chart
    plt.figure(figsize=(10, 6))
    dataset_ids = [tx["dataset_id"] for tx in transactions]
    dataset_counts = pd.Series(dataset_ids).value_counts()
    dataset_counts.plot(kind='bar', color='skyblue')
    plt.title('Dataset Transaction Volume')
    plt.xlabel('Dataset ID')
    plt.ylabel('Transaction Count')
    plt.tight_layout()
    plt.savefig(charts_dir / "dataset_transactions.png")
    
    # 2. Provider income pie chart
    plt.figure(figsize=(10, 6))
    provider_income = {}
    for tx in transactions:
        provider_id = tx["provider_id"]
        price = tx["price"]
        if provider_id in provider_income:
            provider_income[provider_id] += price
        else:
            provider_income[provider_id] = price
    
    plt.pie(provider_income.values(), labels=provider_income.keys(), autopct='%1.1f%%')
    plt.title('Provider Income Distribution')
    plt.tight_layout()
    plt.savefig(charts_dir / "provider_income.png")
    
    # 3. Dataset price comparison
    plt.figure(figsize=(10, 6))
    dataset_prices = {}
    for dataset_id in market.datasets:
        dataset_prices[dataset_id] = market.datasets[dataset_id]["price"]
    
    plt.bar(dataset_prices.keys(), dataset_prices.values(), color='lightgreen')
    plt.title('Dataset Price Comparison')
    plt.xlabel('Dataset ID')
    plt.ylabel('Price')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(charts_dir / "dataset_prices.png")
    
    logger.info(f"Market statistics charts saved to {charts_dir}")

def visualize_privacy_comparison(original_df, private_df):
    """Visualize comparison between original and private data"""
    # Create charts directory
    charts_dir = ROOT_DIR / "examples" / "charts"
    charts_dir.mkdir(exist_ok=True)
    
    # Select columns to visualize
    columns = ["blood_pressure", "heart_rate", "cholesterol"]
    
    for col in columns:
        plt.figure(figsize=(12, 6))
        
        # Plot histogram
        plt.subplot(1, 2, 1)
        plt.hist(original_df[col], bins=30, alpha=0.5, label='Original Data')
        plt.hist(private_df[col], bins=30, alpha=0.5, label='Differential Privacy Data')
        plt.title(f'{col} Distribution Comparison')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.legend()
        
        # Plot box plot
        plt.subplot(1, 2, 2)
        boxplot_data = [original_df[col], private_df[col]]
        plt.boxplot(boxplot_data, labels=['Original Data', 'Differential Privacy Data'])
        plt.title(f'{col} Box Plot Comparison')
        plt.ylabel(col)
        
        plt.tight_layout()
        plt.savefig(charts_dir / f"privacy_comparison_{col}.png")
    
    logger.info(f"Privacy comparison charts saved to {charts_dir}")

def main():
    """Main function"""
    logger.info("Starting data market demo...")
    
    # Setup data market
    market, blockchain, contract = setup_data_market()
    
    # Register participants
    providers, consumers = register_participants(market)
    
    # Initialize data handler
    data_handler = DataHandler(data_dir=ROOT_DIR / "data")
    
    # Generate sample datasets
    datasets = generate_sample_datasets(data_handler)
    
    # List datasets on market
    dataset_ids = list_datasets_on_market(market, providers, datasets)
    
    # Simulate market transactions
    transactions = simulate_market_transactions(market, consumers, dataset_ids)
    
    # Initialize differential privacy
    differential_privacy = DifferentialPrivacy(epsilon=0.5, delta=1e-6)
    
    # Apply differential privacy protection to data
    original_df, private_df = apply_privacy_to_dataset(data_handler, differential_privacy)
    
    # Visualize market statistics
    visualize_market_statistics(market, transactions)
    
    # Visualize privacy comparison
    visualize_privacy_comparison(original_df, private_df)
    
    logger.info("Data market demo completed!")

if __name__ == "__main__":
    main() 