"""
Blockchain Simulation Module
For simulating blockchain network and basic functionality
"""

import hashlib
import json
import time
from typing import List, Dict, Any


class Block:
    """Block class, represents a single block in the blockchain"""
    
    def __init__(self, index: int, timestamp: float, transactions: List[Dict[str, Any]], 
                 previous_hash: str, nonce: int = 0):
        """
        Initialize block
        
        Args:
            index: Block index
            timestamp: Timestamp
            transactions: List of transactions
            previous_hash: Hash of the previous block
            nonce: Random number used for proof of work
        """
        self.index = index
        self.timestamp = timestamp
        self.transactions = transactions
        self.previous_hash = previous_hash
        self.nonce = nonce
        self.hash = self.calculate_hash()
    
    def calculate_hash(self) -> str:
        """
        Calculate block hash
        
        Returns:
            Block hash
        """
        block_string = json.dumps({
            "index": self.index,
            "timestamp": self.timestamp,
            "transactions": self.transactions,
            "previous_hash": self.previous_hash,
            "nonce": self.nonce
        }, sort_keys=True).encode()
        
        return hashlib.sha256(block_string).hexdigest()
    
    def mine_block(self, difficulty: int) -> None:
        """
        Mine block (proof of work)
        
        Args:
            difficulty: Mining difficulty (number of leading zeros)
        """
        target = "0" * difficulty
        
        while self.hash[:difficulty] != target:
            self.nonce += 1
            self.hash = self.calculate_hash()
            
        print(f"Block #{self.index} mined: {self.hash}")


class Blockchain:
    """Blockchain class, manages blockchain"""
    
    def __init__(self, difficulty: int = 4):
        """
        Initialize blockchain
        
        Args:
            difficulty: Mining difficulty
        """
        self.chain: List[Block] = []
        self.difficulty = difficulty
        self.pending_transactions: List[Dict[str, Any]] = []
        self.mining_reward = 100
        
        # Create genesis block
        self.create_genesis_block()
    
    def create_genesis_block(self) -> None:
        """Create genesis block"""
        genesis_block = Block(0, time.time(), [], "0")
        genesis_block.hash = genesis_block.calculate_hash()
        self.chain.append(genesis_block)
    
    def get_latest_block(self) -> Block:
        """
        Get latest block
        
        Returns:
            Latest block
        """
        return self.chain[-1]
    
    def mine_pending_transactions(self, mining_reward_address: str) -> None:
        """
        Mine pending transactions
        
        Args:
            mining_reward_address: Address to receive mining reward
        """
        # Create reward transaction
        self.pending_transactions.append({
            "sender": "system",
            "recipient": mining_reward_address,
            "amount": self.mining_reward,
            "timestamp": time.time(),
            "type": "mining_reward"
        })
        
        # Create new block
        block = Block(
            len(self.chain),
            time.time(),
            self.pending_transactions,
            self.get_latest_block().hash
        )
        
        # Mine block
        block.mine_block(self.difficulty)
        
        # Add block to chain
        self.chain.append(block)
        
        # Clear pending transactions
        self.pending_transactions = []
    
    def create_transaction(self, sender: str, recipient: str, amount: float, 
                          transaction_type: str = "transfer", data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Create transaction
        
        Args:
            sender: Sender address
            recipient: Recipient address
            amount: Transaction amount
            transaction_type: Transaction type
            data: Additional data
            
        Returns:
            Created transaction
        """
        transaction = {
            "sender": sender,
            "recipient": recipient,
            "amount": amount,
            "timestamp": time.time(),
            "type": transaction_type
        }
        
        if data:
            transaction["data"] = data
            
        self.pending_transactions.append(transaction)
        return transaction
    
    def get_balance(self, address: str) -> float:
        """
        Get address balance
        
        Args:
            address: Address to query
            
        Returns:
            Address balance
        """
        balance = 0
        
        for block in self.chain:
            for transaction in block.transactions:
                if transaction["recipient"] == address:
                    balance += transaction["amount"]
                if transaction["sender"] == address:
                    balance -= transaction["amount"]
        
        return balance
    
    def is_chain_valid(self) -> bool:
        """
        Validate blockchain
        
        Returns:
            True if blockchain is valid, False otherwise
        """
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i - 1]
            
            # Validate current block hash
            if current_block.hash != current_block.calculate_hash():
                return False
            
            # Validate blockchain connection
            if current_block.previous_hash != previous_block.hash:
                return False
        
        return True


# Example usage
if __name__ == "__main__":
    # Create blockchain
    blockchain = Blockchain(difficulty=2)
    
    # Create some transactions
    blockchain.create_transaction("address1", "address2", 100)
    blockchain.create_transaction("address2", "address1", 50)
    
    # Mine block
    print("Starting mining...")
    blockchain.mine_pending_transactions("miner-address")
    
    # Check miner balance
    print(f"Miner balance: {blockchain.get_balance('miner-address')}")
    
    # Create more transactions
    blockchain.create_transaction("address1", "address2", 200)
    blockchain.create_transaction("address2", "address1", 75)
    
    # Mine again
    print("Starting mining second block...")
    blockchain.mine_pending_transactions("miner-address")
    
    # Check miner balance again
    print(f"Miner balance: {blockchain.get_balance('miner-address')}")
    
    # Validate blockchain
    print(f"Blockchain valid: {blockchain.is_chain_valid()}") 