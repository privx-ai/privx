#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Homomorphic Encryption module
Implement homomorphic encryption functionality, supporting calculations on encrypted data
"""

import logging
import numpy as np
import tenseal as ts
from typing import Dict, List, Tuple, Optional, Any, Union

logger = logging.getLogger(__name__)

class HomomorphicEncryption:
    """Homomorphic Encryption class, implementing homomorphic encryption functionality"""
    
    def __init__(self, scheme: str = 'ckks', poly_modulus_degree: int = 8192, 
                bit_sizes: List[int] = None, global_scale: float = 2**40):
        """
        Initialize homomorphic encryption
        
        Args:
            scheme: Encryption scheme (ckks, bfv)
            poly_modulus_degree: Polynomial modulus degree
            bit_sizes: List of bit sizes
            global_scale: Global scale factor (only used in CKKS scheme)
        """
        self.scheme = scheme
        self.poly_modulus_degree = poly_modulus_degree
        self.bit_sizes = bit_sizes or [60, 40, 40, 60]
        self.global_scale = global_scale
        
        # Generate keys
        self._generate_keys()
        
        logger.info(f"Homomorphic encryption initialized, scheme: {scheme}, polynomial modulus degree: {poly_modulus_degree}")
    
    def _generate_keys(self):
        """Generate encryption keys"""
        if self.scheme == 'ckks':
            # CKKS scheme, supports approximate real number calculations
            context = ts.context(
                ts.SCHEME_TYPE.CKKS,
                poly_modulus_degree=self.poly_modulus_degree,
                coeff_mod_bit_sizes=self.bit_sizes
            )
            context.global_scale = self.global_scale
            context.generate_galois_keys()
            
        elif self.scheme == 'bfv':
            # BFV scheme, supports integer calculations
            context = ts.context(
                ts.SCHEME_TYPE.BFV,
                poly_modulus_degree=self.poly_modulus_degree,
                coeff_mod_bit_sizes=self.bit_sizes,
                plain_modulus=1032193
            )
            
        else:
            logger.error(f"Unsupported encryption scheme: {self.scheme}")
            raise ValueError(f"Unsupported encryption scheme: {self.scheme}")
        
        # Save context and keys
        self.context = context
        self.secret_key = context.secret_key()
        self.public_key = context.public_key()
        
        logger.info(f"Generated {self.scheme} key pair")
    
    def encrypt_vector(self, vector: np.ndarray) -> Any:
        """
        Encrypt vector
        
        Args:
            vector: Original vector
            
        Returns:
            Any: Encrypted vector
        """
        if self.scheme == 'ckks':
            # Encrypt vector in CKKS scheme
            encrypted_vector = ts.ckks_vector(self.context, vector)
            
        elif self.scheme == 'bfv':
            # Encrypt vector in BFV scheme
            encrypted_vector = ts.bfv_vector(self.context, vector.astype(int))
            
        else:
            logger.error(f"Unsupported encryption scheme: {self.scheme}")
            raise ValueError(f"Unsupported encryption scheme: {self.scheme}")
        
        logger.info(f"Vector encryption completed, length: {len(vector)}")
        return encrypted_vector
    
    def decrypt_vector(self, encrypted_vector: Any) -> np.ndarray:
        """
        Decrypt vector
        
        Args:
            encrypted_vector: Encrypted vector
            
        Returns:
            np.ndarray: Decrypted vector
        """
        # Decrypt vector
        decrypted_vector = encrypted_vector.decrypt()
        
        logger.info(f"Vector decryption completed, length: {len(decrypted_vector)}")
        return np.array(decrypted_vector)
    
    def encrypt_matrix(self, matrix: np.ndarray) -> List[Any]:
        """
        Encrypt matrix (encrypt row by row)
        
        Args:
            matrix: Original matrix
            
        Returns:
            List[Any]: Encrypted matrix (list of row vectors)
        """
        encrypted_rows = []
        
        for row in matrix:
            encrypted_row = self.encrypt_vector(row)
            encrypted_rows.append(encrypted_row)
        
        logger.info(f"Matrix encryption completed, shape: {matrix.shape}")
        return encrypted_rows
    
    def decrypt_matrix(self, encrypted_rows: List[Any]) -> np.ndarray:
        """
        Decrypt matrix
        
        Args:
            encrypted_rows: Encrypted matrix (list of row vectors)
            
        Returns:
            np.ndarray: Decrypted matrix
        """
        decrypted_rows = []
        
        for encrypted_row in encrypted_rows:
            decrypted_row = self.decrypt_vector(encrypted_row)
            decrypted_rows.append(decrypted_row)
        
        logger.info(f"Matrix decryption completed, number of rows: {len(decrypted_rows)}")
        return np.array(decrypted_rows)
    
    def add_vectors(self, vec1: Any, vec2: Any) -> Any:
        """
        Add encrypted vectors
        
        Args:
            vec1: First encrypted vector
            vec2: Second encrypted vector
            
        Returns:
            Any: Addition result
        """
        result = vec1 + vec2
        logger.info("Encrypted vector addition completed")
        return result
    
    def add_plain(self, encrypted_vec: Any, plain_vec: np.ndarray) -> Any:
        """
        Add encrypted vector and plain vector
        
        Args:
            encrypted_vec: Encrypted vector
            plain_vec: Plain vector
            
        Returns:
            Any: Addition result
        """
        result = encrypted_vec + plain_vec
        logger.info("Encrypted vector and plain vector addition completed")
        return result
    
    def multiply_vectors(self, vec1: Any, vec2: Any) -> Any:
        """
        Multiply encrypted vectors (element-wise)
        
        Args:
            vec1: First encrypted vector
            vec2: Second encrypted vector
            
        Returns:
            Any: Multiplication result
        """
        result = vec1 * vec2
        logger.info("Encrypted vector multiplication completed")
        return result
    
    def multiply_plain(self, encrypted_vec: Any, plain_vec: np.ndarray) -> Any:
        """
        Multiply encrypted vector and plain vector
        
        Args:
            encrypted_vec: Encrypted vector
            plain_vec: Plain vector
            
        Returns:
            Any: Multiplication result
        """
        result = encrypted_vec * plain_vec
        logger.info("Encrypted vector and plain vector multiplication completed")
        return result
    
    def dot_product(self, encrypted_vec: Any, plain_vec: np.ndarray) -> Any:
        """
        Dot product of encrypted vector and plain vector
        
        Args:
            encrypted_vec: Encrypted vector
            plain_vec: Plain vector
            
        Returns:
            Any: Dot product result
        """
        if self.scheme == 'ckks':
            # For CKKS, dot product can be calculated directly
            result = encrypted_vec.dot(plain_vec)
            
        elif self.scheme == 'bfv':
            # For BFV, dot product needs to be calculated manually
            result = self.multiply_plain(encrypted_vec, plain_vec)
            result = result.sum()
            
        else:
            logger.error(f"Unsupported encryption scheme: {self.scheme}")
            raise ValueError(f"Unsupported encryption scheme: {self.scheme}")
        
        logger.info("Encrypted vector and plain vector dot product completed")
        return result
    
    def linear_transform(self, encrypted_vec: Any, weight_matrix: np.ndarray, bias: Optional[np.ndarray] = None) -> Any:
        """
        Linear transformation: y = Wx + b
        
        Args:
            encrypted_vec: Encrypted vector x
            weight_matrix: Weight matrix W
            bias: Bias vector b
            
        Returns:
            Any: Transformation result
        """
        if self.scheme == 'ckks':
            # For CKKS, linear transformation can be calculated directly
            result = encrypted_vec.matmul(weight_matrix)
            
            if bias is not None:
                result = result + bias
                
        elif self.scheme == 'bfv':
            # For BFV, linear transformation needs to be calculated manually
            n_out = weight_matrix.shape[0]
            result = None
            
            for i in range(n_out):
                row_result = self.dot_product(encrypted_vec, weight_matrix[i])
                
                if result is None:
                    result = [row_result]
                else:
                    result.append(row_result)
            
            if bias is not None:
                for i in range(n_out):
                    result[i] = result[i] + bias[i]
                
        else:
            logger.error(f"Unsupported encryption scheme: {self.scheme}")
            raise ValueError(f"Unsupported encryption scheme: {self.scheme}")
        
        logger.info("Encrypted vector linear transformation completed")
        return result
    
    def sigmoid_approximation(self, encrypted_vec: Any, degree: int = 3) -> Any:
        """
        Sigmoid function approximation (only supported in CKKS scheme)
        
        Args:
            encrypted_vec: Encrypted vector
            degree: Polynomial approximation degree
            
        Returns:
            Any: Approximation result
        """
        if self.scheme != 'ckks':
            logger.error(f"Sigmoid approximation is only supported in CKKS scheme")
            raise ValueError(f"Sigmoid approximation is only supported in CKKS scheme")
        
        # Approximate Sigmoid function using polynomial
        if degree == 1:
            # Linear approximation: 0.5 + 0.25 * x
            result = encrypted_vec * 0.25 + 0.5
            
        elif degree == 3:
            # Cubic polynomial approximation: 0.5 + 0.197 * x - 0.004 * x^3
            x3 = encrypted_vec * encrypted_vec * encrypted_vec
            result = encrypted_vec * 0.197 - x3 * 0.004 + 0.5
            
        elif degree == 5:
            # Quintic polynomial approximation
            x2 = encrypted_vec * encrypted_vec
            x3 = x2 * encrypted_vec
            x5 = x3 * x2
            result = encrypted_vec * 0.2159 - x3 * 0.0082 + x5 * 0.0019 + 0.5
            
        else:
            logger.error(f"Unsupported polynomial approximation degree: {degree}")
            raise ValueError(f"Unsupported polynomial approximation degree: {degree}")
        
        logger.info(f"Sigmoid function approximation completed, polynomial degree: {degree}")
        return result
    
    def secure_logistic_inference(self, encrypted_features: Any, weights: np.ndarray, bias: np.ndarray) -> Any:
        """
        Secure logistic regression inference
        
        Args:
            encrypted_features: Encrypted feature vector
            weights: Model weights
            bias: Model bias
            
        Returns:
            Any: Encrypted prediction result
        """
        # Linear transformation
        linear_output = self.dot_product(encrypted_features, weights)
        
        if bias is not None:
            linear_output = linear_output + bias
        
        # Sigmoid approximation
        prediction = self.sigmoid_approximation(linear_output)
        
        logger.info("Secure logistic regression inference completed")
        return prediction
    
    def save_context(self, path: str) -> None:
        """
        Save encryption context
        
        Args:
            path: Save path
        """
        self.context.serialize(path)
        logger.info(f"Encryption context saved to: {path}")
    
    def load_context(self, path: str) -> None:
        """
        Load encryption context
        
        Args:
            path: Load path
        """
        if self.scheme == 'ckks':
            self.context = ts.context_from(path, ts.SCHEME_TYPE.CKKS)
        elif self.scheme == 'bfv':
            self.context = ts.context_from(path, ts.SCHEME_TYPE.BFV)
        else:
            logger.error(f"Unsupported encryption scheme: {self.scheme}")
            raise ValueError(f"Unsupported encryption scheme: {self.scheme}")
        
        logger.info(f"Encryption context loaded from: {path}")
    
    def serialize_encrypted_vector(self, encrypted_vec: Any) -> bytes:
        """
        Serialize encrypted vector
        
        Args:
            encrypted_vec: Encrypted vector
            
        Returns:
            bytes: Serialized data
        """
        serialized = encrypted_vec.serialize()
        logger.info(f"Encrypted vector serialization completed, size: {len(serialized)} bytes")
        return serialized
    
    def deserialize_encrypted_vector(self, serialized: bytes) -> Any:
        """
        Deserialize encrypted vector
        
        Args:
            serialized: Serialized data
            
        Returns:
            Any: Encrypted vector
        """
        if self.scheme == 'ckks':
            encrypted_vec = ts.ckks_vector_from(self.context, serialized)
        elif self.scheme == 'bfv':
            encrypted_vec = ts.bfv_vector_from(self.context, serialized)
        else:
            logger.error(f"Unsupported encryption scheme: {self.scheme}")
            raise ValueError(f"Unsupported encryption scheme: {self.scheme}")
        
        logger.info("Encrypted vector deserialization completed")
        return encrypted_vec
