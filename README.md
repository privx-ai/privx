# Privx

This project demonstrates a decentralized data market and privacy computing platform based on blockchain technology, integrated with federated learning techniques, allowing AI models to be trained while protecting data privacy.

## Project Features

- **Decentralized AI Training**: Combines Federated Learning (FL) on the blockchain to train AI models while protecting data privacy.
- **Data Trading Market**: Users can sell anonymized data on the blockchain market, and AI models can securely use the data for training via smart contracts.
- **Privacy Protection**: Utilizes differential privacy, homomorphic encryption, and other technologies to ensure data privacy.
- **Smart Contracts**: Uses simulated blockchain smart contracts to manage data transactions and model training.

## Project Structure

```
.
├── README.md                 # Project documentation
├── requirements.txt          # Project dependencies
├── src/                      # Source code
│   ├── blockchain/           # Blockchain-related code
│   │   ├── smart_contract.py # Smart contract simulation
│   │   └── blockchain.py     # Blockchain simulation
│   ├── federated_learning/   # Federated learning-related code
│   │   ├── client.py         # Client code
│   │   ├── server.py         # Server code
│   │   └── model.py          # Model definition
│   ├── privacy/              # Privacy computation-related code
│   │   ├── differential_privacy.py # Differential privacy
│   │   └── homomorphic_encryption.py # Homomorphic encryption
│   ├── data_market/          # Data market-related code
│   │   ├── market.py         # Core market functionality
│   │   └── data_handler.py   # Data handling
│   └── main.py               # Main program entry
└── examples/                 # Example code
    ├── market_demo.py        # Data market example
    └── federated_learning_demo.py # Federated learning example
```

## Core Functionality

### 1. Data Market

The data market module implements blockchain-based data trading functionality, including:

- **Data Provider Management**: Registration, reputation scoring, dataset management
- **Data Consumer Management**: Registration, balance management, purchase history
- **Dataset Management**: Listing, pricing, trading, rating
- **Market Analysis**: Transaction statistics, market trends, popular datasets

Main Interfaces:
- `register_provider`: Register data provider
- `register_consumer`: Register data consumer
- `list_dataset`: List dataset
- `purchase_dataset`: Purchase dataset
- `rate_dataset`: Rate dataset
- `search_datasets`: Search datasets
- `get_market_statistics`: Get market statistics

### 2. Federated Learning

The federated learning module implements decentralized AI training functionality, including:

- **Server Side**: Model initialization, aggregation, evaluation
- **Client Side**: Local training, model update, privacy protection
- **Model Management**: Weight encryption, blockchain record, version control
- **Performance Evaluation**: Accuracy, precision, recall, F1 score

Training Process:
1. The server initializes the global model
2. Clients download the global model
3. Clients train the model using local data
4. Clients upload model updates (not raw data)
5. The server aggregates all clients' model updates
6. Repeat steps 2-5 until the model converges

### 3. Privacy Computing

The privacy computing module implements data privacy protection functionality, including:

#### Differential Privacy
- **Data Perturbation**: Adding carefully calibrated noise to protect individual privacy
- **Privacy Budget**: Controlling the strength of privacy protection
- **Private Statistics**: Mean, variance, count, etc., with differential privacy
- **Private Learning**: Differential privacy machine learning models

Main Interfaces:
- `privatize_dataframe`: Add differential privacy to a dataframe
- `private_mean/std/count/sum`: Differential privacy statistical functions
- `privatize_gradient`: Add differential privacy to gradients

#### Homomorphic Encryption
- **Encrypted Computation**: Performing computations on encrypted data
- **Secure Inference**: Model inference in an encrypted state
- **Key Management**: Encryption context and key generation

Main Interfaces:
- `encrypt_vector/matrix`: Encrypt vector/matrix
- `add_vectors/multiply_vectors`: Encrypted vector operations
- `linear_transform`: Encrypted linear transformation
- `secure_logistic_inference`: Secure logistic regression inference

### 4. Blockchain Integration

The blockchain module provides decentralized infrastructure, including:

- **Blockchain Implementation**: Simplified blockchain structure
- **Smart Contracts**: Data trading and model training contracts
- **Transaction Verification**: Ensuring transaction validity and security
- **Data Hashing**: Hash storage of data and models

## Installation & Usage

### Environment Requirements

- Python 3.8+
- Dependencies (see requirements.txt)

### Installation Steps

1. Clone the repository
```bash
git clone https://github.com/privx-ai/privx.git
cd data-market-privacy-computing
```

2. Create a virtual environment (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

### Running Examples

#### Data Market Example

The data market example demonstrates the basic functionality of the data trading market, including data provider registration, dataset listing, trading, and rating, etc.

```bash
python examples/market_demo.py
```

The example will:
- Generate various types of sample datasets (user behavior, medical, finance, location, social network)
- Simulate data provider and consumer registration
- List and trade datasets on the market
- Apply differential privacy to protect sensitive data
- Visualize market statistics and privacy protection effects

Results will be saved as charts in the `examples/charts` directory.

#### Federated Learning Example

The federated learning example demonstrates the basic functionality of decentralized AI training, including data segmentation, local training, model aggregation, and performance evaluation, etc.

```bash
python examples/federated_learning_demo.py
```

The example will:
- Prepare a dataset for diabetes risk prediction
- Distribute the data to multiple clients in a non-IID manner
- Apply differential privacy to the training process
- Train a federated learning model and record it to the blockchain
- Compare the performance with centralized training models
- Visualize the training process and model performance

Results will be saved as charts in the `examples/charts` directory.

### Running Main Program

The main program provides various running modes, which can be controlled by command-line parameters:

```bash
python src/main.py --mode demo
```

Available modes include:
- `demo`: Run a simplified demonstration
- `market`: Run only the data market functionality
- `federated`: Run only the federated learning functionality
- `full`: Run the full functionality of the data market and federated learning

Other parameters:
- `--clients`: Number of federated learning clients
- `--rounds`: Number of federated learning training rounds
- `--privacy`: Enable differential privacy
- `--encryption`: Enable homomorphic encryption

## Technical Details

### Differential Privacy

Differential privacy adds carefully calibrated noise to data to ensure that individual information cannot be inferred from query results. This project implements differential privacy using Laplace and Gaussian mechanisms, with main parameters including:

- `epsilon`: Privacy budget, smaller values indicate stronger privacy protection
- `delta`: Privacy failure probability
- `sensitivity`: Query sensitivity

### Homomorphic Encryption

Homomorphic encryption allows computations to be performed on encrypted data without decryption. This project implements CKKS and BFV homomorphic encryption schemes, supporting operations including:

- Addition and multiplication of encrypted vectors
- Operations between encrypted vectors and plaintext vectors
- Linear transformations and approximate nonlinear functions

### Federated Learning

Federated learning allows multiple participants to train AI models without sharing raw data. This project implements the FedAvg algorithm, with main steps including:

1. The server initializes the global model
2. Clients download the global model
3. Clients train the model using local data
4. Clients upload model updates (not raw data)
5. The server aggregates all clients' model updates
6. Repeat steps 2-5 until the model converges

## Extension & Customization

### Adding New Datasets

New sample datasets can be generated using the `generate_sample_dataset` method of the `DataHandler` class, or external datasets can be loaded using the `load_dataset` method.

Example:
```python
from src.data_market.data_handler import DataHandler

# Initialize data handler
data_handler = DataHandler(data_dir="./data")

# Define columns for new dataset
columns = [
    {"name": "feature1", "type": "float", "min": 0, "max": 100},
    {"name": "feature2", "type": "int", "min": 1, "max": 10},
    {"name": "category", "type": "category", "categories": ["A", "B", "C"]}
]

# Generate dataset
df = data_handler.generate_sample_dataset("my_dataset", rows=1000, columns=columns)
```

### Implementing New Models

New model classes can be added in `federated_learning/model.py`, by implementing the following methods:
- `__init__`: Initialize model
- `fit`: Train model
- `predict`: Predict results
- `get_weights`: Get model weights
- `set_weights`: Set model weights

### Customizing Privacy Settings

Privacy protection levels can be customized by adjusting the parameters of the `DifferentialPrivacy` class, or encryption parameters can be modified in the `HomomorphicEncryption` class.

Example:
```python
from src.privacy.differential_privacy import DifferentialPrivacy

# Create strong privacy protection
dp = DifferentialPrivacy(epsilon=0.1, delta=1e-6)

# Apply differential privacy
private_df = dp.privatize_dataframe(
    df,
    numeric_columns=["feature1", "feature2"],
    sensitivities={"feature1": 5.0, "feature2": 1.0},
    mechanism="gaussian"
)
```

## Contribution

Feel free to submit issues and pull requests. For major changes, please open an issue first to discuss the changes you wish to make.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For any questions or suggestions, please contact us via:

- [GitHub](https://github.com/privx-ai/privx)
- [Twitter](https://x.com/Ai_Privx)