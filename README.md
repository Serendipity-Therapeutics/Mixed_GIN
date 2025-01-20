# Mixed_GIN

Mixed_GIN is a research-oriented framework designed for robust prediction on the TDC ADMET benchmark group using advanced graph neural network models from the Deep Graph Library (DGL). The core objective of Mixed_GIN is to fuse multiple variants of Graph Isomorphism Network (GIN) embeddings—specifically, `gin_supervised_masking`, `gin_supervised_infomax`, `gin_supervised_edgepred`, and `gin_supervised_contextpred`—into a single, robust predictive model. This fusion in the embedding space leverages the strengths of each GIN variant to improve prediction accuracy and generalization for complex molecular property prediction tasks.

---

## Directory Structure

```
Mixed_GIN/  
├── .gitignore  
├── README.md  
├── main.py  
├── .log/  
│   ├── cyp2c9.log 
│   ├── cyp2d6.log 
│   └── cyp3a4.log
├── data/  
│   ├── data_loader.py  
│   └── tdc_benchmark.py                  
├── models/  
│   ├── __init__.py  
│   └── gin_infomax.py    
├── utils/  
│   ├── __init__.py  
│   ├── loss_fn.py      
│   └── util.py    
```

### Key Files and Directories
- **`.gitignore`**: Specifies files and directories to be ignored by Git.
- **`main.py`**: The main execution script that orchestrates data loading, model training, evaluation, and prediction on the TDC ADMET benchmarks.
- **`data/`**: Contains datasets and related resources.
    - **`data_loader.py`**: Provides functions and classes for loading and 
    - **`tdc_benchmark.py`**: Defines the `ADMETBenchmarks` class and other utilities to interface with the TDC benchmark suite.
- **`models/`**: Contains model definitions. The key file `gin_infomax.py` implements a GIN-based model variant.
- **`utils/`**: Contains utility modules, including custom loss functions (`loss_fn.py`) and various helper utilities (`util.py`) for scheduling, early stopping, and other training utilities.
- **`.log/`**: The logging data about each cyp prediction test result

---

## Objectives

The primary goal of Mixed_GIN is to:

1. Utilize and integrate several supervised GIN models (`gin_supervised_masking`, `gin_supervised_infomax`, `gin_supervised_edgepred`, `gin_supervised_contextpred`) into a unified architecture.
2. Perform fusion at the embedding level to create a more robust representation of molecular graphs.
3. Achieve improved predictive performance on TDC ADMET tasks, which cover a range of ADMET (Absorption, Distribution, Metabolism, Excretion, and Toxicity) properties.
4. Leverage DGL (Deep Graph Library) implementations for efficient and scalable graph neural network operations.

---

## Features

- **Embedding Fusion**: Combines different GIN model embeddings to capture diverse structural and contextual information from molecular graphs.
- **TDC ADMET Predictions**: Direct support for evaluating models on TDC ADMET benchmarks with robust training, validation, and testing workflows.
- **Flexible Data Handling**: Custom data loaders and dataset classes to handle molecular SMILES strings and associated labels.
- **Robust Training Utilities**: Utilities for learning rate scheduling, early stopping, and custom loss functions tailored for molecular property prediction.

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/Mixed_GIN.git
cd Mixed_GIN
```

**Note**: Ensure that `requirements.txt` includes necessary packages such as `torch`, `dgl`, `tqdm`, `datamol`, and `tdc`.

---

## Usage

To run the main training and evaluation pipeline on a selected ADMET benchmark:

```bash
python main.py
```

The script will:

1. Load the desired TDC ADMET benchmark dataset.
2. Initialize and train the Mixed_GIN model with fused embeddings.
3. Monitor training using `tqdm` progress bars, apply learning rate scheduling, and utilize early stopping.
4. Evaluate the model on the test set and output performance metrics.

---

## Customization

1. **Model Fusion**: Modify the fusion strategy in the model definition (e.g., within `models/gin_infomax.py` or extend with additional modules) to experiment with different ways of combining GIN embeddings.
2. **Hyperparameters**: Adjust training hyperparameters such as learning rate, batch size, hidden dimensions, and early stopping criteria in `main.py`.
3. **Loss Functions & Utilities**: Explore and modify custom loss functions in `utils/loss_fn.py` and training utilities in `utils/util.py` to tailor the training process.

---

## Contributing

Contributions to Mixed_GIN are welcome. Please open issues or pull requests for bug fixes, feature enhancements, or research ideas related to robust GIN-based models for molecular property prediction.

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

Mixed_GIN aims to advance the state-of-the-art in molecular property prediction by robustly integrating multiple GIN-based embeddings. For further detail