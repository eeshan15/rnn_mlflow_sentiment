````markdown
# Sentiment Analysis Pipeline (RNN + MLflow + DVC)

This project demonstrates an intermediate MLOps workflow using DVC Pipelines to automate the machine learning lifecycle. It implements a Sentiment Analysis model using a PyTorch RNN, where data preprocessing and training steps are orchestrated via a `dvc.yaml` file.

## Project Overview

* **Domain:** Natural Language Processing (Sentiment Analysis).
* **Model:** Recurrent Neural Network (RNN) with Embedding layers.
* **Pipeline Automation:** DVC (`dvc.yaml`) manages dependencies and execution order.
* **Smart Caching:** DVC ensures only changed stages are re-executed.
* **Experiment Tracking:** MLflow logs training metrics and artifacts.

## Prerequisites

Ensure you have Python installed along with the following libraries:

* torch
* pandas
* mlflow
* dvc

## Setup and Installation

1.  **Initialize the environment:**
    ```bash
    mkdir rnn_dvc_pipeline
    cd rnn_dvc_pipeline
    ```

2.  **Install dependencies:**
    ```bash
    pip install torch pandas mlflow dvc
    ```

3.  **Initialize Git and DVC:**
    ```bash
    git init
    dvc init
    ```

## Workflow Instructions

### 1. Generate Simulated Data
Since this is a demonstration, we first generate a synthetic dataset containing dummy reviews and labels.

```bash
python src/get_data.py
````

  * **Output:** Creates `data/raw_data.csv`.

### 2\. Run the DVC Pipeline

Instead of running scripts manually, we use DVC to reproduce the entire pipeline. This command checks the `dvc.yaml` file and executes the defined stages in order.

```bash
dvc repro
```

**What happens during execution:**

1.  **Stage: prepare** (`src/prepare.py`)
      * Reads `data/raw_data.csv`.
      * Tokenizes text and builds a vocabulary.
      * Saves processed tensors to `data/processed/train_data.pt`.
2.  **Stage: train** (`src/train.py`)
      * Loads processed data.
      * Trains the RNN model.
      * Logs metrics (Loss) and parameters to MLflow.
      * Saves the final model to `models/sentiment_model.pth`.

### 3\. Verify Smart Caching

If you run `dvc repro` again without changing any code or data:

```bash
dvc repro
```

**Result:** DVC will skip execution ("Stage 'prepare' didn't change"), saving time and computational resources.

### 4\. View Experiments in MLflow

To visualize the training performance:

```bash
mlflow ui
```

Open `http://127.0.0.1:5000` in your browser.

## Pipeline Structure (`dvc.yaml`)

The workflow is defined in `dvc.yaml`:

  * **prepare:** Depends on `data/raw_data.csv` and `src/prepare.py`. Outputs `data/processed/train_data.pt`.
  * **train:** Depends on `src/train.py` and the output of the prepare stage. Outputs the trained model.

## Project Directory Layout

  * `dvc.yaml`: Defines the pipeline stages and dependencies.
  * `dvc.lock`: Automatically generated file that records the state of the pipeline (hashes of dependencies and outputs).
  * `src/get_data.py`: Script to generate raw simulation data.
  * `src/prepare.py`: Script for tokenization and padding.
  * `src/train.py`: Script for model training and MLflow logging.
  * `data/`: Contains raw and processed datasets (managed by DVC).
  * `models/`: Stores the trained PyTorch models.

<!-- end list -->
