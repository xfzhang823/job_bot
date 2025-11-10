# job_bot
A bot helps me in job hunting, such as resume optimization, interview guide, so on...

# Project Structure

```
project_root/
├── src/
│   ├── db_io/                        # Existing database and I/O logic
│   │   ├── state_sync.py
│   │   ├── duckdb_adapter.py
│   │   ├── db_utils.py
│   │   ├── schema_definitions.py
│   │   ├── db_transform.py
│   │   ├── db_inserters.py
│   │   └── db_loaders.py
│   │
│   ├── fsm/                          # Existing FSM management utilities
│   │   ├── __init__.py
│   │   ├── fsm_utils.py
│   │   ├── pipeline_fsm_manager.py
|   |   ├── fsm_integrity_checker.py   # Validate FSM states & transitions
|   |   ├── fsm_state_control.py       # Control/manage FSM lifecycle (initialization, updates)
|   |   └── llm_fsm.py                 # Future module for LLM-driven FSM decisions
│   │
│   ├── models/                       # Existing Pydantic models
│   ├── pipelines/                    # Existing pipeline implementations
│   ├── pipelines_with_fsm/           # 🆕 FSM-integrated pipeline implementations
│   ├── prompts/                      # 🆕 Prompt templates and management
│   ├── human_review_and_editing/     # 🆕 Human-in-the-loop review interfaces
│   ├── preprocessing/                # 🆕 Data preprocessing utilities
│   ├── evaluation_optimization/      # 🆕 Pipeline evaluation and optimization tools
│   ├── utils/                        # Existing general utilities/helpers
│   └── llm_providers/                # Existing LLM integration logic
│
└── tests/
    └── test_fsm/                     # Unit tests specifically for FSM/pipeline_control
```

## Key Components

### Core Systems
- **Database I/O (`db_io/`)**
  - Core database operations and utilities
  - Includes DuckDB adapter, schema definitions, and transformation logic

- **Finite State Machine (`fsm/`)**
  - FSM utilities and pipeline state management
  - **New Components**:
    - `pipeline_control.py`: Centralized state management
    - `fsm_analytics.py`: Pipeline control analytics
    - `integrity_checker.py`: FSM validation and integrity checks

### Pipeline Ecosystem
- **`pipelines/`**: Baseline pipeline implementations
- **`pipelines_with_fsm/`**: 🆕 Production pipelines with integrated state management
- **`preprocessing/`**: 🆕 Data cleaning/normalization utilities

### LLM Components
- **`prompts/`**: 🆕 Prompt templates and version management
- **`llm_providers/`**: LLM API integrations and adapters

### Quality Control
- **`human_review_and_editing/`**: 🆕 Interfaces for manual review/annotation
- **`evaluation_optimization/`**: 🆕 Performance metrics and tuning tools

### Supporting Directories
- `models/`: Pydantic model definitions
- `utils/`: General utility functions

### Tests
- `test_fsm/`: Unit tests for FSM and pipeline control components


# Pipeline Management Overview

In this project, pipelines are defined, managed, and executed in a modular fashion. The main components responsible for managing the pipelines include the following files:

1. **pipeline_config.py**: Contains configuration settings for the pipelines, such as input/output paths, parameters, and model specifications. This file organizes the pipeline logic and prepares it for execution.
   
2. **pipeline_manager.py**: The core logic for managing the execution of different pipeline steps. It manages loading configurations, initializing components, and orchestrating the pipeline's flow.
   
3. **run_pipelines.py**: This file handles the orchestration of the pipeline execution. It provides functionality to trigger specific pipelines, execute them in sequence, and handle any necessary cleanup or logging.

4. **main.py**: The entry point for running the entire pipeline. It imports the necessary components and starts the pipeline execution, passing configurations and managing the flow.


## Step-by-Step Instructions to Run Pipelines

### 1. **Set Up Environment**
   Before running the pipeline, ensure that all required dependencies are installed. These dependencies are typically listed in a `requirements.txt` file or can be identified through the `import` statements in the code.

   ```bash
   pip install -r requirements.txt
   ```

### 2. **Configure Pipeline Settings**
   The pipeline configuration file (`pipeline_config.py`) contains key parameters for pipeline execution, such as input directories, output paths, and specific settings for each pipeline step. Review and modify this file if necessary.

   - Ensure paths for input/output are correct.
   - Set model configurations or other parameters needed by your pipeline steps.

### 3. **Understand the Pipeline Steps**
   The pipelines in the project are divided into various steps. Each step performs a specific task, such as data processing, transformation, or model inference. These steps are orchestrated in `pipeline_manager.py` and executed in sequence in `run_pipelines.py`.

   The pipeline steps include:
   - **Pre-processing**: Prepare data or configuration.
   - **Execution**: Run the model or task.
   - **Post-processing**: Handle results, e.g., saving them to a file.

### 4. **Run the Pipelines**

   You can run the pipeline from the `main.py` file. This file imports all the components necessary to start the pipeline and triggers the execution process.

   To run the pipeline, execute the following command:

   ```bash
   python main.py
   ```

   This command will:
   - Initialize the pipeline manager.
   - Load configuration settings from `pipeline_config.py`.
   - Orchestrate the execution of the steps defined in `run_pipelines.py`.

   You can also modify `main.py` to run specific pipelines or adjust the parameters passed to the pipeline manager.

### 5. **Check the Output**
   After running the pipeline, check the output in the directories specified in the configuration. The results may include processed data, model outputs, or logs. These results can be used for further analysis or model evaluation.

---

## Notes
- The project is modular and allows for easy adjustments to the pipeline configuration and execution flow.
- If you want to execute only a specific part of the pipeline, you can modify `run_pipelines.py` to focus on individual steps or use flags in `main.py`.
---

# Alignment Transformer Model Pipeline

## Pipeline Overview
This repository contains a modular pipeline for training and evaluating an Alignment Transformer Model. The pipeline leverages transformer-based models like BERT to align resume sections with job descriptions, incorporating a hybrid attention mechanism to refine embeddings.

## Module Breakdown

### 1. `alignment_preprocessing.py`
**Purpose:** Handles data preprocessing for hybrid alignment.  
**Key Functions:**  
- Load composite scores and position data from CSV files.  
- Apply logarithmic scaling to position keys.  

### 2. `transformer_model.py`
**Purpose:** Defines the architecture of the Alignment Transformer Model.  
**Key Components:**  
- **Input Encoder:** Uses a pre-trained BERT model to encode text.  
- **Hybrid Attention Model:** Combines cross-attention and self-attention.  

### 3. `alignment_losses.py`
**Purpose:** Defines the loss function used for training.  
**Key Losses:**  
- **Alignment Loss:** Based on cosine similarity.  
- **Substance Loss:** Preserves the original resume embedding.  
- **Coherence Loss:** Ensures linguistic consistency.  

### 4. `transformer_training.py`
**Purpose:** Orchestrates the training loop for the model.  
**Key Tasks:**  
- Handles the forward pass through the model.  
- Calculates the combined loss.  
- Optimizes model parameters using AdamW optimizer.  

### 5. `model_saver.py`
**Purpose:** Saves the trained model and embeddings.  
**Key Features:**  
- Saves the model weights after training.  
- Saves the embeddings for faster reuse.  

### 6. `transformer_training_pipeline.py`
**Purpose:** The main orchestration script.  
**Key Functions:**  
- Loads data and applies preprocessing steps.  
- Initializes the model and loss function.  
- Runs the training loop and saves the results.

## How to Run the Pipeline

### Step 1: Set Up Environment
Install dependencies using:  
```bash
pip install -r requirements.txt