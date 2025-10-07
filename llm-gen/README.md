# Role Classification for Cyberbullying Detection

A comprehensive zero-shot classification system for identifying roles in cyberbullying scenarios using Large Language Models (LLMs).

## 📋 Overview

This project implements a sophisticated role classification system that can identify different participant roles in cyberbullying conversations using zero-shot, one-shot, and few-shot learning approaches. The system is designed to work with LLAMA models and provides comprehensive analysis capabilities.

## 🎯 Role Categories

The system classifies text into four distinct roles:

- **Bully**: A person who initiates harassment, bullying, or targeting others with negative behavior
- **Victim**: A person who is being harassed, bullied, or targeted with negative behavior  
- **Bully Support**: A bystander-assistant who takes part in or encourages the actions of the harasser
- **Victim Support**: A bystander-defender who helps the victim and discourages the harasser

## 🚀 Features

### Core Classification (`roleClassification.py`)
- **Zero-shot, One-shot, and Few-shot Learning**: Multiple prompt templates for different classification approaches
- **Multi-Dataset Support**: Processes multiple test datasets simultaneously
- **Comprehensive Metrics**: Accuracy, precision, recall, F1-score, and confusion matrices
- **GPU Monitoring**: Real-time GPU memory and power consumption tracking
- **Checkpointing**: Resume interrupted jobs from where they left off
- **Detailed Logging**: Complete experiment tracking and raw model outputs
- **Performance Mode**: Optimized for high-performance inference

### Data Generation (`gen.py`)
- **Scenario-based Generation**: Generate conversations from predefined cyberbullying scenarios
- **Flexible Prompt Templates**: Support for custom prompt templates
- **Metadata Tracking**: Comprehensive logging of generation parameters and performance metrics
- **Multi-format Support**: Handles both standard and transformed scenario formats

### Analysis Tools
- **Error Analysis** (`Error_analysis.py`): Conversation-level similarity analysis with metadata overlays
- **Similarity Check** (`Similarity_check.py`): Advanced similarity analysis between conversations

## 📁 Project Structure

```
Codebase/
├── roleClassification.py    # Main classification script
├── gen.py                   # Data generation script
├── Error_analysis.py        # Error analysis and similarity tools
├── Similarity_check.py      # Advanced similarity analysis
└── README.md               # This file
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- PyTorch with CUDA support

### Dependencies
```bash
pip install torch transformers accelerate pandas numpy scikit-learn
pip install matplotlib seaborn networkx sentence-transformers
pip install pynvml tqdm
```

## 🎮 Usage

### 1. Role Classification

Run the main classification script:

```bash
python roleClassification.py \
    --model_name "meta-llama/Llama-2-7b-chat-hf" \
    --data_dir "./data" \
    --output_dir "./results" \
    --prompt_type "zero_shot" \
    --batch_size 8
```

**Key Parameters:**
- `--model_name`: HuggingFace model identifier
- `--data_dir`: Directory containing test datasets
- `--output_dir`: Output directory for results
- `--prompt_type`: Classification approach (`zero_shot`, `one_shot`, `few_shot`)
- `--batch_size`: Batch size for processing

### 2. Data Generation

Generate conversation data from scenarios:

```bash
python gen.py \
    --model_name "meta-llama/Llama-2-7b-chat-hf" \
    --scenarios_file "scenarios.json" \
    --prompt_file "prompt_template.txt" \
    --output_dir "./generated_data" \
    --num_scenarios 5
```

### 3. Analysis

Run similarity analysis:

```bash
python Similarity_check.py
```

Run error analysis:

```bash
python Error_analysis.py
```

## 📊 Output Structure

### Classification Results
```
results/
├── classification_reports/
│   ├── confusion_matrices/
│   ├── performance_metrics/
│   └── detailed_reports/
├── raw_outputs/
├── checkpoints/
└── experiment_summaries/
```

### Generated Data
```
generated_data/
├── conversations/
├── metadata/
└── experiment_logs/
```

## 🔧 Configuration

### Model Configuration
- Supports any HuggingFace-compatible LLM
- Automatic GPU memory management
- Mixed precision inference (FP16)

### Performance Optimization
- **Performance Mode**: Enabled by default for optimal inference speed
- **Batch Processing**: Configurable batch sizes
- **GPU Monitoring**: Real-time performance tracking
- **Checkpointing**: Automatic job recovery

### Prompt Templates
The system includes three types of prompt templates:
- **Zero-shot**: Direct classification without examples
- **One-shot**: Single example for guidance
- **Few-shot**: Multiple examples for better accuracy

## 📈 Performance Metrics

The system provides comprehensive evaluation metrics:
- **Accuracy**: Overall classification accuracy
- **Precision/Recall/F1**: Per-class performance metrics
- **Confusion Matrix**: Detailed classification breakdown
- **Confidence Scores**: Model confidence for each prediction
- **GPU Statistics**: Memory usage and power consumption

## 🔍 Analysis Capabilities

### Similarity Analysis
- **Intra-scenario**: Compare runs within the same scenario
- **Inter-scenario**: Compare across different scenarios
- **Network Visualization**: Graph-based similarity representation
- **Metadata Overlay**: Analysis by aggression type, mode, and target gender

### Error Analysis
- **Conversation-level Analysis**: Group-level similarity assessment
- **Cross-scenario Pairs**: Identify similar conversations across scenarios
- **Subtype Analysis**: Performance breakdown by cyberbullying type


## 📄 License

TBD


**Note**: This system is designed for research purposes and should be used responsibly. Always ensure compliance with relevant data protection and privacy regulations when processing real-world data. 