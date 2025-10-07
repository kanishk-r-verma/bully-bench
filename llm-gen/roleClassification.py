#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import pandas as pd
import numpy as np
import argparse
import os
import random
import time
import json
import re
import logging
import sys
import pickle
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from accelerate import Accelerator
from pynvml import *
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns


def init_accelerator():
    """Initialize and return a hardware accelerator for multi-GPU."""
    return Accelerator(device_placement=True, split_batches=True)


def init_model(model_name, is_shared=False, shared_path=None):
    """Initialize and return the model and tokenizer."""
    if is_shared and shared_path:
        full_model_path = os.path.join(shared_path, model_name)
        if not os.path.exists(shared_path):
            raise FileNotFoundError(f"Shared model directory not found: {shared_path}")
        if not os.path.exists(full_model_path):
            raise FileNotFoundError(f"Model not found in shared directory: {full_model_path}")
        
        logging.info(f"Loading model from shared path: {full_model_path}")
        model_path = full_model_path
    else:
        model_path = model_name
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except Exception as e:
        raise Exception(f"Failed to load tokenizer from {model_path}: {str(e)}")
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16
        )
    except Exception as e:
        raise Exception(f"Failed to load model from {model_path}: {str(e)}")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


def set_random_seed(seed_value):
    """Set random seeds for reproducibility across all libraries and hardware"""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    set_seed(seed_value)


def get_gpu_stats():
    """Get GPU memory usage and power consumption"""
    try:
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)
        info = nvmlDeviceGetMemoryInfo(handle)
        power = nvmlDeviceGetPowerUsage(handle) / 1000.0
        name = nvmlDeviceGetName(handle).decode('utf-8')
        
        return {
            'GPU_Name': name,
            'GPU_Memory_Used_GB': round(info.used / 1024**3, 2),
            'GPU_Memory_Total_GB': round(info.total / 1024**3, 2),
            'GPU_Power_Usage_W': round(power, 2)
        }
    except Exception as e:
        logging.warning(f"Could not get GPU stats: {e}")
        return {
            'GPU_Name': 'Unknown',
            'GPU_Memory_Used_GB': -1,
            'GPU_Memory_Total_GB': -1,
            'GPU_Power_Usage_W': -1
        }


def load_prompt_template(template_type="zero_shot"):
    """Load different prompt templates for role classification"""
    
    # Define the new prompt templates
    prompt_templates = {
        "zero_shot": """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are tasked with classifying text into one of four categories related to online harassment scenarios.

Categories and Definitions:
- victim: A person who is being harassed, bullied, or targeted with negative behavior
- bully: A person who initiates harassment, bullying, or targeting others with negative behavior
- victim_support: A bystander-defender who helps the victim and discourages the harasser
- bully_support: A bystander-assistant who takes part in or encourages the actions of the harasser

Analyze the text and classify it into exactly ONE of the four categories above. Respond with only the category label.<|eot_id|><|start_header_id|>user<|end_header_id|>

{text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        
        "one_shot": """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are tasked with classifying text into one of four categories related to online harassment scenarios.

Categories and Definitions:
- victim: A person who is being harassed, bullied, or targeted with negative behavior
- bully: A person who initiates harassment, bullying, or targeting others with negative behavior
- victim_support: A bystander-defender who helps the victim and discourages the harasser
- bully_support: A bystander-assistant who takes part in or encourages the actions of the harasser

Analyze the text and classify it into exactly ONE of the four categories above. Respond with only the category label.<|eot_id|><|start_header_id|>user<|end_header_id|>

Please stop making fun of Sarah's appearance. That's really hurtful and unnecessary.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

victim_support<|eot_id|><|start_header_id|>user<|end_header_id|>

{text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        
        "few_shot": """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are tasked with classifying text into one of four categories related to online harassment scenarios.

Categories and Definitions:
- victim: A person who is being harassed, bullied, or targeted with negative behavior
- bully: A person who initiates harassment, bullying, or targeting others with negative behavior
- victim_support: A bystander-defender who helps the victim and discourages the harasser
- bully_support: A bystander-assistant who takes part in or encourages the actions of the harasser

Analyze the text and classify it into exactly ONE of the four categories above. Respond with only the category label.<|eot_id|><|start_header_id|>user<|end_header_id|>

Shut up you idiot, you only care about the school you are all day in front of the books NERD<|eot_id|><|start_header_id|>assistant<|end_header_id|>

bully_support<|eot_id|><|start_header_id|>user<|end_header_id|>

we're so sad that we're normal, so shut up<|eot_id|><|start_header_id|>assistant<|end_header_id|>

bully_support<|eot_id|><|start_header_id|>user<|end_header_id|>

harper dont get too comfy were watchin u 👀<|eot_id|><|start_header_id|>assistant<|end_header_id|>

bully_support<|eot_id|><|start_header_id|>user<|end_header_id|>

yes, your mother gets down on her knees then reaches up with her hands to grab my balls<|eot_id|><|start_header_id|>assistant<|end_header_id|>

bully<|eot_id|><|start_header_id|>user<|end_header_id|>

fuck your mothers you fags<|eot_id|><|start_header_id|>assistant<|end_header_id|>

bully<|eot_id|><|start_header_id|>user<|end_header_id|>

gideon ur moms basement dweller 😂<|eot_id|><|start_header_id|>assistant<|end_header_id|>

bully<|eot_id|><|start_header_id|>user<|end_header_id|>

Meanwhile you get suspended and I got an A<|eot_id|><|start_header_id|>assistant<|end_header_id|>

victim<|eot_id|><|start_header_id|>user<|end_header_id|>

in real life I'm sure we could be friends, it's just that you're jealous<|eot_id|><|start_header_id|>assistant<|end_header_id|>

victim<|eot_id|><|start_header_id|>user<|end_header_id|>

dont wanna get involved 🙅<|eot_id|><|start_header_id|>assistant<|end_header_id|>

victim<|eot_id|><|start_header_id|>user<|end_header_id|>

You were talking so much about tomorrow, that you make him pay but today you didn't even have the courage to look him in the face . .. I feel bad for you poor guys<|eot_id|><|start_header_id|>assistant<|end_header_id|>

victim_support<|eot_id|><|start_header_id|>user<|end_header_id|>

This is not how you're supposed to think about life<|eot_id|><|start_header_id|>assistant<|end_header_id|>

victim_support<|eot_id|><|start_header_id|>user<|end_header_id|>

bryce thats enough uve gone too far 😡<|eot_id|><|start_header_id|>assistant<|end_header_id|>

victim_support<|eot_id|><|start_header_id|>user<|end_header_id|>

{text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
    }
    
    # Get the requested template type
    template = prompt_templates.get(template_type, prompt_templates["zero_shot"])
    logging.info(f"Using prompt template type: {template_type}")
    return template


def extract_classification_from_response(response_text):
    """Extract role classification from model response"""
    if not response_text:
        return "victim"  # Default to victim if no response
    
    # Clean the response
    response_text = response_text.strip().lower()
    
    # Look for role keywords - only 4 categories now
    role_patterns = {
        'bully': r'\b(bully|aggressor|attacker|instigator)\b',
        'victim': r'\b(victim|target)\b',
        'bully_support': r'\b(bully_support|bully support|supporter|encourager)\b',
        'victim_support': r'\b(victim_support|victim support|defender|helper)\b'
    }
    
    # Check for exact matches first
    for role, pattern in role_patterns.items():
        if re.search(pattern, response_text):
            return role
    
    # If no pattern found, try to infer from context
    if any(word in response_text for word in ['attack', 'insult', 'threat', 'harm']):
        return 'bully'
    elif any(word in response_text for word in ['defend', 'help', 'support', 'protect']):
        return 'victim_support'
    elif any(word in response_text for word in ['hurt', 'scared', 'afraid', 'target']):
        return 'victim'
    else:
        return 'victim'  # Default to victim if unclear


def calculate_confidence_score(response_text):
    """Calculate confidence score based on response characteristics"""
    if not response_text:
        return 0.0
    
    # Base confidence
    confidence = 0.5
    
    # Increase confidence for clear, direct responses
    if len(response_text.strip()) < 50:
        confidence += 0.2
    
    # Increase confidence for responses with clear role keywords
    role_keywords = ['bully', 'victim', 'support']
    if any(keyword in response_text.lower() for keyword in role_keywords):
        confidence += 0.2
    
    # Decrease confidence for uncertain language
    uncertain_words = ['maybe', 'perhaps', 'could be', 'might be', 'not sure']
    if any(word in response_text.lower() for word in uncertain_words):
        confidence -= 0.2
    
    return min(max(confidence, 0.0), 1.0)


def classify_text(model, tokenizer, text, prompt_template, generation_params, accelerator):
    """Classify a single text using the model"""
    
    # Format the prompt
    formatted_prompt = prompt_template.format(text=text)
    
    # Tokenize input
    inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=2048)
    
    # Move to device
    inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            **generation_params
        )
    
    # Decode response
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract the generated part (after the prompt)
    prompt_length = len(tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True))
    response_text = full_response[prompt_length:].strip()
    
    # Extract classification and confidence
    classification = extract_classification_from_response(response_text)
    confidence = calculate_confidence_score(response_text)
    
    return {
        'classification': classification,
        'confidence': confidence,
        'raw_response': response_text,
        'full_response': full_response,
        'formatted_prompt': formatted_prompt,
        'input_tokens': inputs['input_ids'].shape[1],
        'output_tokens': outputs.shape[1] - inputs['input_ids'].shape[1]
    }


def save_checkpoint(output_dir, dataset_name, results, true_labels, predicted_labels, current_idx, total_samples):
    """Save checkpoint to resume from where we left off"""
    checkpoint_data = {
        'results': results,
        'true_labels': true_labels,
        'predicted_labels': predicted_labels,
        'current_idx': current_idx,
        'total_samples': total_samples,
        'timestamp': datetime.now().isoformat()
    }
    
    checkpoint_file = os.path.join(output_dir, f'checkpoint_{dataset_name}.pkl')
    try:
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        logging.info(f"Checkpoint saved for {dataset_name} at index {current_idx}/{total_samples}")
    except Exception as e:
        logging.error(f"Error saving checkpoint: {e}")


def load_checkpoint(output_dir, dataset_name):
    """Load checkpoint to resume from where we left off"""
    checkpoint_file = os.path.join(output_dir, f'checkpoint_{dataset_name}.pkl')
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'rb') as f:
                checkpoint_data = pickle.load(f)
            logging.info(f"Loaded checkpoint for {dataset_name} from index {checkpoint_data['current_idx']}")
            return checkpoint_data
        except Exception as e:
            logging.error(f"Error loading checkpoint: {e}")
    return None


def save_raw_outputs_log(output_dir, dataset_name, results_df):
    """Save detailed raw model outputs to a separate log file"""
    raw_log_file = os.path.join(output_dir, f'raw_model_outputs_{dataset_name}.log')
    
    try:
        with open(raw_log_file, 'w', encoding='utf-8') as f:
            f.write(f"# Raw Model Outputs for {dataset_name}\n")
            f.write(f"# Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
            
            for idx, row in results_df.iterrows():
                f.write(f"Sample {idx+1}:\n")
                f.write(f"ID: {row['id']}\n")
                f.write(f"Name: {row['name']}\n")
                f.write(f"Text: {row['text']}\n")
                f.write(f"True Label: {row['true_label']}\n")
                f.write(f"Predicted Label: {row['predicted_label']}\n")
                f.write(f"Confidence: {row['confidence']:.3f}\n")
                f.write(f"Raw Response: {row['raw_response']}\n")
                f.write(f"Full Response: {row['full_response']}\n")
                f.write(f"Input Tokens: {row.get('input_tokens', 'N/A')}\n")
                f.write(f"Output Tokens: {row.get('output_tokens', 'N/A')}\n")
                f.write("-"*80 + "\n\n")
        
        logging.info(f"Raw outputs log saved: {raw_log_file}")
        return raw_log_file
    except Exception as e:
        logging.error(f"Error saving raw outputs log: {e}")
        return None


def create_confusion_matrix_plot(y_true, y_pred, output_dir, model_name, dataset_name):
    """Create and save confusion matrix plot"""
    try:
        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=['bully', 'victim', 'bully_support', 'victim_support'])
        
        # Create plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['bully', 'victim', 'bully_support', 'victim_support'],
                   yticklabels=['bully', 'victim', 'bully_support', 'victim_support'])
        plt.title(f'Confusion Matrix - {dataset_name} - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        
        # Save plot
        cm_file = os.path.join(output_dir, f'confusion_matrix_{dataset_name}_{model_name.replace("/", "_")}.png')
        plt.savefig(cm_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Confusion matrix saved: {cm_file}")
        return cm_file
        
    except Exception as e:
        logging.error(f"Error creating confusion matrix: {e}")
        return None


def save_detailed_results(results_df, output_dir, model_name, experiment_name, dataset_name):
    """Save detailed classification results"""
    try:
        # Create filename
        results_file = os.path.join(output_dir, f'results_{dataset_name}_{model_name.replace("/", "_")}_{experiment_name}.csv')
        
        # Save to CSV
        results_df.to_csv(results_file, index=False)
        
        logging.info(f"Detailed results saved: {results_file}")
        return results_file
        
    except Exception as e:
        logging.error(f"Error saving detailed results: {e}")
        return None


def generate_performance_report(y_true, y_pred, results_df, output_dir, model_name, experiment_name, gpu_stats, dataset_name):
    """Generate comprehensive performance report"""
    try:
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        
        # Generate classification report
        class_report = classification_report(y_true, y_pred, output_dict=True)
        
        # Create report content
        report_content = f"""
# Role Classification Performance Report

## Dataset: {dataset_name}
## Model: {model_name}
## Experiment: {experiment_name}
## Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Overall Performance Metrics
- Accuracy: {accuracy:.4f}
- Weighted Precision: {precision:.4f}
- Weighted Recall: {recall:.4f}
- Weighted F1-Score: {f1:.4f}

## Per-Class Performance
"""
        
        # Add per-class metrics
        for class_name in ['bully', 'victim', 'bully_support', 'victim_support']:
            if class_name in class_report:
                metrics = class_report[class_name]
                report_content += f"""
### {class_name.upper()}
- Precision: {metrics['precision']:.4f}
- Recall: {metrics['recall']:.4f}
- F1-Score: {metrics['f1-score']:.4f}
- Support: {metrics['support']}
"""
        
        # Add confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=['bully', 'victim', 'bully_support', 'victim_support'])
        report_content += f"""
## Confusion Matrix
```
{cm}
```

## GPU Statistics
- GPU Name: {gpu_stats['GPU_Name']}
- GPU Memory Used: {gpu_stats['GPU_Memory_Used_GB']} GB
- GPU Memory Total: {gpu_stats['GPU_Memory_Total_GB']} GB
- GPU Power Usage: {gpu_stats['GPU_Power_Usage_W']} W

## Sample Predictions
"""
        
        # Add sample predictions
        sample_size = min(10, len(results_df))
        for i in range(sample_size):
            row = results_df.iloc[i]
            report_content += f"""
### Sample {i+1}
- Text: {row['text'][:100]}...
- True Role: {row['true_label']}
- Predicted Role: {row['predicted_label']}
- Confidence: {row['confidence']:.3f}
- Raw Response: {row['raw_response'][:100]}...
"""
        
        # Save report
        report_file = os.path.join(output_dir, f'performance_report_{dataset_name}_{model_name.replace("/", "_")}_{experiment_name}.md')
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logging.info(f"Performance report saved: {report_file}")
        return report_file
        
    except Exception as e:
        logging.error(f"Error generating performance report: {e}")
        return None


def load_and_preprocess_data(data_file):
    """Load and preprocess data from different file formats"""
    try:
        df = pd.read_csv(data_file)
        
        # Handle different column names for role labels
        role_columns = ['ROLE', 'role', 'Role']
        text_columns = ['TEXT', 'text', 'Text']
        
        # Find the correct column names
        role_col = None
        text_col = None
        
        for col in role_columns:
            if col in df.columns:
                role_col = col
                break
        
        for col in text_columns:
            if col in df.columns:
                text_col = col
                break
        
        if role_col is None or text_col is None:
            raise ValueError(f"Could not find role or text columns in {data_file}")
        
        # Standardize column names
        df = df.rename(columns={role_col: 'ROLE', text_col: 'TEXT'})
        
        # Handle missing values
        df = df.dropna(subset=['TEXT', 'ROLE'])
        
        # Standardize role labels - only 4 categories now
        role_mapping = {
            'bully': 'bully',
            'victim': 'victim', 
            'bully_support': 'bully_support',
            'victim_support': 'victim_support',
            'none': 'victim',  # Map 'none' to 'victim' as default
            'None': 'victim',
            'NONE': 'victim'
        }
        
        df['ROLE'] = df['ROLE'].map(role_mapping).fillna('victim')
        
        return df
        
    except Exception as e:
        logging.error(f"Error loading data from {data_file}: {e}")
        raise


def save_detailed_classification_report(results_df, output_dir, model_name, experiment_name, dataset_name):
    """Save detailed classification report with comprehensive analysis, including macro/weighted avg."""
    try:
        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
        
        y_true = results_df['true_label'].tolist()
        y_pred = results_df['predicted_label'].tolist()
        
        # Overall metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        class_report = classification_report(y_true, y_pred, output_dict=True)
        cm = confusion_matrix(y_true, y_pred, labels=['bully', 'victim', 'bully_support', 'victim_support'])
        
        report_content = f"""
# Detailed Classification Report

## Dataset Information
- Dataset Name: {dataset_name}
- Model: {model_name}
- Experiment: {experiment_name}
- Total Samples: {len(results_df)}
- Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Overall Performance Metrics
- Accuracy: {accuracy:.4f}
- Weighted Precision: {precision:.4f}
- Weighted Recall: {recall:.4f}
- Weighted F1-Score: {f1:.4f}

## All Metrics (from sklearn classification_report)

| Class         | Precision | Recall | F1-Score | Support |
|-------------- |---------- |------- |--------- |-------- |
"""
        for label in class_report:
            if isinstance(class_report[label], dict):
                report_content += f"| {label:14} | {class_report[label]['precision']:.4f} | {class_report[label]['recall']:.4f} | {class_report[label]['f1-score']:.4f} | {class_report[label]['support']}     |\n"
        report_content += "\n## Confusion Matrix\n```\n" + str(cm) + "\n```\n"
        report_content += f"""
## Detailed Sample Analysis
"""
        sample_size = min(20, len(results_df))
        for i in range(sample_size):
            row = results_df.iloc[i]
            correct = "✓" if row['true_label'] == row['predicted_label'] else "✗"
            report_content += f"""
### Sample {i+1} {correct}
- Text: {row['text'][:150]}...
- True Role: {row['true_label']}
- Predicted Role: {row['predicted_label']}
- Confidence: {row['confidence']:.3f}
- Raw Response: {row['raw_response'][:100]}...
- Input Tokens: {row.get('input_tokens', 'N/A')}
- Output Tokens: {row.get('output_tokens', 'N/A')}
"""
        incorrect_predictions = results_df[results_df['true_label'] != results_df['predicted_label']]
        if len(incorrect_predictions) > 0:
            report_content += f"""
## Error Analysis
Total Incorrect Predictions: {len(incorrect_predictions)} ({len(incorrect_predictions)/len(results_df)*100:.2f}%)

### Most Common Error Patterns
"""
            error_patterns = incorrect_predictions.groupby(['true_label', 'predicted_label']).size().sort_values(ascending=False)
            for (true_label, pred_label), count in error_patterns.head(10).items():
                report_content += f"- {true_label} → {pred_label}: {count} times\n"
        report_file = os.path.join(output_dir, f'detailed_classification_report_{dataset_name}_{model_name.replace("/", "_")}_{experiment_name}.md')
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        logging.info(f"Detailed classification report saved: {report_file}")
        return report_file
    except Exception as e:
        logging.error(f"Error generating detailed classification report: {e}")
        return None


def save_json_results(results_df, output_dir, model_name, experiment_name, dataset_name, gpu_stats):
    """Save results in JSON format with comprehensive metadata, including macro/weighted avg and all per-class metrics."""
    try:
        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
        y_true = results_df['true_label'].tolist()
        y_pred = results_df['predicted_label'].tolist()
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        class_report = classification_report(y_true, y_pred, output_dict=True)
        cm = confusion_matrix(y_true, y_pred, labels=['bully', 'victim', 'bully_support', 'victim_support'])
        json_data = {
            "metadata": {
                "dataset_name": dataset_name,
                "model_name": model_name,
                "experiment_name": experiment_name,
                "timestamp": datetime.now().isoformat(),
                "total_samples": len(results_df),
                "gpu_stats": gpu_stats
            },
            "performance_metrics": {
                "overall": {
                    "accuracy": float(accuracy),
                    "weighted_precision": float(precision),
                    "weighted_recall": float(recall),
                    "weighted_f1_score": float(f1)
                },
                "all_metrics": class_report
            },
            "confusion_matrix": {
                "labels": ['bully', 'victim', 'bully_support', 'victim_support'],
                "matrix": cm.tolist()
            },
            "predictions": []
        }
        for idx, row in results_df.iterrows():
            prediction = {
                "sample_id": int(row.get('id', idx)),
                "name": str(row.get('name', '')),
                "text": str(row['text']),
                "true_label": str(row['true_label']),
                "predicted_label": str(row['predicted_label']),
                "confidence": float(row['confidence']),
                "raw_response": str(row['raw_response']),
                "input_tokens": int(row.get('input_tokens', 0)),
                "output_tokens": int(row.get('output_tokens', 0)),
                "correct": bool(row['true_label'] == row['predicted_label'])
            }
            json_data["predictions"].append(prediction)
        json_file = os.path.join(output_dir, f'results_{dataset_name}_{model_name.replace("/", "_")}_{experiment_name}.json')
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        logging.info(f"JSON results saved: {json_file}")
        return json_file
    except Exception as e:
        logging.error(f"Error saving JSON results: {e}")
        return None


def create_prompt_comparison_report(all_results, output_dir, model_name, current_time):
    """Create a comprehensive comparison report when running all prompt types"""
    try:
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        
        report_content = f"""
# Prompt Type Comparison Report

## Model: {model_name}
## Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Summary
This report compares the performance of different prompt types (zero-shot, one-shot, few-shot) across all datasets.

"""
        
        # Create comparison table
        report_content += "## Performance Comparison\n\n"
        report_content += "| Dataset | Prompt Type | Accuracy | Precision | Recall | F1-Score | Processing Time |\n"
        report_content += "|---------|-------------|----------|-----------|--------|----------|-----------------|\n"
        
        best_performances = {}
        
        for dataset_name, prompt_results in all_results.items():
            dataset_accuracies = []
            
            for prompt_type, results in prompt_results.items():
                y_true = results['true_labels']
                y_pred = results['predicted_labels']
                
                accuracy = accuracy_score(y_true, y_pred)
                precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
                processing_time = results['processing_time']
                
                report_content += f"| {dataset_name} | {prompt_type} | {accuracy:.4f} | {precision:.4f} | {recall:.4f} | {f1:.4f} | {processing_time:.2f}s |\n"
                
                dataset_accuracies.append((prompt_type, accuracy))
            
            # Find best performing prompt type for this dataset
            if dataset_accuracies:
                best_prompt, best_acc = max(dataset_accuracies, key=lambda x: x[1])
                best_performances[dataset_name] = (best_prompt, best_acc)
        
        # Add best performance summary
        report_content += "\n## Best Performing Prompt Types by Dataset\n\n"
        for dataset_name, (best_prompt, best_acc) in best_performances.items():
            report_content += f"- **{dataset_name}**: {best_prompt} (Accuracy: {best_acc:.4f})\n"
        
        # Add overall statistics
        all_accuracies = []
        for prompt_results in all_results.values():
            for results in prompt_results.values():
                accuracy = accuracy_score(results['true_labels'], results['predicted_labels'])
                all_accuracies.append(accuracy)
        
        if all_accuracies:
            avg_accuracy = sum(all_accuracies) / len(all_accuracies)
            report_content += f"\n## Overall Statistics\n"
            report_content += f"- Average Accuracy Across All Experiments: {avg_accuracy:.4f}\n"
            report_content += f"- Total Experiments: {len(all_accuracies)}\n"
            report_content += f"- Best Single Accuracy: {max(all_accuracies):.4f}\n"
            report_content += f"- Worst Single Accuracy: {min(all_accuracies):.4f}\n"
        
        # Save comparison report
        comparison_file = os.path.join(output_dir, f'prompt_comparison_report_{model_name.replace("/", "_")}_{current_time}.md')
        with open(comparison_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logging.info(f"Prompt comparison report saved: {comparison_file}")
        return comparison_file
        
    except Exception as e:
        logging.error(f"Error creating prompt comparison report: {e}")
        return None


def main():
    """Main function for zero-shot role classification"""
    
    print("=" * 80)
    print("Zero-Shot Role Classification for Cyberbullying Detection")
    print("=" * 80)
    print("This script performs zero-shot classification on multiple test datasets.")
    print("\nExample commands:")
    print("  python main.py                                    # Run with defaults")
    print("  python main.py --model_name meta-llama/Llama-3.3-70B-Instruct")
    print("  python main.py --prompt_type zero_shot            # Use zero-shot prompt")
    print("  python main.py --prompt_type one_shot             # Use one-shot prompt")
    print("  python main.py --prompt_type few_shot             # Use few-shot prompt")
    print("  python main.py --prompt_type all                  # Run all three prompt types")
    print("  python main.py --use_shared --shared_model_path /path/to/models  # Use shared models")
    print("  python main.py --debug                            # Enable debug mode")
    print("  python main.py --resume                           # Resume from checkpoint")
    print("\nAvailable prompt types: zero_shot, one_shot, few_shot, all")
    print("\nTest datasets:")
    print("  - DATA/all_data/role_all/llm_both/llm_both_test.csv")
    print("  - DATA/GOLD_TEST.csv\n")
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Zero-shot role classification for cyberbullying detection.")
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-3.3-70B-Instruct', 
                        help='Name/path of the model to use')
    parser.add_argument('--prompt_type', type=str, default='zero_shot', 
                        choices=['zero_shot', 'one_shot', 'few_shot', 'all'],
                        help='Type of prompt template to use (use "all" to run all three types)')
    parser.add_argument('--custom_prompt_file', type=str, default=None,
                        help='Path to custom prompt template file (overrides prompt_type)')
    parser.add_argument('--shared_model_path', type=str, default='/home/support/llm/', 
                        help='Path to shared model directory')
    parser.add_argument('--use_shared', action='store_true', default=False,
                        help='Use model from shared directory')
    parser.add_argument('--max_samples', type=int, default=None, 
                        help='Maximum number of samples to process per dataset (for testing)')
    parser.add_argument('--debug', action='store_true', default=False, 
                        help='Enable debug mode with additional print statements')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed for reproducibility')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='Resume from checkpoint if available')
    parser.add_argument('--checkpoint_interval', type=int, default=50,
                        help='Save checkpoint every N samples')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for inference (default: 32)')
    args = parser.parse_args()

    # Set random seed
    set_random_seed(args.seed)

    # Create output directory
    current_time = datetime.now().strftime("%Y_%m_%d_%H_%M")
    output_dir = f'role_classification_results_{current_time}'
    os.makedirs(output_dir, exist_ok=True)

    # Configure logging
    log_file = os.path.join(output_dir, 'classification.log')
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        filename=log_file,
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Also log to console if debug mode is enabled
    if args.debug:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logging.getLogger().addHandler(console_handler)

    # Log configuration
    logging.info("=== Zero-Shot Role Classification Started ===")
    logging.info(f"Configuration:")
    logging.info(f"- Model: {args.model_name}")
    logging.info(f"- Prompt Type: {args.prompt_type}")
    logging.info(f"- Use Shared: {args.use_shared}")
    logging.info(f"- Max Samples: {args.max_samples}")
    logging.info(f"- Debug Mode: {args.debug}")
    logging.info(f"- Seed: {args.seed}")
    logging.info(f"- Resume: {args.resume}")
    logging.info(f"- Checkpoint Interval: {args.checkpoint_interval}")
    logging.info(f"- Batch Size: {args.batch_size}")

    # Define test datasets
    test_datasets = [
        {
            'name': 'role_all_llm_both',
            'path': 'DATA/all_data/role_all/llm_both/llm_both_test.csv'
        },
        {
            'name': 'gold_test',
            'path': 'DATA/GOLD_TEST.csv'
        }
    ]

    # Load model and tokenizer
    logging.info("Loading model and tokenizer...")
    if args.debug:
        print("Loading model and tokenizer...")

    try:
        model, tokenizer = init_model(args.model_name, is_shared=args.use_shared, shared_path=args.shared_model_path)
        logging.info(f"Model loaded successfully: {args.model_name}")
        if args.debug:
            print(f"Model loaded successfully: {args.model_name}")
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        print(f"Error loading model: {e}")
        return 1

    # Initialize accelerator
    accelerator = init_accelerator()
    model = accelerator.prepare(model)
    model.eval()

    # Load prompt template
    if args.custom_prompt_file:
        # Load custom prompt file
        if not os.path.exists(args.custom_prompt_file):
            logging.error(f"Custom prompt file {args.custom_prompt_file} not found. Exiting.")
            print(f"Error: Custom prompt file {args.custom_prompt_file} not found.")
            return 1
        
        try:
            with open(args.custom_prompt_file, 'r', encoding='utf-8') as f:
                prompt_template = f.read().strip()
            logging.info(f"Loaded custom prompt template from: {args.custom_prompt_file}")
            if args.debug:
                print(f"Loaded custom prompt template from: {args.custom_prompt_file}")
        except Exception as e:
            logging.error(f"Error loading custom prompt file: {e}")
            print(f"Error loading custom prompt file: {e}")
            return 1
    else:
        # Load standard prompt template
        prompt_template = load_prompt_template(args.prompt_type)
        logging.info(f"Using prompt type: {args.prompt_type}")

    # Define generation parameters
    generation_params = {
        'temperature': 0.1,        # Low temperature for more consistent outputs
        'do_sample': True,
        'max_new_tokens': 50,      # Short responses for classification
        'top_p': 0.9,
        'top_k': 20,
        'repetition_penalty': 1.1,
        'pad_token_id': tokenizer.pad_token_id,
        'eos_token_id': tokenizer.eos_token_id
    }

    def process_dataset_with_prompt_type(dataset, prompt_type, model, tokenizer, generation_params, accelerator, args, output_dir, current_time):
        """Process a single dataset with a specific prompt type"""
        dataset_name = dataset['name']
        data_file = dataset['path']
        
        logging.info(f"Processing dataset: {dataset_name} with prompt type: {prompt_type}")
        print(f"\nProcessing dataset: {dataset_name} with prompt type: {prompt_type}")
        
        # Check if data file exists
        if not os.path.exists(data_file):
            logging.error(f"Data file {data_file} not found. Skipping.")
            print(f"Error: Data file {data_file} not found. Skipping.")
            return None

        try:
            # Load and preprocess data
            df = load_and_preprocess_data(data_file)
            logging.info(f"Loaded {len(df)} samples from {data_file}")
            
            # Limit samples if specified
            if args.max_samples:
                df = df.head(args.max_samples)
                logging.info(f"Limited to {len(df)} samples for testing")

            # Load prompt template for this type
            if args.custom_prompt_file:
                with open(args.custom_prompt_file, 'r', encoding='utf-8') as f:
                    prompt_template = f.read().strip()
            else:
                prompt_template = load_prompt_template(prompt_type)

            # Initialize results storage
            results = []
            true_labels = []
            predicted_labels = []
            start_idx = 0

            # Check for checkpoint if resume is enabled
            if args.resume:
                checkpoint_data = load_checkpoint(output_dir, f"{dataset_name}_{prompt_type}")
                if checkpoint_data:
                    results = checkpoint_data['results']
                    true_labels = checkpoint_data['true_labels']
                    predicted_labels = checkpoint_data['predicted_labels']
                    start_idx = checkpoint_data['current_idx']
                    logging.info(f"Resuming from checkpoint at index {start_idx}")

            # Classification loop
            logging.info(f"Starting classification for {dataset_name} with {prompt_type} from index {start_idx}...")
            start_time = time.time()

            batch_size = args.batch_size
            num_samples = len(df)
            for batch_start in range(start_idx, num_samples, batch_size):
                batch_end = min(batch_start + batch_size, num_samples)
                batch_rows = df.iloc[batch_start:batch_end]
                texts = batch_rows['TEXT'].tolist()
                true_labels_batch = batch_rows['ROLE'].tolist()

                # Format prompts
                prompts = [prompt_template.format(text=text) for text in texts]
                # Tokenize as a batch
                inputs = tokenizer(prompts, return_tensors="pt", truncation=True, max_length=2048, padding=True)
                inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        **generation_params
                    )
                # Decode responses
                for i in range(len(texts)):
                    input_len = (inputs['input_ids'][i] != tokenizer.pad_token_id).sum().item()
                    full_response = tokenizer.decode(outputs[i], skip_special_tokens=True)
                    prompt_length = len(tokenizer.decode(inputs['input_ids'][i], skip_special_tokens=True))
                    response_text = full_response[prompt_length:].strip()
                    classification = extract_classification_from_response(response_text)
                    confidence = calculate_confidence_score(response_text)
                    result = {
                        'id': batch_rows.iloc[i].get('ID', batch_start + i),
                        'name': batch_rows.iloc[i].get('NAME', ''),
                        'text': texts[i],
                        'true_label': true_labels_batch[i],
                        'predicted_label': classification,
                        'confidence': confidence,
                        'raw_response': response_text,
                        'full_response': full_response,
                        'formatted_prompt': prompts[i],
                        'input_tokens': input_len,
                        'output_tokens': outputs[i].shape[0] - input_len
                    }
                    results.append(result)
                    true_labels.append(true_labels_batch[i])
                    predicted_labels.append(classification)

                # Save checkpoint periodically
                if (batch_end) % args.checkpoint_interval == 0:
                    save_checkpoint(output_dir, f"{dataset_name}_{prompt_type}", results, true_labels, predicted_labels, batch_end, len(df))
                    temp_df = pd.DataFrame(results)
                    temp_file = os.path.join(output_dir, f'temp_results_{dataset_name}_{prompt_type}_{batch_end}.csv')
                    temp_df.to_csv(temp_file, index=False)
                    logging.info(f"Intermediate results saved: {temp_file}")

                if args.debug and batch_end % 50 == 0:
                    logging.info(f"Processed {batch_end}/{len(df)} samples for {dataset_name} with {prompt_type}")

            # Calculate processing time
            end_time = time.time()
            processing_time = end_time - start_time
            logging.info(f"Classification completed for {dataset_name} with {prompt_type} in {processing_time:.2f} seconds")

            # Get GPU stats
            gpu_stats = get_gpu_stats()

            # Create results DataFrame
            results_df = pd.DataFrame(results)
            
            # Save raw outputs log
            logging.info(f"Saving raw outputs log for {dataset_name} with {prompt_type}...")
            raw_log_file = save_raw_outputs_log(output_dir, f"{dataset_name}_{prompt_type}", results_df)

            # Generate performance report
            logging.info(f"Generating performance report for {dataset_name} with {prompt_type}...")
            report_file = generate_performance_report(
                true_labels, predicted_labels, results_df, output_dir, 
                args.model_name, f"{prompt_type}_{current_time}", gpu_stats, f"{dataset_name}_{prompt_type}"
            )

            # Save detailed results
            logging.info(f"Saving detailed results for {dataset_name} with {prompt_type}...")
            results_file = save_detailed_results(
                results_df, output_dir, args.model_name, f"{prompt_type}_{current_time}", f"{dataset_name}_{prompt_type}"
            )

            # Create confusion matrix plot
            logging.info(f"Creating confusion matrix for {dataset_name} with {prompt_type}...")
            cm_file = create_confusion_matrix_plot(
                true_labels, predicted_labels, output_dir, args.model_name.replace("/", "_"), f"{dataset_name}_{prompt_type}"
            )

            # Generate detailed classification report
            logging.info(f"Generating detailed classification report for {dataset_name} with {prompt_type}...")
            detailed_report_file = save_detailed_classification_report(
                results_df, output_dir, args.model_name, f"{prompt_type}_{current_time}", f"{dataset_name}_{prompt_type}"
            )

            # Save JSON results
            logging.info(f"Saving JSON results for {dataset_name} with {prompt_type}...")
            json_file = save_json_results(
                results_df, output_dir, args.model_name, f"{prompt_type}_{current_time}", f"{dataset_name}_{prompt_type}", gpu_stats
            )

            # Print summary for this dataset
            accuracy = accuracy_score(true_labels, predicted_labels)
            print(f"Dataset {dataset_name} with {prompt_type}:")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Processing time: {processing_time:.2f} seconds")
            print(f"  Results saved to: {output_dir}")
            if raw_log_file:
                print(f"  Raw outputs log: {raw_log_file}")

            return {
                'df': results_df,
                'true_labels': true_labels,
                'predicted_labels': predicted_labels,
                'processing_time': processing_time,
                'prompt_type': prompt_type
            }

        except Exception as e:
            logging.error(f"Error processing dataset {dataset_name} with {prompt_type}: {e}")
            print(f"Error processing dataset {dataset_name} with {prompt_type}: {e}")
            return None

    # Process each dataset
    all_results = {}
    
    # Determine which prompt types to run
    if args.prompt_type == 'all':
        prompt_types = ['zero_shot', 'one_shot', 'few_shot']
    else:
        prompt_types = [args.prompt_type]
    
    for dataset in test_datasets:
        dataset_name = dataset['name']
        all_results[dataset_name] = {}
        
        for prompt_type in prompt_types:
            result = process_dataset_with_prompt_type(
                dataset, prompt_type, model, tokenizer, generation_params, 
                accelerator, args, output_dir, current_time
            )
            
            if result:
                all_results[dataset_name][prompt_type] = result

    # Create comparison report if running all prompt types
    if args.prompt_type == 'all':
        logging.info("Creating prompt comparison report...")
        comparison_file = create_prompt_comparison_report(all_results, output_dir, args.model_name, current_time)
        if comparison_file:
            print(f"Prompt comparison report saved: {comparison_file}")

    # Print final summary
    print(f"\n" + "="*80)
    print("CLASSIFICATION COMPLETED!")
    print("="*80)
    
    total_accuracy = 0
    total_experiments = 0
    
    for dataset_name, prompt_results in all_results.items():
        print(f"\nDataset: {dataset_name}")
        print("-" * 40)
        
        for prompt_type, results in prompt_results.items():
            accuracy = accuracy_score(results['true_labels'], results['predicted_labels'])
            total_accuracy += accuracy
            total_experiments += 1
            print(f"  {prompt_type}: Accuracy = {accuracy:.4f}, Time = {results['processing_time']:.2f}s")
    
    if total_experiments > 0:
        avg_accuracy = total_accuracy / total_experiments
        print(f"\nOverall Average Accuracy: {avg_accuracy:.4f}")
        print(f"Total Experiments: {total_experiments}")
    
    print(f"\nAll results saved to: {output_dir}")
    print(f"Log file: {log_file}")

    # Log final summary
    logging.info("=== Classification Summary ===")
    for dataset_name, prompt_results in all_results.items():
        logging.info(f"Dataset: {dataset_name}")
        for prompt_type, results in prompt_results.items():
            accuracy = accuracy_score(results['true_labels'], results['predicted_labels'])
            logging.info(f"  {prompt_type}: Accuracy = {accuracy:.4f}, Time = {results['processing_time']:.2f}s")
    
    if total_experiments > 0:
        logging.info(f"Overall Average Accuracy: {avg_accuracy:.4f}")
        logging.info(f"Total Experiments: {total_experiments}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
