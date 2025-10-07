#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import required libraries
import torch                                                            # PyTorch deep learning framework
import pandas as pd                                                     # Data manipulation and analysis
import numpy as np                                                      # Numerical computing
import argparse                                                         # Command line argument parsing
import os                                                               # Operating system interface
import random                                                           # Random number generation
import time                                                             # Time-related functions
import json                                                             # For reading JSON files
import re                                                               # Regular expressions
import logging                                                          # Logging functionality
import sys                                                              # System-specific parameters and functions
from datetime import datetime                                           # Date and time handling
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed  # Hugging Face transformers
from accelerate import Accelerator                                      # Hardware acceleration
from pynvml import *                                                    # For NVIDIA GPU monitoring


def process_conversation_text(text: str, file_number: int, metadata: dict):
    """
    Extract lines in the form:
       N. USERNAME: message
    from `text`. Returns a list of dictionaries, one per line.
    """
    messages = []
    pattern = re.compile(r'^\s*(\d+)\.\s*([^:]+)\s*:(.*)$', re.MULTILINE)

    for match in pattern.finditer(text):
        sentence_number_str = match.group(1)
        username = match.group(2).strip()
        message_content = match.group(3).strip()

        row = {
            'File': file_number,
            'SentenceNumber': int(sentence_number_str),
            'Username': username,
            'Text': message_content,
            'Model_Name': metadata.get('Model_Name', ''),
            'Temperature': metadata.get('Temperature', ''),
            'Do_Sample': metadata.get('Do_Sample', ''),
            'Eos_Token_ID': metadata.get('Eos_Token_ID', ''),
            'Pad_Token_ID': metadata.get('Pad_Token_ID', ''),
            'Max_New_Tokens': metadata.get('Max_New_Tokens', ''),
            'Seed_Value': metadata.get('Seed_Value', ''),
            'Inference_Time(s)': metadata.get('Inference_Time(s)', ''),
            'GPU_Memory_Used_GB': metadata.get('GPU_Memory_Used_GB', ''),
            'GPU_Memory_Total_GB': metadata.get('GPU_Memory_Total_GB', ''),
            'GPU_Power_Usage_W': metadata.get('GPU_Power_Usage_W', ''),
            'Is_Reasoning_Model': metadata.get('Is_Reasoning_Model', False)
        }
        messages.append(row)

    return messages


def init_accelerator():
    """Initialize and return a hardware accelerator."""
    return Accelerator()


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
    
    # Add a try-catch block for more descriptive error handling
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
    
    # Handle tokenizer special tokens
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


def load_scenarios_from_json(file_path):
    """Load scenarios from JSON file"""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Error loading scenarios from {file_path}: {e}")
        print(f"Error loading scenarios from {file_path}: {e}")
        return None


def detect_scenario_format(scenarios):
    """
    Detect whether the loaded scenarios are in the standard format or the transformed format.
    
    Standard format: {"A": {"scenario": "...", "problem": "..."}}
    Transformed format: {"Lit_1": {"scenario": "...", "category": "...", "platforms": [...], "features": [...]}}
    
    Returns True if it's the transformed format, False otherwise.
    """
    if not scenarios or not isinstance(scenarios, dict):
        return False
    
    # Get the first scenario key
    first_key = next(iter(scenarios))
    first_scenario = scenarios[first_key]
    
    # Check if it has the transformed format's fields
    has_category = 'category' in first_scenario
    has_platforms = 'platforms' in first_scenario and isinstance(first_scenario['platforms'], list)
    has_features = 'features' in first_scenario and isinstance(first_scenario['features'], list)
    
    # If it has any of the transformed format's distinctive fields, consider it transformed
    return has_category or (has_platforms and has_features)


def load_prompt_template(file_path):
    """Load prompt template from txt file"""
    try:
        with open(file_path, 'r') as f:
            return f.read()
    except Exception as e:
        logging.error(f"Error loading prompt template from {file_path}: {e}")
        print(f"Error loading prompt template from {file_path}: {e}")
        return None


def get_gpu_stats():
    """Get GPU memory usage and power consumption"""
    try:
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)  # Assuming using first GPU
        info = nvmlDeviceGetMemoryInfo(handle)
        power = nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert mW to W
        
        return {
            'GPU_Memory_Used_GB': round(info.used / 1024**3, 2),  # Convert bytes to GB
            'GPU_Memory_Total_GB': round(info.total / 1024**3, 2),
            'GPU_Power_Usage_W': round(power, 2)
        }
    except Exception as e:
        logging.warning(f"Could not get GPU stats: {e}")
        return {
            'GPU_Memory_Used_GB': -1,
            'GPU_Memory_Total_GB': -1,
            'GPU_Power_Usage_W': -1
        }


def extract_reasoning_and_conversation(text: str):
    """
    Extract the reasoning and conversation sections from the generated text when using the reasoning model.
    Returns a tuple of (reasoning_section, conversation_section).
    """
    reasoning = ""
    conversation = ""
    
    # Extract the reasoning section
    reasoning_pattern = re.compile(r'<reasoning>(.*?)</reasoning>', re.DOTALL)
    reasoning_match = reasoning_pattern.search(text)
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()
    
    # Extract the conversation section
    conversation_pattern = re.compile(r'<conversation>(.*?)</conversation>', re.DOTALL)
    conversation_match = conversation_pattern.search(text)
    if conversation_match:
        conversation = conversation_match.group(1).strip()
    
    return reasoning, conversation


def save_experiment_summary(output_dir, args, scenarios, selected_scenarios, generation_parameters, seed_value, is_transformed_format, gpu_stats=None):
    """
    Save a comprehensive summary of the experiment settings to a text file.
    
    Args:
        output_dir: Directory to save the summary file
        args: Command line arguments
        scenarios: All scenarios from the JSON file
        selected_scenarios: Scenarios that were used in this run
        generation_parameters: Model generation parameters
        seed_value: Random seed used
        is_transformed_format: Whether the scenario format is transformed
        gpu_stats: GPU statistics if available
    """
    # Create a descriptive filename
    model_name_short = args.model_name.split('/')[-1] if not args.reasoning_model else "DeepSeek-R1-Distill"
    temp_str = f"temp{generation_parameters['temperature']:.1f}".replace('.', 'p')
    seed_info = f"seed{seed_value}" if not args.vary_seed else "varySeed"
    reasoning_tag = "_reasoned" if args.reasoning_model else ""
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    summary_file = os.path.join(
        output_dir,
        f'experiment_summary_{model_name_short}_{temp_str}_{seed_info}{reasoning_tag}_{current_time}.txt'
    )
    
    with open(summary_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("EXPERIMENT SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        # Date and time
        f.write(f"Date and Time: {current_time}\n\n")
        
        # Command line arguments
        f.write("COMMAND LINE ARGUMENTS:\n")
        f.write("-" * 50 + "\n")
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
        f.write("\n")
        
        # Model information
        f.write("MODEL INFORMATION:\n")
        f.write("-" * 50 + "\n")
        f.write(f"Model Name: {args.model_name}\n")
        f.write(f"Is Reasoning Model: {args.reasoning_model}\n")
        f.write("\n")
        
        # Generation parameters
        f.write("GENERATION PARAMETERS:\n")
        f.write("-" * 50 + "\n")
        for param, value in generation_parameters.items():
            f.write(f"{param}: {value}\n")
        f.write(f"Seed Value: {seed_value}\n")
        f.write(f"Vary Seed: {args.vary_seed}\n")
        f.write("\n")
        
        # GPU information if available
        if gpu_stats:
            f.write("GPU INFORMATION:\n")
            f.write("-" * 50 + "\n")
            f.write(f"GPU Memory Used: {gpu_stats.get('GPU_Memory_Used_GB', 'N/A')} GB\n")
            f.write(f"GPU Memory Total: {gpu_stats.get('GPU_Memory_Total_GB', 'N/A')} GB\n")
            f.write(f"GPU Power Usage: {gpu_stats.get('GPU_Power_Usage_W', 'N/A')} W\n")
            f.write("\n")
        
        # Scenario information
        f.write("SCENARIOS INFORMATION:\n")
        f.write("-" * 50 + "\n")
        f.write(f"Total Scenarios: {len(selected_scenarios)}\n")
        f.write(f"Scenario Format: {'Transformed' if is_transformed_format else 'Standard'}\n\n")
        
        # Detailed scenario information
        for idx, (key, scenario_data) in enumerate(selected_scenarios.items(), 1):
            f.write(f"Scenario {idx} (ID: {key}):\n")
            f.write(f"  Description: {scenario_data['scenario']}\n")
            
            if is_transformed_format:
                problem_type = scenario_data.get('category', scenario_data.get('problem', 'unknown'))
                platforms = scenario_data.get('platforms', [])
                features = scenario_data.get('features', [])
                
                f.write(f"  Problem Type: {problem_type}\n")
                f.write(f"  Platforms: {', '.join(platforms)}\n")
                f.write(f"  Features: {', '.join(features)}\n")
            else:
                f.write(f"  Problem Type: {scenario_data['problem']}\n")
            
            f.write("\n")
            
        f.write("=" * 80 + "\n")
        f.write("END OF EXPERIMENT SUMMARY\n")
        f.write("=" * 80 + "\n")
    
    logging.info(f"Saved experiment summary to: {summary_file}")
    
    return summary_file


def main():
    """
    Main function for generating synthetic cyberbullying conversations.
    """
    # Print welcome message and basic usage
    print("=" * 80)
    print("Synthetic Cyberbullying Conversation Generator")
    print("=" * 80)
    print("This script generates synthetic conversations for cyberbullying detection research.")
    print("Required files:")
    print("  - scenarios.json:       Contains the cyberbullying scenarios (standard format)")
    print("  - scenarios_transformed.json: Alternative format with platforms and features (transformed format)")
    print("  - prompt_template.txt:  Contains the prompt template for generation")
    print("  - reasoning_prompt_template.txt: Optional template for reasoning model")
    print("\nExample commands:")
    print("  python gen.py                            # Run with defaults")
    print("  python gen.py --scenario all             # Generate for all scenarios")
    print("  python gen.py --scenario Lit_1,Lit_2,Lit_3  # Generate for specific scenarios")
    print("  python gen.py --runs 5 --vary_seed       # Generate 5 conversations with different seeds")
    print("  python gen.py --scenarios_file custom.json --prompt_file custom.txt  # Use custom files")
    print("  python gen.py --reasoning_model          # Use DeepSeek-R1-Distill-Llama-70B from shared directory")
    print("  python gen.py --reasoning_model --reasoning_prompt_file custom_reasoning.txt  # Custom reasoning prompt")
    print("  python gen.py --scenarios_file scenarios_transformed.json    # Use transformed scenarios format")
    print("\nAvailable models in shared directory (/home/support/llm/):")
    print("  - Llama-3.1-8B")
    print("  - Mistral-7B-v0.3")
    print("  - Meta-Llama-3-8B")
    print("  - Qwen2.5-VL-7B-Instruct")
    print("  - DeepSeek-R1-Distill-Llama-70B (via --reasoning_model option)")
    print("  - DeepSeek-R1\n")
    
    # Set up command line argument parser
    parser = argparse.ArgumentParser(description="Generate conversations for cyberbullying detection.")
    parser.add_argument('--runs', type=int, default=1, help='Number of runs to generate conversations')
    parser.add_argument('--vary_seed', action='store_true', help='Vary seed for each run')
    parser.add_argument('--scenario', type=str, default=None, 
                        help='Scenario(s) to run: use "all" for all scenarios, or a comma-separated list of scenario IDs. If not provided, first scenario in file will be used as default.')
    parser.add_argument('--debug', action='store_true', default=False, help='Enable debug mode with additional print statements')
    parser.add_argument('--scenarios_file', type=str, default='scenarios_transformed.json', help='Path to JSON file containing scenarios. Format (standard or transformed) is automatically detected.')
    parser.add_argument('--prompt_file', type=str, default='prompt_template.txt', help='Path to txt file containing prompt template')
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-3.3-70B-Instruct', help='Name/path of the model to use')
    parser.add_argument('--reasoning_model', action='store_true', help='Use the DeepSeek-R1-Distill-Llama-70B reasoning model from shared directory')
    parser.add_argument('--reasoning_prompt_file', type=str, default='reasoning_prompt_template.txt', help='Path to txt file containing reasoning prompt template')
    parser.add_argument('--shared_model_path', type=str, default='/home/support/llm/', help='Path to shared model directory')
    args = parser.parse_args()

    # Initialize random seed for this execution
    seed_value = random.randint(0, 2**32 - 1)
    set_random_seed(seed_value)

    # Create directory structure for output organization
    current_day = datetime.now().strftime("%Y_%m_%d")  # e.g., "2023_11_15"
    day_dir = f'synthetic_conversations/{current_day}'
    os.makedirs(day_dir, exist_ok=True)
    
    if args.debug:
        print(f"Created directory: {day_dir}")

    # Create timestamped output directory
    current_time = datetime.now().strftime("%H_%M")  # e.g., "14_30"
    output_dir = os.path.join(day_dir, f'{current_time}_conversation_generation')
    os.makedirs(output_dir, exist_ok=True)

    # Configure logging system
    log_file_path = os.path.join(output_dir, 'conversation_generation.log')
    logging.basicConfig(
        filename=log_file_path,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Load scenarios from JSON file
    if not os.path.exists(args.scenarios_file):
        logging.error(f"Scenarios file {args.scenarios_file} not found. Exiting.")
        print(f"Error: Scenarios file {args.scenarios_file} not found. Please provide a valid scenarios file.")
        return 1
    
    scenarios = load_scenarios_from_json(args.scenarios_file)
    if scenarios is None:
        logging.error("Failed to load scenarios. Exiting.")
        print("Error: Failed to load scenarios. Please check the file format and try again.")
        return 1
    
    # Detect the scenario format
    is_transformed_format = detect_scenario_format(scenarios)
    format_type = "transformed" if is_transformed_format else "standard"
    logging.info(f"Loaded scenarios from {args.scenarios_file} (detected {format_type} format)")
    if args.debug:
        print(f"Loaded scenarios from {args.scenarios_file} (detected {format_type} format)")
    
    # If scenario not specified, use the first scenario in the file as default
    if args.scenario is None:
        # Get the first scenario key from the loaded file
        args.scenario = next(iter(scenarios.keys()))
        logging.info(f"No scenario specified, defaulting to first scenario: '{args.scenario}'")
        if args.debug:
            print(f"No scenario specified, defaulting to first scenario: '{args.scenario}'")
    
    # Handle comma-separated list of scenarios
    if args.scenario != 'all' and ',' in args.scenario:
        # Split by comma and strip whitespace
        scenario_list = [s.strip() for s in args.scenario.split(',')]
        
        # Validate each scenario exists
        invalid_scenarios = [s for s in scenario_list if s not in scenarios]
        if invalid_scenarios:
            logging.error(f"Scenarios not found: {invalid_scenarios}. Available scenarios: {list(scenarios.keys())}")
            print(f"Error: The following scenarios were not found: {invalid_scenarios}")
            print(f"Available scenarios: {list(scenarios.keys())}")
            return 1
        
        # Create a subset of scenarios
        selected_scenarios = {k: scenarios[k] for k in scenario_list}
        logging.info(f"Using subset of scenarios: {list(selected_scenarios.keys())}")
        if args.debug:
            print(f"Using subset of scenarios: {list(selected_scenarios.keys())}")
    
    # Handle single scenario
    elif args.scenario != 'all' and args.scenario not in scenarios:
        logging.error(f"Scenario '{args.scenario}' not found in scenarios file. Available scenarios: {list(scenarios.keys())}")
        print(f"Error: Scenario '{args.scenario}' not found in scenarios file.")
        print(f"Available scenarios: {list(scenarios.keys())}")
        return 1
    
    # Load prompt template from txt file
    if not os.path.exists(args.prompt_file):
        logging.error(f"Prompt file {args.prompt_file} not found. Exiting.")
        print(f"Error: Prompt file {args.prompt_file} not found. Please provide a valid prompt template file.")
        return 1
    
    prompt_template = load_prompt_template(args.prompt_file)
    if prompt_template is None:
        logging.error("Failed to load prompt template. Exiting.")
        print("Error: Failed to load prompt template. Please check the file format and try again.")
        return 1
    
    logging.info(f"Loaded prompt template from {args.prompt_file}")
    if args.debug:
        print(f"Loaded prompt template from {args.prompt_file}")
    
    # If using reasoning model, load the reasoning prompt template as well
    reasoning_prompt_template = None
    if args.reasoning_model:
        if not os.path.exists(args.reasoning_prompt_file):
            logging.error(f"Reasoning prompt file {args.reasoning_prompt_file} not found. Exiting.")
            print(f"Error: Reasoning prompt file {args.reasoning_prompt_file} not found. Please provide a valid reasoning prompt template file.")
            return 1
        
        reasoning_prompt_template = load_prompt_template(args.reasoning_prompt_file)
        if reasoning_prompt_template is None:
            logging.error("Failed to load reasoning prompt template. Exiting.")
            print("Error: Failed to load reasoning prompt template. Please check the file format and try again.")
            return 1
        
        logging.info(f"Loaded reasoning prompt template from {args.reasoning_prompt_file}")
        if args.debug:
            print(f"Loaded reasoning prompt template from {args.reasoning_prompt_file}")

    # Select scenarios based on command line argument
    if args.scenario == 'all':
        selected_scenarios = scenarios
        logging.info(f"Using all {len(scenarios)} scenarios: {list(scenarios.keys())}")
        if args.debug:
            print(f"Using all {len(scenarios)} scenarios: {list(scenarios.keys())}")
    elif ',' in args.scenario:
        # Already handled above
        pass
    else:
        selected_scenarios = {args.scenario: scenarios[args.scenario]}
        logging.info(f"Using single scenario: {args.scenario}")
        if args.debug:
            print(f"Using single scenario: {args.scenario}")
    
    # Create scenario-specific directories
    for scenario_key in selected_scenarios.keys():
        scenario_dir = os.path.join(output_dir, f'Scenario_{scenario_key}')
        os.makedirs(scenario_dir, exist_ok=True)
        if args.debug:
            print(f"Created scenario directory: {scenario_dir}")

    # Define parameters for text generation
    generation_parameters = {
        'temperature': 0.8,        # Controls randomness: higher = more random
        'do_sample': True,         # Use sampling instead of greedy decoding
        'eos_token_id': None,      # Will be set after tokenizer initialization
        'pad_token_id': None,      # Will be set after tokenizer initialization
        'max_new_tokens': 3072,    # Maximum length of the conversation
        'top_p': 0.92,             # Nucleus sampling parameter
        'top_k': 40,               # Top-k sampling parameter
        'repetition_penalty': 1.15 # Penalize repeated tokens
    }

    # Log initial configuration
    logging.info("=== Conversation Generation Started ===")
    logging.info(f"Configuration:")
    logging.info(f"- Model: {args.model_name}")
    logging.info(f"- Number of runs: {args.runs}")
    logging.info(f"- Scenario mode: {args.scenario}")
    logging.info(f"- Debug mode: {args.debug}")
    logging.info(f"- Vary seed: {args.vary_seed}")
    logging.info(f"- Scenarios file: {args.scenarios_file}")
    logging.info(f"- Prompt file: {args.prompt_file}")
    logging.info(f"- Initial seed value: {seed_value}")
    logging.info(f"- Generation parameters:")
    for param, value in generation_parameters.items():
        logging.info(f"  - {param}: {value}")
    logging.info("=====================================")

    # Initialize the language model and tokenizer
    logging.info("Loading tokenizer and model...")
    if args.debug:
        print("Loading tokenizer and model...")

    try:
        if args.reasoning_model:
            model_name = "DeepSeek-R1-Distill-Llama-70B"
            model, tokenizer = init_model(model_name, is_shared=True, shared_path=args.shared_model_path)
            logging.info(f"Reasoning model loaded successfully from shared path: {args.shared_model_path}{model_name}")
            if args.debug:
                print(f"Reasoning model loaded successfully from shared path: {args.shared_model_path}{model_name}")
        else:
            model, tokenizer = init_model(args.model_name)
            logging.info(f"Model loaded successfully: {args.model_name}")
            if args.debug:
                print(f"Model loaded successfully: {args.model_name}")
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        if args.debug:
            print(f"Error loading model: {e}")
        return 1

    # Update generation parameters with tokenizer-specific values
    generation_parameters['eos_token_id'] = tokenizer.eos_token_id
    generation_parameters['pad_token_id'] = tokenizer.pad_token_id

    # Initialize hardware accelerator
    accelerator = init_accelerator()
    
    # Prepare model for hardware acceleration
    model = accelerator.prepare(model)
    model.eval()  # Set model to evaluation mode

    # Initialize storage for generated conversations
    all_conversations_combined = []
    all_messages = []  # List to store processed messages

    # Main generation loop
    for run in range(args.runs):
        logging.info(f"\nStarting Run {run + 1}/{args.runs}")
        
        for key, scenario_data in selected_scenarios.items():
            logging.info(f"\n--- Processing Scenario {key} ---")
            logging.info(f"Scenario description: {scenario_data['scenario']}")
            
            if is_transformed_format:
                logging.info(f"Problem type: {scenario_data.get('category', scenario_data.get('problem', 'unknown'))}")
                platforms_str = ", ".join(scenario_data.get('platforms', []))
                features_str = ", ".join(scenario_data.get('features', []))
                logging.info(f"Platforms: {platforms_str}")
                logging.info(f"Features: {features_str}")
            else:
                logging.info(f"Problem type: {scenario_data['problem']}")
            
            if args.vary_seed:
                # Generate new seed for this iteration
                seed_value = random.randint(0, 2**32 - 1)
                set_random_seed(seed_value)
                logging.info(f"New seed value for this iteration: {seed_value}")
            
            # Before generation
            logging.info("Generating conversation...")
            start_time = time.time()
            
            # Construct prompt for the current scenario
            try:
                # Prepare format parameters based on scenario format
                format_params = {
                    'scenario': scenario_data['scenario']
                }
                
                # Add problem field using the appropriate key based on format
                if is_transformed_format:
                    format_params['problem'] = scenario_data.get('category', scenario_data.get('problem', 'unknown'))
                    platforms_str = ", ".join(scenario_data.get('platforms', []))
                    features_str = ", ".join(scenario_data.get('features', []))
                    
                    # Add conditional sections
                    format_params['platforms_section'] = f"Platforms: \"{platforms_str}\"" if platforms_str else ""
                    format_params['features_section'] = f"Features: \"{features_str}\"" if features_str else ""
                else:
                    format_params['problem'] = scenario_data['problem']
                    # Add empty sections for standard format
                    format_params['platforms_section'] = ""
                    format_params['features_section'] = ""
                
                # Format the prompt with the parameters
                if args.reasoning_model and reasoning_prompt_template:
                    prompt = reasoning_prompt_template.format(**format_params)
                else:
                    prompt = prompt_template.format(**format_params)
                    
            except KeyError as e:
                logging.error(f"Error formatting prompt template: {e}")
                logging.error("Your prompt template may contain unescaped curly braces {} that should be escaped as {{ }}.")
                print(f"Error formatting prompt: {e}")
                print("Check your prompt_template.txt file. If it contains literal curly braces {}, they need to be escaped as {{ }}.")
                # Let's try escaping braces in the template before formatting
                try:
                    # Replace any standalone '{' or '}' not part of a format specifier with doubled braces
                    if args.reasoning_model and reasoning_prompt_template:
                        template_to_escape = reasoning_prompt_template
                    else:
                        template_to_escape = prompt_template
                        
                    escaped_template = re.sub(r'(?<!\{)\{(?!\{)', '{{', template_to_escape)
                    escaped_template = re.sub(r'(?<!\})\}(?!\})', '}}', escaped_template)
                    
                    # Keep format specifiers
                    escaped_template = escaped_template.replace('{{scenario}}', '{scenario}')
                    escaped_template = escaped_template.replace('{{problem}}', '{problem}')
                    escaped_template = escaped_template.replace('{{platforms_section}}', '{platforms_section}')
                    escaped_template = escaped_template.replace('{{features_section}}', '{features_section}')
                    
                    # Format with the same parameters as before
                    prompt = escaped_template.format(**format_params)
                    
                    logging.info("Successfully formatted prompt with escaped braces")
                except Exception as nested_e:
                    logging.error(f"Failed to format even with escaped braces: {nested_e}")
                    return 1

            # Prepare input for model
            inputs = tokenizer(prompt, return_tensors="pt").to(accelerator.device)

            if args.debug:
                print(f"Tokenized prompt length: {len(inputs['input_ids'][0])}")
            
            # Generate conversation text
            outputs = model.generate(
                **inputs,
                **generation_parameters
            )
            
            # After generation
            end_time = time.time()
            inference_time = end_time - start_time
            gpu_stats = get_gpu_stats()
            
            # Log raw model output (token IDs)
            logging.info("\n----- RAW MODEL OUTPUT (TOKEN IDS) -----")
            logging.info(f"Raw output tensor: {outputs[0]}")
            logging.info("----- END OF RAW MODEL OUTPUT -----\n")
            
            # Decode with special tokens included
            raw_decoded_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
            logging.info("\n----- RAW DECODED OUTPUT (WITH SPECIAL TOKENS) -----")
            logging.info(raw_decoded_text)
            logging.info("----- END OF RAW DECODED OUTPUT -----\n")
            
            # Decode generated text and clean it
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True) 
            cleaned_text = generated_text[len(prompt):]
            
            # If using reasoning model, extract reasoning and conversation sections
            reasoning_section = ""
            if args.reasoning_model:
                reasoning_section, conversation_section = extract_reasoning_and_conversation(cleaned_text)
                if conversation_section:
                    cleaned_text = conversation_section
                
                # Save reasoning to a separate file if it exists
                if reasoning_section:
                    # Create a short model name for the filename
                    model_name_short = args.model_name.split('/')[-1] if not args.reasoning_model else "DeepSeek-R1-Distill"
                    
                    # Create descriptive filename components
                    temp_str = f"temp{generation_parameters['temperature']:.1f}".replace('.', 'p')
                    seed_info = f"seed{seed_value}" if not args.vary_seed else "varySeed"
                    
                    if args.scenario == 'all':
                        scenario_dir = os.path.join(output_dir, f'Scenario_{key}')
                        reasoning_file = os.path.join(
                            scenario_dir, 
                            f'reasoning_scenario{key}_{model_name_short}_{temp_str}_{seed_info}_run{run+1}_{current_time}.txt'
                        )
                    else:
                        reasoning_file = os.path.join(
                            output_dir, 
                            f'reasoning_scenario{key}_{model_name_short}_{temp_str}_{seed_info}_run{run+1}_{current_time}.txt'
                        )
                    
                    with open(reasoning_file, 'w') as f:
                        f.write(f"SCENARIO: {scenario_data['scenario']}\n")
                        
                        # Use the appropriate field based on the scenario format
                        if is_transformed_format:
                            problem_type = scenario_data.get('category', scenario_data.get('problem', 'unknown'))
                            platforms_str = ", ".join(scenario_data.get('platforms', []))
                            features_str = ", ".join(scenario_data.get('features', []))
                            
                            f.write(f"PROBLEM TYPE: {problem_type}\n")
                            f.write(f"PLATFORMS: {platforms_str}\n")
                            f.write(f"FEATURES: {features_str}\n\n")
                        else:
                            f.write(f"PROBLEM: {scenario_data['problem']}\n\n")
                        
                        f.write(f"REASONING:\n{reasoning_section}\n")
                    
                    logging.info(f"Saved reasoning to: {reasoning_file}")
                    if args.debug:
                        print(f"Saved reasoning to: {reasoning_file}")

            # Log generation statistics
            logging.info(f"Generation completed:")
            logging.info(f"- Inference time: {inference_time:.2f} seconds")
            logging.info(f"- Generated text length: {len(generated_text.split())} words")
            logging.info(f"- GPU Memory Used: {gpu_stats['GPU_Memory_Used_GB']:.2f} GB")
            logging.info(f"- GPU Memory Total: {gpu_stats['GPU_Memory_Total_GB']:.2f} GB")
            logging.info(f"- GPU Power Usage: {gpu_stats['GPU_Power_Usage_W']:.2f} W")
            
            # Log the complete generated conversation
            logging.info("\n----- COMPLETE GENERATED CONVERSATION -----")
            logging.info(f"Prompt:\n{prompt}")
            logging.info(f"Generated conversation:\n{cleaned_text}")
            logging.info("----- END OF GENERATED CONVERSATION -----\n")
            
            # Update the metadata dictionary with GPU stats
            metadata = {
                "Model_Name": args.model_name if not args.reasoning_model else "DeepSeek-R1-Distill-Llama-70B",
                "Temperature": generation_parameters['temperature'],
                "Do_Sample": generation_parameters['do_sample'],
                "Eos_Token_ID": generation_parameters['eos_token_id'],
                "Pad_Token_ID": generation_parameters['pad_token_id'],
                "Max_New_Tokens": generation_parameters['max_new_tokens'],
                "Seed_Value": seed_value,
                "Inference_Time(s)": inference_time,
                "GPU_Memory_Used_GB": gpu_stats['GPU_Memory_Used_GB'],
                "GPU_Memory_Total_GB": gpu_stats['GPU_Memory_Total_GB'],
                "GPU_Power_Usage_W": gpu_stats['GPU_Power_Usage_W'],
                "Is_Reasoning_Model": args.reasoning_model
            }
            
            # Parse the conversation lines
            messages = process_conversation_text(cleaned_text, run + 1, metadata)
            
            # Tag each parsed message with the scenario key
            for msg in messages:
                msg["Scenario"] = key
                
            all_messages.extend(messages)

            # Store generation metadata and results
            conversation_data = {
                "Run": run + 1,
                "Scenario": key,
                "Conversation": generated_text,
                "Model_Name": args.model_name if not args.reasoning_model else "DeepSeek-R1-Distill-Llama-70B",
                "Temperature": generation_parameters['temperature'],
                "Do_Sample": generation_parameters['do_sample'],
                "Eos_Token_ID": generation_parameters['eos_token_id'],
                "Pad_Token_ID": generation_parameters['pad_token_id'],
                "Max_New_Tokens": generation_parameters['max_new_tokens'],
                "Seed_Value": seed_value,
                "Inference_Time(s)": inference_time,
                "Is_Reasoning_Model": args.reasoning_model
            }
            all_conversations_combined.append(conversation_data)

    # Save results to CSV file(s)
    # List of columns to include in output
    output_columns = [
        'File', 'Scenario', 'SentenceNumber', 'Username', 'Text',
        'Model_Name', 'Temperature', 'Do_Sample', 'Eos_Token_ID',
        'Pad_Token_ID', 'Max_New_Tokens', 'Seed_Value',
        'Inference_Time(s)', 'GPU_Memory_Used_GB',
        'GPU_Memory_Total_GB', 'GPU_Power_Usage_W', 'Is_Reasoning_Model'
    ]
    
    # Create a short model name for the filename by extracting just the model name without path
    model_name_for_filename = args.model_name.split('/')[-1] if not args.reasoning_model else "DeepSeek-R1-Distill"
    
    # Include temperature in filename
    temp_str = f"temp{generation_parameters['temperature']:.1f}".replace('.', 'p')
    
    if args.scenario == 'all':
        # Save messages for each scenario
        for scenario_key in scenarios.keys():
            scenario_messages = [
                msg for msg in all_messages 
                if msg["Scenario"] == scenario_key
            ]
            if scenario_messages:
                df_messages = pd.DataFrame(scenario_messages)
                scenario_dir = os.path.join(output_dir, f'Scenario_{scenario_key}')
                os.makedirs(scenario_dir, exist_ok=True)
                
                # Create descriptive filename with metadata
                seed_info = f"seed{seed_value}" if not args.vary_seed else "varySeed"
                reasoning_tag = "_reasoned" if args.reasoning_model else ""
                output_file = os.path.join(
                    scenario_dir, 
                    f'scenario{scenario_key}_{model_name_for_filename}_{temp_str}_{seed_info}{reasoning_tag}_{current_time}.csv'
                )
                
                # Filter output_columns to only include columns that exist in the DataFrame
                available_columns = [col for col in output_columns if col in df_messages.columns]
                if not available_columns:
                    logging.warning(f"No expected columns found in DataFrame. Saving all columns instead.")
                    df_messages.to_csv(output_file, index=False)
                else:
                    logging.info(f"Saving {len(available_columns)} of {len(output_columns)} expected columns")
                    df_messages[available_columns].to_csv(output_file, index=False)
                
                # Log the descriptive filename
                logging.info(f"Saved scenario-specific messages to: {output_file}")

        # Save combined messages file
        df_combined_messages = pd.DataFrame(all_messages)
        
        # Create descriptive filename for combined file
        seed_info = f"seed{seed_value}" if not args.vary_seed else "varySeed"
        reasoning_tag = "_reasoned" if args.reasoning_model else ""
        num_scenarios = len(selected_scenarios.keys())
        combined_output_file = os.path.join(
            output_dir, 
            f'all_{num_scenarios}scenarios_{model_name_for_filename}_{temp_str}_{seed_info}{reasoning_tag}_{current_time}.csv'
        )
        
        # Filter output_columns to only include columns that exist in the DataFrame
        available_columns = [col for col in output_columns if col in df_combined_messages.columns]
        if not available_columns:
            logging.warning(f"No expected columns found in DataFrame. Saving all columns instead.")
            df_combined_messages.to_csv(combined_output_file, index=False)
        else:
            logging.info(f"Saving {len(available_columns)} of {len(output_columns)} expected columns")
            df_combined_messages[available_columns].to_csv(combined_output_file, index=False)
        logging.info(f"Saved combined messages to: {combined_output_file}")

    else:
        # Single scenario
        scenario_dir = os.path.join(output_dir, f'Scenario_{args.scenario}')
        os.makedirs(scenario_dir, exist_ok=True)
        df_messages = pd.DataFrame(all_messages)
        
        # Create descriptive filename with metadata
        seed_info = f"seed{seed_value}" if not args.vary_seed else "varySeed"
        reasoning_tag = "_reasoned" if args.reasoning_model else ""
        output_file = os.path.join(
            scenario_dir, 
            f'scenario{args.scenario}_{model_name_for_filename}_{temp_str}_{seed_info}{reasoning_tag}_{current_time}.csv'
        )
        
        # Filter output_columns to only include columns that exist in the DataFrame
        available_columns = [col for col in output_columns if col in df_messages.columns]
        if not available_columns:
            logging.warning(f"No expected columns found in DataFrame. Saving all columns instead.")
            df_messages.to_csv(output_file, index=False)
        else:
            logging.info(f"Saving {len(available_columns)} of {len(output_columns)} expected columns")
            df_messages[available_columns].to_csv(output_file, index=False)
        logging.info(f"Saved scenario messages to: {output_file}")

    # At the end of the script, add summary logging
    logging.info("\n=== Generation Summary ===")
    logging.info(f"Total scenarios processed: {len(selected_scenarios)}")
    logging.info(f"Total conversations generated: {len(all_conversations_combined)}")
    logging.info(f"Output directory: {output_dir}")
    logging.info("\nFinal GPU Statistics:")
    final_gpu_stats = get_gpu_stats()
    logging.info(f"- GPU Memory Used: {final_gpu_stats['GPU_Memory_Used_GB']:.2f} GB")
    logging.info(f"- GPU Memory Total: {final_gpu_stats['GPU_Memory_Total_GB']:.2f} GB")
    logging.info(f"- GPU Power Usage: {final_gpu_stats['GPU_Power_Usage_W']:.2f} W")
    
    # Save a comprehensive experiment summary
    summary_file = save_experiment_summary(
        output_dir, 
        args, 
        scenarios, 
        selected_scenarios, 
        generation_parameters, 
        seed_value, 
        is_transformed_format, 
        final_gpu_stats
    )
    
    logging.info("\nFiles generated:")
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            if file.endswith('.csv') or file.endswith('.txt'):
                file_path = os.path.join(root, file)
                file_size = os.path.getsize(file_path) / 1024  # Convert to KB
                logging.info(f"- {file_path} ({file_size:.2f} KB)")
    
    # Add summary of all conversations at the end of the log
    logging.info("\n====== SUMMARY OF ALL GENERATED CONVERSATIONS ======")
    for idx, conversation_data in enumerate(all_conversations_combined):
        logging.info(f"\n--- Conversation {idx+1} (Run {conversation_data['Run']}, Scenario {conversation_data['Scenario']}) ---")
        logging.info(f"Full conversation text:\n{conversation_data['Conversation']}")
        logging.info("------------------------------------------------------")
    
    logging.info("=== Generation Complete ===")
    
    return 0


if __name__ == "__main__":
    main()