import wandb
import json
from dataclasses import dataclass
from utils import *
import yaml

# Configuration for sweep extraction
SWEEP_CONFIG = {
    "entity": "entity",  
    "project": "project",  
    "sweep_id": "sweep-id",  
    "run_name": "run_name", 
}

def get_best_hyperparameters_from_sweep(entity, project, sweep_id, run_name):
    """
    Extract the best hyperparameters from a completed wandb sweep
    
    Args:
        entity (str): wandb entity name
        project (str): wandb project name
        sweep_id (str): sweep ID
        run_id (str): run name
        
    Returns:
        dict: Best hyperparameters configuration
    """
    print("🔍 Connecting to wandb and fetching sweep results...")
    
    # Initialize wandb API
    api = wandb.Api()
    
    # Get the sweep
    sweep_path = f"{entity}/{project}/{sweep_id}"
    sweep = api.sweep(sweep_path)

    for run in sweep.runs:
        if run.name == run_name:
            best_config = run.config
    
    print(f"⚙️  Best hyperparameters:")
    for key, value in best_config.items():
        print(f"   {key}: {value}")
    
    return best_config

def load_config_from_yaml(yaml_file_path):
    """
    Load configuration from a YAML file

    Args:
        yaml_file_path (str): Path to the YAML configuration file

    Returns:
        dict: Configuration dictionary
    """
    print(f"📄 Loading configuration from YAML file: {yaml_file_path}")

    with open(yaml_file_path, 'r') as file:
        yaml_config = yaml.safe_load(file)

    # Convert YAML structure to flat config (extract 'value' from each parameter)
    config = {}
    for key, value_dict in yaml_config.items():
        if isinstance(value_dict, dict) and 'value' in value_dict:
            config[key] = value_dict['value']
        else:
            config[key] = value_dict

    print(f"⚙️  Configuration loaded from YAML:")
    for key, value in config.items():
        print(f"   {key}: {value}")

    return config

@dataclass
class OptimizedTrainingConfig:
    """Configuration for training parameters optimized from wandb sweep"""
    
    # Model parameters
    model_name: str = "roberta-base"
    max_length: int = 512
    
    # Training parameters
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 3
    warmup_steps: int = 500
    weight_decay: float = 0.01
    
    # Regularization parameters
    l2_lambda: float = 0.001
    ewc_lambda: float = 1000.0
    dropout_rate: float = 0.1
    
    # Loss parameters
    use_focal_loss: bool = True
    focal_alpha: float = 1.0
    focal_gamma: float = 2.0
    use_ewc: bool = True
    
    # System parameters
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir: str = "./output"
    seed: int = 42
    
    @classmethod
    def from_wandb_config(cls, wandb_config):
        """Create TrainingConfig from wandb sweep results"""
        return cls(
            model_name=wandb_config.get("model_name", "roberta-base"),
            max_length=wandb_config.get("max_length", 512),
            batch_size=wandb_config.get("batch_size", 16),
            learning_rate=wandb_config.get("learning_rate", 2e-5),
            num_epochs=wandb_config.get("num_epochs", 3),
            warmup_steps=wandb_config.get("warmup_steps", 500),
            weight_decay=wandb_config.get("weight_decay", 0.01),
            l2_lambda=wandb_config.get("l2_lambda", 0.001),
            ewc_lambda=wandb_config.get("ewc_lambda", 1000.0),
            dropout_rate=wandb_config.get("dropout_rate", 0.1),
            use_focal_loss=wandb_config.get("use_focal_loss", True),
            focal_alpha=wandb_config.get("focal_alpha", 1.0),
            focal_gamma=wandb_config.get("focal_gamma", 2.0),
            use_ewc=wandb_config.get("use_ewc", True),
            device="cuda" if torch.cuda.is_available() else "cpu",
            output_dir=wandb_config.get("output_dir", "./output"),
            seed=wandb_config.get("seed", 42)
        )

def initialize_single_run(best_config, run_name=None):
    """
    Initialize a single wandb run with the best configuration
    
    Args:
        best_config (dict): Best hyperparameters from sweep
        run_name (str, optional): Name for the run
        
    Returns:
        wandb.Run: Initialized wandb run
    """
    if run_name is None:
        run_name = f"best-config-final-run"
    
    print(f"🚀 Initializing wandb run: {run_name}")
    
    model_name = best_config.get('model_name')
    print(f"📱 Model being used: {model_name}")


    # Initialize wandb run
    run = wandb.init(
        project=SWEEP_CONFIG["project"],
        entity=SWEEP_CONFIG["entity"],
        name=run_name,
        config=best_config,
        tags=["final-run", "best-config", "optimized", f"model-{model_name}"],
        notes=f"Single run with {model_name} using hyperparameters optimized from wandb sweep"
        #tags=["final-run", "best-config", "optimized"],
        #notes="Single run with hyperparameters optimized from wandb sweep"
    )
    
    print(f"✅ Wandb run initialized: {run.url}")
    return run

def run_single_training(strategy, data_source, up_sample, data_config=None, config_source="sweep", yaml_path=None):
    """
    Run a single training session with optimized hyperparameters
    
    Args:
        strategy (str): Training strategy ("multiclass" or "hierarchical_pairwise")
        data_config (dict): Data configuration parameters
    """
    
    # Step 1: Extract best hyperparameters from sweep
    print("=" * 60)
    print("STEP 1: Extracting best hyperparameters from sweep")
    print("=" * 60)
    if config_source == "sweep":
        best_config = get_best_hyperparameters_from_sweep(
            SWEEP_CONFIG["entity"],
            SWEEP_CONFIG["project"],
            SWEEP_CONFIG["sweep_id"],
            SWEEP_CONFIG['run_name']
        )
    elif config_source == "yaml":
        if yaml_path is None:
            raise ValueError("yaml_path must be provided when config_source='yaml'")
        best_config = load_config_from_yaml(yaml_path)
    else:
        raise ValueError("config_source must be either 'sweep' or 'yaml'")
    
    # Step 2: Create optimized training configuration
    print("\n" + "=" * 60)
    print("STEP 2: Creating optimized training configuration")
    print("=" * 60)
    
    config = OptimizedTrainingConfig.from_wandb_config(best_config)
    print(f"📋 Training configuration created:")
    print(f"   Model: {config.model_name}")
    print(f"   Learning Rate: {config.learning_rate}")
    print(f"   Batch Size: {config.batch_size}")
    print(f"   Epochs: {config.num_epochs}")
    print(f"   Dropout: {config.dropout_rate}")
    
    # Step 3: Initialize single wandb run
    print("\n" + "=" * 60)
    print("STEP 3: Initializing single wandb run")
    print("=" * 60)
    
    run_name = f"best-{strategy}-final"
    run = initialize_single_run(best_config, run_name)

    wandb.log({
        "model_architecture": config.model_name,
        "training_strategy": strategy,
        "upsampling_enabled": up_sample
    })
    
    # Step 4: Run training with optimized parameters
    print("\n" + "=" * 60)
    print("STEP 4: Running training with optimized parameters")
    print("=" * 60)
    
    try:
        # Initialize trainer with optimized config
        trainer = BullyingClassifierTrainer(config)
        
        # Load data (modify these parameters as needed)
        if data_config is None:
            data_config = {
                "directory": "role_off",
                "data_point": "en_data", 
                "fold": 0
            }

        if data_source == 'all':
        
            print(f"📂 Loading data: {data_config}")
            up_train, up_val, down_train, down_val, test_data, gold_data = get_en_data(
                data_config["directory"], 
                data_config["data_point"], 
                data_config["fold"], 
                strategy
            )

        elif data_source == 'friten':
            up_train, up_val, down_train, down_val, test_data, gold_data = get_friten_data("../data/all_data/"+data_config['directory'], data_config['fold'], strategy)


        
        # Print data balance
        print_data_balance_report(down_train, down_val, test_data, 'ROLE', gold_data)
        
        if up_sample == True:
            # Apply upsampling if needed
            print("\n🔄 Applying upsampling to training data...")
            down_train_final = upsample_minority_classes(down_train, 'ROLE', seed=config.seed)
            print(f"Training data after upsampling: {down_train_final.shape}")   
        else:
            down_train_final = down_train.copy()
        
        # Convert to lists and map labels
        train_texts = down_train_final['TEXT'].tolist()
        train_labels = down_train_final['ROLE'].apply(map_roles).tolist()
        val_texts = down_val['TEXT'].tolist()
        val_labels = down_val['ROLE'].apply(map_roles).tolist()

        test_texts = test_data['TEXT'].tolist()
        test_labels = test_data['ROLE'].apply(map_roles).tolist()
        gold_texts = gold_data['TEXT'].tolist()
        gold_labels = gold_data['ROLE'].apply(map_roles).tolist()
        
        print(f"\n📊 Final training data: {len(train_texts)} samples")
        print(f"📊 Final validation data: {len(val_texts)} samples")
        print(f"📊 Test data: {len(test_texts)} samples")
        print(f"📊 Gold data: {len(gold_texts)} samples")
        
        # Train based on strategy
        if strategy == "multiclass":
            print("\n🎯 Training multiclass classifier...")
            result = trainer.train_multiclass(
                train_texts, train_labels,
                val_texts, val_labels
            )
            
        elif strategy == "pair_wise":
            print("\n🎯 Training hierarchical pairwise classifiers...")
            result = trainer.train_hierarchical_pairwise_classifiers(
                train_texts, train_labels,
                val_texts, val_labels
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # Log final results
        print("\n✅ Training completed successfully!")
        print(f"📊 Final results: {result['results'] if 'results' in result else result}")

        print("\n" + "=" * 60)
        print("Evaluation on Test and Gold Data")
        print("=" * 60)
        if strategy == "multiclass":
            print("\n🔮 Making multiclass predictions on test data...")
            test_preds = trainer.predict_multiclass(result['model'], test_texts, test_labels)
            
            print("\n🔮 Making multiclass predictions on gold data...")
            gold_preds = trainer.predict_multiclass(result['model'], gold_texts, gold_labels)
            
            # Log predictions to wandb
            wandb.log({
                "test_predictions": test_preds,
                "gold_predictions": gold_preds,
                "test_samples": len(test_texts),
                "gold_samples": len(gold_texts)
            })

            test_f1 = f1_score(test_preds.true_label, test_preds.predicted_label, average='macro', zero_division=0.0)
            gold_f1 = f1_score(gold_preds.true_label, gold_preds.predicted_label, average='macro', zero_division=0.0)
            print("\n🔮 Printing multiclass classification-report on test data...")
            print(classification_report(y_true=test_preds.true_label, y_pred=test_preds.predicted_label))
            print("\n🔮 Printing multiclass classification-report on gold data...")
            print(classification_report(y_true=gold_preds.true_label, y_pred=gold_preds.predicted_label))
            print()

            csv_name = f"multiclass_{config.model_name.split('/')[-1]}_fl={config.use_focal_loss}_up={args.up_sample}_fold={data_config['fold']}"
            test_preds.to_csv(f"{csv_name}_TEST_{round(test_f1, 2)}.csv", index=False)
            gold_preds.to_csv(f"{csv_name}_GOLD_{round(gold_f1, 2)}.csv", index=False)

        elif strategy == "pair_wise":
            print("\n🔮 Making hierarchical pairwise predictions on test data...")
            pair_wise_preds = trainer.predict_hierarchical_pairwise(result['models'], test_texts, test_labels, uncertainty_threshold=0.1)
            
            print("\n🔮 Making hierarchical pairwise predictions on gold data...")
            pair_wise_gold = trainer.predict_hierarchical_pairwise(result['models'], gold_texts, gold_labels, uncertainty_threshold=0.1)
            
            # Log predictions to wandb
            wandb.log({
                "test_predictions": pair_wise_preds,
                "gold_predictions": pair_wise_gold,
                "test_samples": len(test_texts),
                "gold_samples": len(gold_texts)
            })

            test_f1 = f1_score(pair_wise_preds.true_label, pair_wise_preds.predicted_label, average='macro', zero_division=0.0)
            gold_f1 = f1_score(pair_wise_gold.true_label, pair_wise_gold.predicted_label, average='macro', zero_division=0.0)
            print("\n🔮 Printing multiclass classification-report on test data...")
            print(classification_report(y_true=pair_wise_preds.true_label, y_pred=pair_wise_preds.predicted_label))
            print("\n🔮 Printing multiclass classification-report on gold data...")
            print(classification_report(y_true=pair_wise_gold.true_label, y_pred=pair_wise_gold.predicted_label))
            print()
            csv_name = f"pairwise_{config.model_name.split('/')[-1]}_fl={config.use_focal_loss}_up={args.up_sample}_fold={data_config['fold']}"
            pair_wise_preds.to_csv(f"{csv_name}_TEST_{round(test_f1, 2)}.csv", index=False)
            pair_wise_gold.to_csv(f"{csv_name}_GOLD_{round(gold_f1, 2)}.csv", index=False)

        
        return result
        
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        raise e
    
    finally:
        # Always finish wandb run
        wandb.finish()
        print("🏁 Wandb run finished.")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_source', type=str, required=True,
                        choices=['yaml', 'sweep'])
    parser.add_argument('--data_source', type=str, required=True,
            choices=['all', 'friten'])
    parser.add_argument('--yaml_path', type=str, default="")
    parser.add_argument('--strategy', type=str, required=True,
                        choices=['multiclass', 'pair_wise'])
    parser.add_argument('--data_point', type=str, default='llm_both')
    parser.add_argument('--fold', type=str, default='1')
    parser.add_argument('--up_sample', action='store_true', default=False)
    
    # Add sweep-specific arguments
    parser.add_argument('--entity', type=str, default=None)
    parser.add_argument('--project', type=str, default=None)
    parser.add_argument('--sweep_id', type=str, default=None)
    parser.add_argument('--run_name', type=str, default=None)

    args = parser.parse_args()

    SWEEP_CONFIG = {}

    if args.config_source == 'sweep':
        SWEEP_CONFIG["entity"] = args.entity
        SWEEP_CONFIG["project"] = args.project
        SWEEP_CONFIG["sweep_id"] = args.sweep_id
        SWEEP_CONFIG["run_name"] = args.run_name
    else:
        SWEEP_CONFIG["entity"] = args.entity
        SWEEP_CONFIG["project"] = args.project

    return args, SWEEP_CONFIG

if __name__ == "__main__":

    args, SWEEP_CONFIG = get_args()
    

    TRAINING_STRATEGY = args.strategy
    DATA_CONFIG = {
        "directory": "role_all",
        "data_point": args.data_point,
        "fold": args.fold
    }

    CONFIG_SOURCE = args.config_source
    YAML_CONFIG_PATH = args.yaml_path

    try:
        result = run_single_training(
            strategy=TRAINING_STRATEGY,
            data_source = args.data_source,
            up_sample=args.up_sample,
            data_config=DATA_CONFIG,
            config_source=CONFIG_SOURCE,
            yaml_path=YAML_CONFIG_PATH if CONFIG_SOURCE == "yaml" else None
        )
        print("\n🎉 Single run completed successfully!")
        
    except Exception as e:
        print(f"\n💥 Single run failed: {str(e)}")
        raise e

