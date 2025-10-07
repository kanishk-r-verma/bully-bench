from dependencies import *
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROLE_CLASSES = {
            'bully': 0,
            'bully_support': 1,
            'victim_support':2,
            'victim': 3
}

# Configuration for sweep extraction
SWEEP_CONFIG = {
    "entity": "entity",  
    "project": "project",  
    "sweep_id": "sweep-id",  
    "run_name": "run_name", 
}

@dataclass
class TrainingConfig:
    """Configuration for training parameters with wandb integration"""
    model_name: str = "roberta-base" # or "openai-community/gpt2-medium"
    max_length: int = 512 
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 3
    warmup_steps: int = 500
    weight_decay: float = 0.01
    l2_lambda: float = 0.001
    ewc_lambda: float = 1000.0
    dropout_rate: float = 0.1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir: str = "./output"
    seed: int = 42
    use_focal_loss: bool = True
    focal_alpha: float = 1.0
    focal_gamma: float = 2.0
    use_ewc: bool = True

class BullyingDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # prevents nans when probability is 0
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class TransformerClassifier(nn.Module):
    """Custom transformer classifier with configurable heads"""
    
    def __init__(self, model_name: str, num_classes: int, dropout_rate: float = 0.1):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.config.hidden_size, num_classes)
        self.num_classes = num_classes
    
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use pooler output if available, otherwise use [CLS] token
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            pooled_output = outputs.pooler_output
        else:
            if self.config.model_type == "gpt2" in self.config.model_type:
                pooled_output = outputs.last_hidden_state[:, -1, :] # last token
            else:
                pooled_output = outputs.last_hidden_state[:, 0]  # [CLS] token
        
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
        
        return type('ModelOutput', (), {
            'loss': loss,
            'logits': logits,
            'hidden_states': outputs.last_hidden_state
        })()


class BullyingClassifierTrainer:
    """Main trainer class for all classification strategies with wandb integration"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        
        # Add padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Class mappings
        self.role_classes = ['bully', 'bully_support', 'victim_support', 'victim']
        self.role_class_map = {
            'bully': 0,
            'bully_support': 1,  # enabler 
            'victim_support': 2,  # defender
            'victim': 3
        }

        # Set random seed
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        # Set random seed
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

    
    def create_model(self, num_classes: int) -> nn.Module:
        """Create a transformer model with custom classifier head"""
        model = TransformerClassifier(self.config.model_name, num_classes, self.config.dropout_rate)
        return model.to(self.device)
    
    def train_model(self, model: nn.Module, train_loader: DataLoader, 
                   val_loader: DataLoader, model_name: str = "model") -> Dict:
        """Training function with wandb logging"""
        
        optimizer = AdamW(model.parameters(), lr=self.config.learning_rate, 
                         weight_decay=self.config.weight_decay)
        
        total_steps = len(train_loader) * self.config.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.config.warmup_steps,
            num_training_steps=total_steps
        )
        
        best_val_f1 = 0
        best_val_loss = float('inf')
        training_history = []
        patience = 3
        threshold = 0.001
        epochs_without_improvement = 0

        for epoch in range(self.config.num_epochs):
            # Training phase
            model.train()
            total_loss = 0
            total_ce_loss = 0
            total_focal_loss = 0
            total_l2_loss = 0
            
            for batch in tqdm(train_loader, desc=f"Training {model_name} - Epoch {epoch+1}"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                optimizer.zero_grad()
                outputs = model(**batch)
                labels = batch['labels']
                
                # Choose loss function based on config
                if self.config.use_focal_loss:
                    criterion = FocalLoss(alpha=self.config.focal_alpha, gamma=self.config.focal_gamma)
                    ce_loss = criterion(outputs.logits, labels)
                else:
                    ce_loss = outputs.loss
                
                loss = ce_loss
                
                # Add L2 regularization
                l2_reg = 0
                for param in model.parameters():
                    l2_reg += torch.norm(param, 2) ** 2
                l2_loss = self.config.l2_lambda * l2_reg
                loss += l2_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                total_loss += loss.item()
                total_ce_loss += ce_loss.item()
                total_focal_loss += ce_loss.item() if self.config.use_focal_loss else 0
                total_l2_loss += l2_loss.item()
            
            avg_train_loss = total_loss / len(train_loader)
            avg_ce_loss = total_ce_loss / len(train_loader)
            avg_focal_loss = total_focal_loss / len(train_loader) if self.config.use_focal_loss else 0
            avg_l2_loss = total_l2_loss / len(train_loader)
            
            # Validation phase
            logger.info("**************** Begin Validation *******************")
            val_metrics = self.evaluate_model(model, val_loader)
            val_f1 = val_metrics['f1_macro']
            val_loss = val_metrics['avg_loss']
            
            # Log to wandb
            wandb.log({
                f'{model_name}_epoch': epoch + 1,
                f'{model_name}_train_loss': avg_train_loss,
                f'{model_name}_train_ce_loss': avg_ce_loss,
                f'{model_name}_train_focal_loss': avg_focal_loss,
                f'{model_name}_train_l2_loss': avg_l2_loss,
                f'{model_name}_val_loss': val_loss,
                f'{model_name}_val_f1_macro': val_f1,
                f'{model_name}_val_accuracy': val_metrics['accuracy'],
                f'{model_name}_learning_rate': scheduler.get_last_lr()[0] if scheduler else self.config.learning_rate
            })

            logger.info(f"{model_name} - Epoch {epoch+1}: "
                       f"Train Loss: {avg_train_loss:.4f}, "
                       f"Val F1: {val_f1:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
            
            if val_f1 > best_val_f1 + threshold and val_loss < best_val_loss + threshold:
                best_val_f1 = val_f1
                best_val_loss = val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs.")
                break
            
            training_history.append({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_f1': val_f1,
                'val_accuracy': val_metrics['accuracy']
            })
        
        wandb.log({
            f'{model_name}_final_best_f1': best_val_f1,
            f'{model_name}_final_best_loss': best_val_loss
        })

        return {
            'best_f1': best_val_f1,
            'history': training_history,
            'final_metrics': val_metrics
        }
    

    def evaluate_model(self, model: nn.Module, data_loader: DataLoader) -> Dict:
        """Evaluate model performance"""
        model.eval()
        all_preds = []
        all_labels = []
        total_loss = 0
        
        with torch.no_grad():
            for batch in data_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = model(**batch)
                
                total_loss += outputs.loss.item()
                
                preds = torch.argmax(outputs.logits, dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch['labels'].cpu().numpy())
        
        print(f"Unique predicted classes: {set(all_preds)}")
        print(f"Unique true classes: {set(all_labels)}")
        print()

        accuracy = accuracy_score(all_labels, all_preds)
        f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0.0)
        f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0.0)

        return {
            'accuracy': accuracy,
            'f1_weighted': f1_weighted,
            'f1_macro': f1_macro,
            'avg_loss': total_loss / len(data_loader),
            'classification_report': classification_report(all_labels, all_preds)
        }

    
    def train_multiclass(self, train_texts: List[str], train_labels: List[int],
                                    val_texts: List[str], val_labels: List[int]) -> Dict:
        """Multiclass classifier"""
        logger.info("Training multiclass classifier...")
        
        # Create datasets
        train_dataset = BullyingDataset(train_texts, train_labels, self.tokenizer, self.config.max_length)
        val_dataset = BullyingDataset(val_texts, val_labels, self.tokenizer, self.config.max_length)
        
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size)
        
        # Create and train model
        model = self.create_model(num_classes=4)  # 4 role classes
        results = self.train_model(model, train_loader, val_loader, model_name="multiclass")
        
        return {'model': model, 'results': results}
    
    def train_hierarchical_pairwise_classifiers(self, 
                                               train_texts: List[str], 
                                               train_labels: List[int],
                                               val_texts: List[str], 
                                               val_labels: List[int]) -> Dict:
        """Hierarchical 2-level pairwise classification (bully vs victim -> specific roles)"""
        logger.info("Training hierarchical 2-level pairwise classifiers...")
        
        results = {}
        models = {}
        
        # Level 1: Train bully vs victim classifier (f1)
        # Map original 4-class labels to binary: bully/enabler = 1, victim/defender = 0
        logger.info("Training Level-1: bully vs victim classifier...")
        
        level1_train_labels = []
        level1_val_labels = []

        for label in train_labels:
            if label in [0, 1]:  # bully or bully_support (enabler)
                level1_train_labels.append(1)  # bully side
            else:  # victim or victim_support (defender)
                level1_train_labels.append(0)  # victim side
        
        for label in val_labels:
            if label in [0, 1]:  # bully or bully_support (enabler)
                level1_val_labels.append(1)  # bully side
            else:  # victim or victim_support (defender)
                level1_val_labels.append(0)  # victim side

        print("*"*60)
        print("Data balance for train data for level-1")
        print(Counter(level1_train_labels))
        print("Data balance for val data for level-1")
        print(Counter(level1_val_labels))
        print("*"*60)

        # Create datasets for Level 1
        level1_train_dataset = BullyingDataset(train_texts, level1_train_labels, 
                                             self.tokenizer, self.config.max_length)
        level1_val_dataset = BullyingDataset(val_texts, level1_val_labels, 
                                           self.tokenizer, self.config.max_length)
        
        level1_train_loader = DataLoader(level1_train_dataset, batch_size=self.config.batch_size, shuffle=True)
        level1_val_loader = DataLoader(level1_val_dataset, batch_size=self.config.batch_size)
        
        # Train Level 1 model
        level1_model = self.create_model(num_classes=2)
        level1_results = self.train_model(level1_model, level1_train_loader, level1_val_loader, 
                                        model_name="level1_bully_vs_victim")
        
        models['level1_bully_vs_victim'] = level1_model
        results['level1_bully_vs_victim'] = level1_results
        
        # Level 2a: Train bully vs enabler classifier (f2a) - only on bully-side samples
        logger.info("Training Level-2a: bully vs enabler classifier...")
        
        level2a_train_texts = []
        level2a_train_labels = []
        level2a_val_texts = []
        level2a_val_labels = []
        
        # Filter training data for bully-side samples only
        for text, label in zip(train_texts, train_labels):
            if label in [0, 1]:  # bully or enabler
                level2a_train_texts.append(text)
                level2a_train_labels.append(0 if label == 0 else 1)  # 0=bully, 1=enabler
        
        for text, label in zip(val_texts, val_labels):
            if label in [0, 1]:  # bully or enabler
                level2a_val_texts.append(text)
                level2a_val_labels.append(0 if label == 0 else 1)  # 0=bully, 1=enabler
        
        
        print("*"*60)
        print("Training label distro for level2a-bully_v_enabler")
        print(Counter(level2a_train_labels))
        print("Validation label distro for level2a-bully_v_enabler")
        print(Counter(level2a_val_labels))
        print("*"*60)


        if len(level2a_train_texts) > 0:
            level2a_train_dataset = BullyingDataset(level2a_train_texts, level2a_train_labels, 
                                                  self.tokenizer, self.config.max_length)
            level2a_val_dataset = BullyingDataset(level2a_val_texts, level2a_val_labels, 
                                                self.tokenizer, self.config.max_length)
            
            level2a_train_loader = DataLoader(level2a_train_dataset, batch_size=self.config.batch_size, shuffle=True)
            level2a_val_loader = DataLoader(level2a_val_dataset, batch_size=self.config.batch_size)
            
            level2a_model = self.create_model(num_classes=2)
            level2a_results = self.train_model(level2a_model, level2a_train_loader, level2a_val_loader, 
                                             model_name="level2a_bully_vs_enabler")
            
            models['level2a_bully_vs_enabler'] = level2a_model
            results['level2a_bully_vs_enabler'] = level2a_results
        else:
            logger.warning("No training data for Level-2a: bully vs enabler")
        
        # Level 2b: Train victim vs defender classifier (f2b) - only on victim-side samples
        logger.info("Training Level-2b: victim vs defender classifier...")
        
        level2b_train_texts = []
        level2b_train_labels = []
        level2b_val_texts = []
        level2b_val_labels = []
        
        # Filter training data for victim-side samples only
        for text, label in zip(train_texts, train_labels):
            if label in [2, 3]:  # defender or victim
                level2b_train_texts.append(text)
                level2b_train_labels.append(0 if label == 3 else 1)  # 0=victim, 1=defender
        
        for text, label in zip(val_texts, val_labels):
            if label in [2, 3]:  # defender or victim
                level2b_val_texts.append(text)
                level2b_val_labels.append(0 if label == 3 else 1)  # 0=victim, 1=defender
        
        print("*"*60)
        print("Training distro for level2-victim_v_defender")
        print(Counter(level2b_train_labels))
        print("Validation distro for level2-victim_v_defender")
        print(Counter(level2b_val_labels))
        print()

        if len(level2b_train_texts) > 0:
            level2b_train_dataset = BullyingDataset(level2b_train_texts, level2b_train_labels, 
                                                  self.tokenizer, self.config.max_length)
            level2b_val_dataset = BullyingDataset(level2b_val_texts, level2b_val_labels, 
                                                self.tokenizer, self.config.max_length)
            
            level2b_train_loader = DataLoader(level2b_train_dataset, batch_size=self.config.batch_size, shuffle=True)
            level2b_val_loader = DataLoader(level2b_val_dataset, batch_size=self.config.batch_size)
            
            level2b_model = self.create_model(num_classes=2)
            level2b_results = self.train_model(level2b_model, level2b_train_loader, level2b_val_loader, 
                                             model_name="level2b_victim_vs_defender")
            
            models['level2b_victim_vs_defender'] = level2b_model
            results['level2b_victim_vs_defender'] = level2b_results
        else:
            logger.warning("No training data for Level-2b: victim vs defender")
        
        return {'models': models, 'results': results}


    def predict_hierarchical_pairwise(self, models: Dict[str, nn.Module], test_texts: List[str], 
                                          test_labels: Optional[List[int]] = None,
                                          uncertainty_threshold: float = 0.3) -> pd.DataFrame:
        """
        Predict using hierarchical 2-level pairwise approach with uncertainty thresholds and joint probability computation
    
        Computes P(role|x) for all 4 roles: bully, bully_support (enabler), victim, victim_support (defender)
        Then applies uncertainty thresholding on the final normalized probabilities
        
        """
        results = []
    
        # Set all models to eval mode
        for model in models.values():
            model.eval()
        
        level1_model = models.get('level1_bully_vs_victim')
        level2a_model = models.get('level2a_bully_vs_enabler') 
        level2b_model = models.get('level2b_victim_vs_defender')
        
        if level1_model is None:
            raise ValueError("Level 1 model (bully vs victim) not found")
        
        # Role mapping for final output
        role_names = ['bully', 'bully_support', 'victim_support', 'victim']
        
        for i, text in enumerate(test_texts):
            encoding = self.tokenizer(text, padding='max_length', truncation=True,
                                    max_length=self.config.max_length, return_tensors="pt")
            
            input_ids = encoding["input_ids"].to(self.device)
            attention_mask = encoding["attention_mask"].to(self.device)
            
            with torch.no_grad():
                # Level 1: Get full probability distribution over bully_side vs victim_side
                level1_outputs = level1_model(input_ids=input_ids, attention_mask=attention_mask)
                level1_probs = F.softmax(level1_outputs.logits, dim=-1).squeeze()
                
                # P(victim_side|x), P(bully_side|x)
                prob_victim_side = level1_probs[0].item()  # victim_side
                prob_bully_side = level1_probs[1].item()   # bully_side
                
                # Initialize joint probabilities for all 4 roles
                role_probs = torch.zeros(4)  # [bully, bully_support, victim_support, victim]
                
                # Compute joint probabilities via conditional paths
                
                # Bully side path: P(bully|x) and P(bully_support|x)
                if level2a_model is not None:
                    level2a_outputs = level2a_model(input_ids=input_ids, attention_mask=attention_mask)
                    level2a_probs = F.softmax(level2a_outputs.logits, dim=-1).squeeze()
                    
                    # P(bully|x, bully_side) and P(bully_support|x, bully_side)
                    prob_bully_given_bully_side = level2a_probs[0].item()      # bully
                    prob_enabler_given_bully_side = level2a_probs[1].item()    # bully_support
                    
                    # Joint probabilities: P(role|x) = P(side|x) * P(role|x, side)
                    role_probs[0] = prob_bully_side * prob_bully_given_bully_side      # P(bully|x)
                    role_probs[1] = prob_bully_side * prob_enabler_given_bully_side    # P(bully_support|x)
                else:
                    # Fallback: distribute bully_side probability equally
                    role_probs[0] = prob_bully_side * 0.5  # P(bully|x)
                    role_probs[1] = prob_bully_side * 0.5  # P(bully_support|x)
                
                # Victim side path: P(victim|x) and P(victim_support|x)
                if level2b_model is not None:
                    level2b_outputs = level2b_model(input_ids=input_ids, attention_mask=attention_mask)
                    level2b_probs = F.softmax(level2b_outputs.logits, dim=-1).squeeze()
                    
                    # P(victim|x, victim_side) and P(victim_support|x, victim_side)
                    prob_victim_given_victim_side = level2b_probs[0].item()        # victim
                    prob_defender_given_victim_side = level2b_probs[1].item()      # victim_support
                    
                    # Joint probabilities: P(role|x) = P(side|x) * P(role|x, side)
                    role_probs[2] = prob_victim_side * prob_defender_given_victim_side  # P(victim_support|x)
                    role_probs[3] = prob_victim_side * prob_victim_given_victim_side    # P(victim|x)
                else:
                    # Fallback: distribute victim_side probability equally
                    role_probs[2] = prob_victim_side * 0.5  # P(victim_support|x)
                    role_probs[3] = prob_victim_side * 0.5  # P(victim|x)
                
                # Normalize to ensure probabilities sum to 1 (handles any numerical issues)
                role_probs = role_probs / role_probs.sum()
                
                # Final prediction: role with highest probability
                predicted_role_idx = torch.argmax(role_probs).item()
                predicted_role = role_names[predicted_role_idx]
                final_confidence = role_probs[predicted_role_idx].item()
                
                # Uncertainty detection: check if max probability is below threshold
                is_uncertain = final_confidence < uncertainty_threshold
                
                # Create detailed decision information
                role_prob_dict = {role: prob.item() for role, prob in zip(role_names, role_probs)}
                
                # Determine which path was most influential
                if predicted_role_idx in [0, 1]:  # bully side
                    dominant_path = "bully_side"
                    level2_model_used = "level2a" if level2a_model is not None else "fallback"
                else:  # victim side
                    dominant_path = "victim_side" 
                    level2_model_used = "level2b" if level2b_model is not None else "fallback"
                
                result = {
                    'text': text,
                    'predicted_label': 'uncertain' if is_uncertain else predicted_role,
                    'confidence': final_confidence,
                    'is_uncertain': is_uncertain,
                    'uncertainty_threshold': uncertainty_threshold,
                    
                    # Detailed probability breakdown
                    'prob_bully': role_prob_dict['bully'],
                    'prob_bully_support': role_prob_dict['bully_support'], 
                    'prob_victim_support': role_prob_dict['victim_support'],
                    'prob_victim': role_prob_dict['victim'],
                    
                    # Level-wise information
                    'level1_prob_bully_side': prob_bully_side,
                    'level1_prob_victim_side': prob_victim_side,
                    'dominant_path': dominant_path,
                    'level2_model_used': level2_model_used,
                    
                    # Decision explanation
                    'decision_rationale': f"Highest prob: {predicted_role} ({final_confidence:.3f})" + 
                                        (f" -> UNCERTAIN (< {uncertainty_threshold})" if is_uncertain else "")
                }
                
                if test_labels is not None:
                    true_label = [k for k, v in self.role_class_map.items() if v == test_labels[i]][0]
                    result['true_label'] = true_label
                    result['correct_prediction'] = (not is_uncertain) and (predicted_role == true_label)
                
                results.append(result)
        
        return pd.DataFrame(results)

    
    def predict_multiclass(self, model: nn.Module, test_texts: List[str], 
                          test_labels: Optional[List[int]] = None) -> pd.DataFrame:
        """Predict using multiclass model"""
        model.eval()
        results = []
        
        for i, text in enumerate(test_texts):
            encoding = self.tokenizer(text, padding='max_length', truncation=True,
                                    max_length=self.config.max_length, return_tensors="pt")
            
            input_ids = encoding["input_ids"].to(self.device)
            attention_mask = encoding["attention_mask"].to(self.device)
            
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                probs = F.softmax(outputs.logits, dim=-1).squeeze()
                pred_idx = torch.argmax(probs).item()
                confidence = probs[pred_idx].item()
            
            pred_label = [k for k, v in self.role_class_map.items() if v == pred_idx][0]
            
            result = {
                'text': text,
                'predicted_label': pred_label,
                'confidence': confidence
            }
            
            if test_labels is not None:
                true_label = [k for k, v in self.role_class_map.items() if v == test_labels[i]][0]
                result['true_label'] = true_label
            
            results.append(result)
        
        return pd.DataFrame(results)


def check_string_lengths(df):
    text_column = 'TEXT'
    text_norm = df[text_column].astype(str).str.strip().str.lower()
    match_hi = text_norm.str.match(r'^(hi\b|hey\b)', na=False)
    short_enough = text_norm.str.split().str.len() <= 3
    matched = df[match_hi & short_enough]
    not_matched = df[~(match_hi & short_enough)]

    return matched, not_matched


def fix_duplicates(df):
    role_counts = df.groupby('TEXT')['ROLE'].nunique().reset_index(name='role_nunique')
    hate_counts = df.groupby('TEXT')['HATE'].nunique().reset_index(name='hate_nunique')

    text_variation = pd.merge(role_counts, hate_counts, on='TEXT')

    conflicting_texts = text_variation[
        (text_variation['role_nunique'] > 1) | (text_variation['hate_nunique'] > 1)
    ]['TEXT']

    result = df[~(df['TEXT'].isin(conflicting_texts))]

    short, long_df = check_string_lengths(result)
    print("="*20)
    print(f"Original Size:\t{len(result)}")
    print(f"'Hi Size till 3':\t{len(short)} ")
    print(f'After del Size:\t{len(long_df)}')
    print("="*20)
    print()

    return long_df

def get_en_data(directory, data_point, fold, strategy):
    """Load and split English data"""

    # Load fold info and main CSV
    with open(f"../data/all_data/{directory}/{data_point}/{data_point}_trainEval.json", "r") as frfile:
        folds = json.load(frfile)
    fold_data = folds[f'fold-{fold}']

    df = pd.read_csv(f"../data/all_data/{directory}/{data_point}/{data_point}_trainEval.csv")
    df = fix_duplicates(df)

    # Extract train/val indices
    train_indices = fold_data['train']
    eval_indices = fold_data['val']

    # Apply HATE mapping always (needed for bin_label logic)
    df['bin_label'] = df.HATE.apply(fix_off_mapping)

    # If role_off and NOT multiclass, we want to keep only offensive samples for downstream
    if directory == 'role_off' and strategy != 'multiclass':
        df_down = df[df['bin_label'] == 'yes']
    else:
        df_down = df.copy()

    # Always keep all data for upstream
    df_up = df.copy()

    # Validate and filter indices
    train_indices_up = [idx for idx in train_indices if idx in df_up.index]
    eval_indices_up = [idx for idx in eval_indices if idx in df_up.index]
    train_indices_down = [idx for idx in train_indices if idx in df_down.index]
    eval_indices_down = [idx for idx in eval_indices if idx in df_down.index]

    # Upstream: HATE labels
    up_train = df_up[['TEXT', 'HATE']].loc[train_indices_up]
    up_val = df_up[['TEXT', 'HATE']].loc[eval_indices_up]
    up_train.HATE = up_train.HATE.apply(fix_off_mapping)
    up_val.HATE = up_val.HATE.apply(fix_off_mapping)

    # Downstream: ROLE labels
    down_train = df_down[['TEXT', 'ROLE']].loc[train_indices_down]
    down_val = df_down[['TEXT', 'ROLE']].loc[eval_indices_down]
    down_train.ROLE = down_train.ROLE.apply(fix_role_mapping)
    down_val.ROLE = down_val.ROLE.apply(fix_role_mapping)
    down_train = down_train[down_train.ROLE != 'none']
    down_val = down_val[down_val.ROLE != 'none']

    # Load test data
    updown_test = pd.read_csv(f"../data/all_data/{directory}/{data_point}/{data_point}_test.csv")
    updown_test = fix_duplicates(updown_test)
    updown_test.HATE = updown_test.HATE.apply(fix_off_mapping)
    updown_test.ROLE = updown_test.ROLE.apply(fix_role_mapping)
    updown_test = updown_test[updown_test.ROLE != 'none']

    # Load gold data
    up_gold = pd.read_csv(f"../data/all_data/{directory}/{data_point}/seed42_ty_proj1_context5.csv")
    updown_gold = up_gold[['TEXT', 'ROLE', 'HATE']]
    updown_gold = fix_duplicates(updown_gold)
    updown_gold.HATE = updown_gold.HATE.apply(fix_off_mapping)
    updown_gold.ROLE = updown_gold.ROLE.apply(fix_role_mapping)
    updown_gold = updown_gold[updown_gold.ROLE != 'none']

    return up_train, up_val, down_train, down_val, updown_test, updown_gold

from sklearn.utils import resample
import pandas as pd

def upsample_minority_classes(df_train: pd.DataFrame, label_column: str, seed: int = 42):
    """
    Upsample minority classes in the training set using random seed.
    
    Args:
        df_train (pd.DataFrame): Original training data
        label_column (str): Column name of the labels (e.g., "ROLE")
        seed (int): Random seed for reproducibility

    Returns:
        pd.DataFrame: New training data with upsampled classes
    """
    # Separate classes
    classes = df_train[label_column].unique()
    max_count = df_train[label_column].value_counts().max()
    
    df_upsampled = []

    for cls in classes:
        df_cls = df_train[df_train[label_column] == cls]
        if len(df_cls) < max_count:
            # Upsample
            df_cls_upsampled = resample(df_cls,
                                        replace=True,          # Sample with replacement
                                        n_samples=max_count,   # Match the majority class
                                        random_state=seed)
        else:
            df_cls_upsampled = df_cls

        df_upsampled.append(df_cls_upsampled)

    # Concatenate and shuffle
    df_train_balanced = pd.concat(df_upsampled).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df_train_balanced


def fix_off_mapping(label):
    if label in ['neutral', 'no']:
        return 'no'
    else:
        return 'yes'

def fix_role_mapping(label):
    if label in ['silent_bystander', 'none']:
        return 'none'
    else:
        return label


def print_data_balance_report(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, column_name, gold_df: pd.DataFrame):
    """Print comprehensive data balance report"""
    print("\n" + "="*60)
    print(f"DATA BALANCE REPORT {column_name}")
    print("="*60)

    for name, df in [("TRAIN", train_df), ("VALIDATION", val_df), ("TEST", test_df), ("GOLD TEST", gold_df)]:
        print(f"\n{name} SET:")
        print("-" * 40)
        role_counts = df[column_name].value_counts().sort_index()
        total = len(df)

        for role, count in role_counts.items():
            percentage = (count / total) * 100
            print(f"  {role:15}: {count:4d} ({percentage:5.1f}%)")
        print(f"  {'TOTAL':15}: {total:4d}")

        # Class imbalance ratio
        max_count = role_counts.max()
        min_count = role_counts.min()
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        print(f"  Imbalance ratio: {imbalance_ratio:.2f}:1")

    print("="*60)

def map_roles(label):
    return ROLE_CLASSES[label]

def map_binary(label):
    if label == 'yes':
        return 1
    else:
        return 0

def extract_only_trans(train, test):
    # Create boolean masks
    train_mask = ~(train.FILE.str.contains('Lit'))
    test_mask = ~(test.FILE.str.contains('Lit'))

    # Filter while preserving indexes
    train_filtered = train.loc[train_mask].copy()
    test_filtered = test.loc[test_mask].copy()

    # Store original indexes if needed
    train_original_indexes = train_filtered.index.tolist()
    test_original_indexes = test_filtered.index.tolist()

    return train_filtered, test_filtered


def fix_label_conflicts(df):
    # Track total before
    total_before = df.shape[0]

    # Count unique ROLEs and HATEs per TEXT
    role_counts = df.groupby('TEXT')['ROLE'].nunique().reset_index(name='role_nunique')
    hate_counts = df.groupby('TEXT')['HATE'].nunique().reset_index(name='hate_nunique')

    # Merge counts
    text_variation = pd.merge(role_counts, hate_counts, on='TEXT')

    # Find texts with inconsistent labels
    conflicting_texts = text_variation[
        (text_variation['role_nunique'] > 1) | (text_variation['hate_nunique'] > 1)
    ]['TEXT']

    # Filter out conflicting texts, preserving index
    df_clean = df[~df['TEXT'].isin(conflicting_texts)].copy()

    # Track total after
    total_after = df_clean.shape[0]
    removed = total_before - total_after

    # Print stats
    print(f"Total entries before cleaning: {total_before}")
    print(f"Total entries removed due to label conflicts: {removed}")
    print(f"Total entries after cleaning: {total_after}")
    print(f"Indexes preserved:", df_clean.index.is_unique and df_clean.index.equals(df.index.difference(df[df['TEXT'].isin(conflicting_texts)].index)))

    return df_clean


def get_friten_data(base_dir, fold, strategy):
    """Load and split French+Italian merged data with full checks"""

    import pandas as pd
    import json

    # Load CSVs
    print("\n📥 Loading data...")
    fren_trainEval = pd.read_csv(f"{base_dir}/llm_fren/llm_fren_trainEval.csv")
    fren_test = pd.read_csv(f"{base_dir}/llm_fren/llm_fren_test.csv")
    iten_trainEval = pd.read_csv(f"{base_dir}/llm_iten/llm_iten_trainEval.csv")
    iten_test = pd.read_csv(f"{base_dir}/llm_iten/llm_iten_test.csv")

    # Extract only translations
    fren_trainEval, fren_test = extract_only_trans(fren_trainEval, fren_test)
    iten_trainEval, iten_test = extract_only_trans(iten_trainEval, iten_test)

    print("\n🔍 Shapes after translation filter:")
    print(f"French trainEval: {fren_trainEval.shape}, test: {fren_test.shape}")
    print(f"Italian trainEval: {iten_trainEval.shape}, test: {iten_test.shape}")

    # Fix label conflicts
    print("\n🧹 Fixing label conflicts...")
    fren_trainEval = fix_label_conflicts(fren_trainEval)
    iten_trainEval = fix_label_conflicts(iten_trainEval)
    friten_trainEval = pd.concat([fren_trainEval, iten_trainEval])
    friten_test = pd.concat([fren_test, iten_test])

    print("\n📊 Combined shapes:")
    print(f"friten_trainEval: {friten_trainEval.shape}, friten_test: {friten_test.shape}")
    print("🧼 NaNs in trainEval:\n", friten_trainEval.isna().sum())
    print("🧼 Duplicates in trainEval:", friten_trainEval.duplicated().sum())

    print("🧪 Columns:", friten_trainEval.columns.tolist())

    # Load fold split
    print(f"\n📂 Loading fold-{fold}...")
    with open(f"{base_dir}/llm_fren/llm_fren_trainEval.json", "r") as frfile:
        folds = json.load(frfile)
    fold_data = folds[f'fold-{fold}']
    train_indices = fold_data['train']
    eval_indices = fold_data['val']

    # Binary label mapping for downstream logic
    friten_trainEval['bin_label'] = friten_trainEval.HATE.apply(fix_off_mapping)

    # Strategy-based filtering
    if strategy != 'multiclass':
        df_down = friten_trainEval[friten_trainEval['bin_label'] == 'yes']
        print("📥 Downstream (filtered for offensive samples only)")
    else:
        df_down = friten_trainEval.copy()
        print("📥 Downstream (no filtering for offensive)")

    df_up = friten_trainEval.copy()

    # Validate indices
    train_indices_up = [idx for idx in train_indices if idx in df_up.index]
    eval_indices_up = [idx for idx in eval_indices if idx in df_up.index]
    train_indices_down = [idx for idx in train_indices if idx in df_down.index]
    eval_indices_down = [idx for idx in eval_indices if idx in df_down.index]

    print("\n🔎 Index validation:")
    print(f"Upstream: train={len(train_indices_up)}, val={len(eval_indices_up)}")
    print(f"Downstream: train={len(train_indices_down)}, val={len(eval_indices_down)}")

    # Upstream splits
    up_train = df_up.loc[train_indices_up, ['TEXT', 'HATE']]
    up_val = df_up.loc[eval_indices_up, ['TEXT', 'HATE']]
    up_train['HATE'] = up_train['HATE'].apply(fix_off_mapping)
    up_val['HATE'] = up_val['HATE'].apply(fix_off_mapping)

    # Downstream splits
    down_train = df_down.loc[train_indices_down, ['TEXT', 'ROLE']]
    down_val = df_down.loc[eval_indices_down, ['TEXT', 'ROLE']]
    down_train['ROLE'] = down_train['ROLE'].apply(fix_role_mapping)
    down_val['ROLE'] = down_val['ROLE'].apply(fix_role_mapping)

    down_train = down_train[down_train['ROLE'] != 'none']
    down_val = down_val[down_val['ROLE'] != 'none']

    print("\n📘 Role label counts (downstream train):")
    print(down_train.ROLE.value_counts())

    # Test set
    friten_test = fix_label_conflicts(friten_test)
    friten_test['HATE'] = friten_test['HATE'].apply(fix_off_mapping)
    friten_test['ROLE'] = friten_test['ROLE'].apply(fix_role_mapping)
    friten_test = friten_test[friten_test['ROLE'] != 'none']
    print("\n🧪 Test set shape after filtering:", friten_test.shape)

    # Gold set (based on French)
    gold = pd.read_csv(f"{base_dir}/llm_fren/seed42_ty_proj1_context5.csv")
    gold = gold[['TEXT', 'ROLE', 'HATE']]
    gold = fix_label_conflicts(gold)
    gold['HATE'] = gold['HATE'].apply(fix_off_mapping)
    gold['ROLE'] = gold['ROLE'].apply(fix_role_mapping)
    gold = gold[gold['ROLE'] != 'none']

    print("\n🏅 Gold data shape:", gold.shape)
    print("✅ get_friten_data completed.\n")

    return up_train, up_val, down_train, down_val, friten_test, gold
