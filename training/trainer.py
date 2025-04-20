"""Training functionality for the image captioning model."""

import wandb
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, default_data_collator
import os
import sys
sys.path.append('..')
import config 
from utils.callbacks import WandbModelCheckpointCallback, EpochTrackingCallback
from training.dataset import ImageCaptioningDataset
from training.metrics import compute_metrics, compute_metrics_from_model

def initialize_wandb(project_name=config.WANDB_PROJECT, experiment_name=config.WANDB_NAME, api_key=None):
    """
    Initialize Weights & Biases logging.
    
    Args:
        project_name: Name of the W&B project
        experiment_name: Name of this specific experiment
        api_key: W&B API key (optional, can be loaded from environment or kaggle)
    """
    if api_key:
        wandb.login(key=api_key)
    
    wandb.init(project=project_name, name=experiment_name)
    
    return wandb.run

def setup_training(model, feature_extractor, tokenizer, dataset, metrics_calculator=None, use_wandb=False):
    """
    Set up the training components.
    
    Args:
        model: The model to train
        feature_extractor: Feature extractor for processing images
        tokenizer: Tokenizer for processing text
        dataset: Dataset dictionary with train and test splits
        metrics_calculator: Optional custom metrics calculator
        use_wandb: Whether to use Weights & Biases for logging
        
    Returns:
        trainer: Configured Seq2SeqTrainer
    """
    # Create datasets
    train_dataset = ImageCaptioningDataset(
        dataset, 'train', tokenizer, feature_extractor, config.MAX_TARGET_LENGTH)
    eval_dataset = ImageCaptioningDataset(
        dataset, 'test', tokenizer, feature_extractor, config.MAX_TARGET_LENGTH)
    
    # Apply the subset to the test dataset as well if needed
    test_dataset = dataset['test']
    if config.USE_SUBSET:
        # Make sure we don't try to use more samples than are available
        subset_size = min(config.TEST_SUBSET_SIZE, len(test_dataset))
        print(f"Using subset of test dataset for direct model evaluation: {subset_size} samples")
        # Create a view with only the first subset_size elements
        test_dataset = dataset['test'].select(range(subset_size))
    else:
        print(f"Using all test dataset for direct model evaluation: {len(test_dataset)} samples")
    
    # Log subset usage information
    if config.USE_SUBSET:
        print(f"Using subset of data for quick testing:")
        print(f"  - Training samples: {len(train_dataset)} (from {len(dataset['train'])})")
        print(f"  - Evaluation samples: {len(eval_dataset)} (from {len(dataset['test'])})")
        
        # Log to wandb if enabled
        if use_wandb:
            wandb.config.update({
                "use_subset": config.USE_SUBSET,
                "train_subset_size": len(train_dataset),
                "test_subset_size": len(eval_dataset)
            })

    # Configure report_to based on wandb availability
    report_to = "wandb" if use_wandb else "none"
    
    # Set up training arguments
    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.EVAL_BATCH_SIZE,
        output_dir=config.OUTPUT_DIR,
        num_train_epochs=config.NUM_EPOCHS,
        report_to=report_to,
        fp16=config.USE_FP16,
        weight_decay=config.WEIGHT_DECAY,
        logging_dir=config.LOGS_DIR,
        logging_strategy="epoch",
        logging_steps=100,
    )
    
    # Setup compute_metrics with tokenizer and paths
    groundtruth_file = config.GROUNDTRUTH_FILE if hasattr(config, 'GROUNDTRUTH_FILE') else None
    coco_annotation_file = config.COCO_ANNOTATION_FILE if hasattr(config, 'COCO_ANNOTATION_FILE') else groundtruth_file
    eval_output_dir = os.path.join(config.OUTPUT_DIR, "eval")
    
    print(f"Using groundtruth file: {groundtruth_file}")
    print(f"Using COCO annotation file: {coco_annotation_file}")
    
    # Configure default beam search parameters for metrics
    num_beams = getattr(config, 'NUM_BEAMS', 3)
    max_length = getattr(config, 'MAX_LENGTH', 24)
    use_coco_eval = getattr(config, 'USE_COCO_EVAL', False)
    skip_spice = getattr(config, 'SKIP_SPICE', True)
    
    print(f"Metrics will be computed with beam search (num_beams={num_beams}, max_length={max_length})")
    print(f"Using COCO evaluation: {use_coco_eval}")
    print(f"Skipping SPICE metric: {skip_spice}")
    
    # Configure callbacks
    epoch_callback = EpochTrackingCallback()
    
    callbacks = [epoch_callback]
    if use_wandb:
        # Create WandbModelCheckpointCallback with optimized settings for Kaggle
        # Save the best model based on CIDEr score (or alternative metric)
        wandb_callback = WandbModelCheckpointCallback(
            save_best_only=True,
            metric_name="model_CIDEr",  # Use CIDEr as primary metric for image captioning
            remove_optimizer=True,      # Remove optimizer.pt to save space
            save_optimizer_separately=False,  # Don't save optimizer separately
            artifact_type="model"       # Artifact type
        )
        callbacks.append(wandb_callback)
        print("Configured wandb callback to save checkpoints, remove optimizer.pt, and track best model by CIDEr score")
    
    # Use CustomTrainer with direct model generation metrics
    trainer = CustomTrainer(
        model=model,
        tokenizer=tokenizer,  # Required for proper image preprocessing
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
        callbacks=callbacks,
        
        # Additional parameters for CustomTrainer
        test_dataset=test_dataset,
        feature_extractor=feature_extractor,
        groundtruth_file=coco_annotation_file if use_coco_eval else groundtruth_file,
        num_beams=num_beams,
        max_length=max_length,
        skip_spice=skip_spice
    )
    
    return trainer

class CustomTrainer(Seq2SeqTrainer):
    """Custom trainer class with support for direct model-based metrics."""
    
    def __init__(self, *args, **kwargs):
        # Extract custom parameters
        self.test_dataset = kwargs.pop('test_dataset', None)
        self.feature_extractor = kwargs.pop('feature_extractor', None)
        self.groundtruth_file = kwargs.pop('groundtruth_file', None)
        self.num_beams = kwargs.pop('num_beams', 3)
        self.max_length = kwargs.pop('max_length', 24)
        self.skip_spice = kwargs.pop('skip_spice', True)
        self.use_wandb = kwargs.get('args', None) and kwargs['args'].report_to == 'wandb'
        super().__init__(*args, **kwargs)
    
    def evaluate(self, *args, **kwargs):
        """Override evaluate to compute model-based metrics."""
        # Call original evaluate method from parent class
        metrics = super().evaluate(*args, **kwargs)
        
        # Add custom metrics if all required components are available
        if self.test_dataset and self.feature_extractor and self.groundtruth_file:
            epoch = self.state.epoch if hasattr(self.state, 'epoch') else None
            
            # Check if we should use COCO evaluation
            use_coco_eval = getattr(config, 'USE_COCO_EVAL', False)
            use_subset = getattr(config, 'USE_SUBSET', False)
            subset_size = getattr(config, 'TEST_SUBSET_SIZE', 7)
            
            # Compute metrics directly using the model
            model_metrics = compute_metrics_from_model(
                model=self.model,
                tokenizer=self.tokenizer,
                feature_extractor=self.feature_extractor,
                eval_dataset=self.test_dataset,
                groundtruth_file=self.groundtruth_file,
                output_dir=os.path.join(self.args.output_dir, "eval"),
                epoch=epoch,
                num_beams=self.num_beams,
                max_length=self.max_length,
                use_coco_eval=use_coco_eval,
                skip_spice=self.skip_spice,
                use_subset=use_subset,
                subset_size=subset_size
            )
            
            # Update metrics with model-based ones
            for k, v in model_metrics.items():
                metrics[f"model_{k}"] = v
            
            # Log specific metrics to console
            print(f"\n{'='*50}\nModel Metrics (Epoch {epoch}):")
            for key, val in model_metrics.items():
                print(f"{key}: {val:.4f}")
            print(f"{'='*50}\n")
            
            # Log to wandb if enabled
            if self.use_wandb:
                try:
                    # Log metrics separately to make them easier to track in wandb
                    wandb_metrics = {}
                    
                    # Define key metrics to track with more descriptive names
                    metrics_to_log = {
                        "BLEU-1": "Bleu-1 Score",
                        "BLEU-2": "Bleu-2 Score",
                        "BLEU-3": "Bleu-3 Score",
                        "BLEU-4": "Bleu-4 Score",
                        "METEOR": "METEOR Score",
                        "ROUGE_L": "ROUGE-L Score",
                        "CIDEr": "CIDEr Score"
                    }
                    
                    # Extract metrics and prepare for logging
                    for metric_key, display_name in metrics_to_log.items():
                        if metric_key in model_metrics:
                            wandb_metrics[f"caption_metrics/{metric_key}"] = model_metrics[metric_key]
                    
                    # Add a summary metric for easier tracking
                    if "CIDEr" in model_metrics and "BLEU-4" in model_metrics:
                        bleu4 = model_metrics.get("BLEU-4", 0)
                        cider = model_metrics.get("CIDEr", 0)
                        # Combined score often used in image captioning papers
                        wandb_metrics["caption_metrics/Combined_Score"] = (bleu4 + cider) / 2
                    
                    # Log current epoch
                    if epoch is not None:
                        wandb_metrics["epoch"] = epoch
                    
                    # Log to wandb
                    wandb.log(wandb_metrics)
                    print(f"Successfully logged metrics to Weights & Biases.")
                except Exception as e:
                    print(f"Error logging to wandb: {e}")
        
        return metrics

def train_model(trainer):
    """
    Train the model.
    
    Args:
        trainer: Configured Seq2SeqTrainer
    
    Returns:
        training_output: Output from the training process
    """
    # Start training
    training_output = trainer.train()
    
    return training_output
