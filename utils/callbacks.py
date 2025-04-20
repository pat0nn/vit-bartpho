"""Custom callbacks for model training and evaluation."""

import os
import shutil
import wandb
from transformers import TrainerCallback

class WandbModelCheckpointCallback(TrainerCallback):
    """
    Custom callback to upload model checkpoints to Weights & Biases.
    Optimized for Kaggle environment with limited storage:
    - Removes optimizer.pt before uploading (unnecessary for inference)
    - Deletes local checkpoints after uploading to save space
    """
    
    def __init__(self, save_best_only=False, metric_name="eval_loss", artifact_type="model", 
                 remove_optimizer=True, save_optimizer_separately=False):
        """
        Initialize the callback.
        
        Args:
            save_best_only: Whether to save only the best checkpoint based on the metric
            metric_name: Name of the metric to monitor if save_best_only is True
            artifact_type: Type of artifact to create
            remove_optimizer: Whether to remove optimizer.pt before uploading to save space
            save_optimizer_separately: Whether to save optimizer state in a separate artifact
        """
        self.save_best_only = save_best_only
        self.metric_name = metric_name
        self.artifact_type = artifact_type
        self.remove_optimizer = remove_optimizer
        self.save_optimizer_separately = save_optimizer_separately
        self.best_metric = float('inf')  # For minimizing metrics like loss
        self.best_step = None
    
    def on_save(self, args, state, control, **kwargs):
        """
        Called when a checkpoint is saved.
        
        Args:
            args: TrainingArguments
            state: TrainerState
            control: TrainerControl
            **kwargs: Additional arguments
            
        Returns:
            control: TrainerControl
        """
        # Check if wandb is being used
        if not wandb.run:
            print("WandbModelCheckpointCallback: wandb run not detected, skipping artifact logging.")
            return control
        
        # Get the checkpoint path
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        
        # Ensure the checkpoint directory exists
        if not os.path.exists(checkpoint_path):
            print(f"Warning: Checkpoint path {checkpoint_path} does not exist, skipping artifact logging.")
            return control
        
        # Check if we should save this checkpoint based on metrics
        # First check if metrics attribute exists and is not empty
        has_metrics = hasattr(state, 'metrics') and state.metrics
        
        if self.save_best_only and has_metrics:
            current_metric = state.metrics.get(self.metric_name)
            
            if current_metric is not None:
                # For metrics where lower is better (like loss)
                is_better = current_metric < self.best_metric
                
                # Some metrics are higher-is-better (accuracy, BLEU, etc.)
                if self.metric_name.startswith(("eval_accuracy", "eval_BLEU", "eval_METEOR", 
                                              "eval_ROUGE", "eval_CIDEr", "eval_model_")):
                    is_better = current_metric > self.best_metric
                
                if is_better:
                    self.best_metric = current_metric
                    self.best_step = state.global_step
                    print(f"New best model at step {state.global_step} with {self.metric_name}={current_metric:.4f}")
                else:
                    # Skip this checkpoint if it's not the best, and remove it to save space
                    print(f"Skipping checkpoint at step {state.global_step} (not better than best {self.metric_name}={self.best_metric:.4f})")
                    print(f"Removing checkpoint directory: {checkpoint_path}")
                    shutil.rmtree(checkpoint_path)
                    return control
        elif self.save_best_only and not has_metrics:
            # If we're supposed to save only the best model but don't have metrics yet,
            # log the info and continue saving this checkpoint
            print(f"Note: save_best_only=True but no metrics available yet. Saving checkpoint at step {state.global_step}.")
        
        # Log the epoch if available
        epoch_info = f"epoch_{state.epoch:.1f}_" if hasattr(state, 'epoch') and state.epoch is not None else ""
        
        # Create artifact name with informative naming
        artifact_name = f"model-{epoch_info}step-{state.global_step}"
        
        try:
            print(f"Processing checkpoint for wandb: {checkpoint_path}")
            
            # Check if optimizer state should be saved separately or removed
            optimizer_path = os.path.join(checkpoint_path, "optimizer.pt")
            optimizer_saved = False
            
            if os.path.exists(optimizer_path):
                if self.save_optimizer_separately:
                    # Save optimizer state as a separate artifact
                    optimizer_artifact = wandb.Artifact(
                        name=f"optimizer-{epoch_info}step-{state.global_step}",
                        type="optimizer",
                        description=f"Optimizer state at step {state.global_step}"
                    )
                    optimizer_artifact.add_file(optimizer_path)
                    wandb.log_artifact(optimizer_artifact)
                    optimizer_saved = True
                    print(f"Saved optimizer state as separate artifact")
                
                if self.remove_optimizer:
                    # Remove optimizer.pt to save space
                    print(f"Removing optimizer.pt file to save space ({os.path.getsize(optimizer_path) / (1024*1024):.2f} MB)")
                    os.remove(optimizer_path)
            
            # Create and log wandb artifact
            print(f"Creating and logging wandb artifact: {artifact_name}")
            
            # Create an artifact
            artifact = wandb.Artifact(
                name=artifact_name,
                type=self.artifact_type,
                description=f"Model checkpoint at step {state.global_step} (epoch {state.epoch:.1f if hasattr(state, 'epoch') and state.epoch is not None else 'unknown'})"
            )
            
            # Add metadata about the training state
            if has_metrics:
                metrics_dict = state.metrics.copy()
                for key, value in metrics_dict.items():
                    if isinstance(value, (int, float)):
                        artifact.metadata[key] = value
            
            # Add optimizer info to metadata
            artifact.metadata["optimizer_included"] = not self.remove_optimizer
            artifact.metadata["optimizer_saved_separately"] = optimizer_saved
            
            # Add the model directory to the artifact
            artifact.add_dir(checkpoint_path, name="model")
            
            # Log the artifact to wandb
            wandb.log_artifact(artifact)
            wandb.run.log({"checkpoint_step": state.global_step})
            
            print(f"Successfully logged model checkpoint to wandb as artifact: {artifact_name}")
            
            # Always remove the local checkpoint to save space (for Kaggle environment)
            print(f"Removing local checkpoint directory to save space: {checkpoint_path}")
            shutil.rmtree(checkpoint_path)
        
        except Exception as e:
            print(f"Error logging model checkpoint to wandb: {e}")
            # Even if there's an error, try to remove the checkpoint to save space
            if os.path.exists(checkpoint_path):
                try:
                    shutil.rmtree(checkpoint_path)
                    print(f"Removed checkpoint directory after error: {checkpoint_path}")
                except Exception as cleanup_error:
                    print(f"Failed to clean up checkpoint directory: {cleanup_error}")
        
        return control
    
    def on_train_end(self, args, state, control, **kwargs):
        """
        Called at the end of training to save the final model.
        
        Args:
            args: TrainingArguments
            state: TrainerState
            control: TrainerControl
            **kwargs: Additional arguments
            
        Returns:
            control: TrainerControl
        """
        # Save the final model as an artifact
        try:
            if wandb.run:
                final_model_path = args.output_dir
                
                if os.path.exists(final_model_path):
                    # Check if optimizer.pt exists in the final model directory
                    optimizer_path = os.path.join(final_model_path, "optimizer.pt")
                    if os.path.exists(optimizer_path) and self.remove_optimizer:
                        print(f"Removing optimizer.pt from final model to save space")
                        os.remove(optimizer_path)
                    
                    artifact = wandb.Artifact(
                        name=f"model-final",
                        type=self.artifact_type,
                        description="Final model after training completed"
                    )
                    
                    # Add metadata about the training state
                    has_metrics = hasattr(state, 'metrics') and state.metrics
                    if has_metrics:
                        for key, value in state.metrics.items():
                            if isinstance(value, (int, float)):
                                artifact.metadata[key] = value
                    
                    # Add training step information
                    if hasattr(state, 'global_step'):
                        artifact.metadata["global_step"] = state.global_step
                    
                    # Add epoch information
                    if hasattr(state, 'epoch') and state.epoch is not None:
                        artifact.metadata["epoch"] = state.epoch
                    
                    artifact.add_dir(final_model_path, name="model")
                    wandb.log_artifact(artifact)
                    print("Successfully logged final model to wandb")
                else:
                    print(f"Warning: Final model path {final_model_path} does not exist")
        except Exception as e:
            print(f"Error logging final model to wandb: {e}")
        
        return control


class EpochTrackingCallback(TrainerCallback):
    """
    Callback to track epoch numbers and provide them to the evaluation function.
    """
    def __init__(self):
        self.current_epoch = 0
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        """Called at the beginning of each epoch"""
        # Update current epoch
        if state.epoch is not None:
            self.current_epoch = int(state.epoch)
        else:
            self.current_epoch += 1
        
        print(f"Starting epoch {self.current_epoch}")
    
    def on_evaluate(self, args, state, control, **kwargs):
        """Called before evaluation begins"""
        # Provide the current epoch number to the trainer
        trainer = kwargs.get("trainer", None)
        if trainer is not None:
            # Store current epoch in trainer's state
            trainer.current_epoch = self.current_epoch
            print(f"Evaluation at epoch {self.current_epoch}")
