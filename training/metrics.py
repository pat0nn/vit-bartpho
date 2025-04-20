import numpy as np
import json
import os
from underthesea import sent_tokenize
import evaluate
import metrics
import torch
from tqdm import tqdm
from PIL import Image
import sys
import importlib
import subprocess
from pathlib import Path

# Try to import pycocotools
try:
    from pycocotools.coco import COCO
    from pycocoevalcap.eval import COCOEvalCap
    HAS_PYCOCOTOOLS = True
except ImportError:
    HAS_PYCOCOTOOLS = False
    print("Warning: pycocotools not found. Some metrics might not be available.")

def load_groundtruth_ids(groundtruth_file):
    """Load image IDs from groundtruth file."""
    if not groundtruth_file or not os.path.exists(groundtruth_file):
        return None
    
    try:
        with open(groundtruth_file, 'r', encoding='utf-8') as f:
            gt_data = json.load(f)
        return list(gt_data.keys())
    except Exception as e:
        print(f"Error loading groundtruth IDs: {e}")
        return None

def save_predictions_to_json(predictions, image_ids, output_file, groundtruth_file=None, is_coco_format=False):
    """
    Save predictions to a JSON file with image IDs as keys.
    If image_ids is not provided, tries to use groundtruth_file to align IDs.
    
    Args:
        predictions: List of captions or list of dicts with image_id and caption
        image_ids: List of image IDs (optional if using dicts in predictions)
        output_file: Path to save the file
        groundtruth_file: Path to ground truth file (optional)
        is_coco_format: Whether to save in COCO format for evaluation
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # If predictions are already in dict form with image_id
    if predictions and isinstance(predictions[0], dict) and 'image_id' in predictions[0]:
        if is_coco_format:
            # Convert to COCO format
            coco_predictions = []
            for item in predictions:
                coco_predictions.append({
                    'image_id': item['image_id'],
                    'caption': item['caption']
                })
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(coco_predictions, f, ensure_ascii=False, indent=2)
        else:
            # Convert to standard format (image_id as key)
            predictions_dict = {item['image_id']: item['caption'] for item in predictions}
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(predictions_dict, f, ensure_ascii=False, indent=2)
    else:
        # Traditional format with separate image_ids and predictions
        predictions_dict = {}
        
        # If image_ids not provided but groundtruth_file is available, use its IDs
        if (image_ids is None or len(image_ids) == 0) and groundtruth_file:
            image_ids = load_groundtruth_ids(groundtruth_file)
        
        # Map predictions to image IDs
        for i, pred in enumerate(predictions):
            predictions_dict[image_ids[i]] = pred
        
        # Save to file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(predictions_dict, f, ensure_ascii=False, indent=2)
    
    print(f"Saved predictions to {output_file}")
    return output_file

# Create modified COCOEvalCap class to skip SPICE
class ModifiedCOCOEvalCap(COCOEvalCap):
    """
    Modified COCOEvalCap class that skips SPICE metric due to common Java errors.
    """
    def __init__(self, coco, cocoRes):
        # Initialize key attributes
        self.evalImgs = []
        self.eval = {}
        self.imgToEval = {}
        self.params = {'image_id': coco.getImgIds()}
        self.coco = coco
        self.cocoRes = cocoRes
        self.imgIds = self.params['image_id']
    
    def setEval(self, score, method):
        """
        Set evaluation score for a metric.
        """
        self.eval[method] = score
    
    def setImgToEvalImgs(self, scores, imgIds, method):
        """
        Set evaluation scores for individual images.
        """
        for imgId, score in zip(imgIds, scores):
            if imgId not in self.imgToEval:
                self.imgToEval[imgId] = {}
                self.imgToEval[imgId]["image_id"] = imgId
            self.imgToEval[imgId][method] = score
    
    def tokenize(self):
        """
        Tokenize ground truth and prediction captions.
        """
        imgIds = self.params['image_id']
        gts = {}
        res = {}
        
        print('tokenization...')
        
        from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
        tokenizer = PTBTokenizer()
        
        for imgId in imgIds:
            try:
                gts[imgId] = [{'caption': self.coco.anns[ann_id]['caption']} 
                           for ann_id in self.coco.getAnnIds(imgIds=imgId)]
            except KeyError:
                # Try with alternative key for caption in case we're using COCO annotation with segmentation
                try:
                    gts[imgId] = [{'caption': self.coco.anns[ann_id]['segment_caption'] if 'segment_caption' in self.coco.anns[ann_id] else self.coco.anns[ann_id]['caption']} 
                               for ann_id in self.coco.getAnnIds(imgIds=imgId)]
                except KeyError as e:
                    print(f"Error finding caption for image {imgId}: {e}")
                    # Skip this image if cannot find captions
                    continue
                
            res[imgId] = [{'caption': self.cocoRes.anns[ann_id]['caption']} 
                       for ann_id in self.cocoRes.getAnnIds(imgIds=imgId)]
        
        # Tokenize
        self.gts = tokenizer.tokenize(gts)
        self.res = tokenizer.tokenize(res)
        print('Done')
        
    def evaluate(self):
        # Ensure tokenization has been performed
        if not hasattr(self, 'gts') or not hasattr(self, 'res'):
            # Call tokenization method if not already done
            self.tokenize()
            
        # Get the image IDs to evaluate
        imgIds = self.params['image_id']
        
        # =================================================
        # Set up scorers
        # =================================================
        print('setting up scorers...')
        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.meteor.meteor import Meteor
        from pycocoevalcap.rouge.rouge import Rouge
        from pycocoevalcap.cider.cider import Cider
        
        # =================================================
        # Compute scores
        # =================================================
        self.eval = {}
        self.imgToEval = {}
        
        # Compute BLEU scores (special handling because it returns multiple values)
        print('computing Bleu score...')
        scorer = Bleu(4)
        score, scores = scorer.compute_score(self.gts, self.res)
        
        # Handle BLEU scores separately because they come as a list
        if isinstance(score, list):
            for i, sc in enumerate(score):
                method = f"Bleu_{i+1}"
                self.setEval(sc, method)
                # For image scores, we need to handle differently since scores might not be a list of lists
                if isinstance(scores[0], list):
                    # Extract per-image scores for this specific BLEU-N
                    img_scores = [s[i] if i < len(s) else 0.0 for s in scores]
                    self.setImgToEvalImgs(img_scores, self.imgIds, method)
                else:
                    # If scores is just a flat list, use it directly
                    self.setImgToEvalImgs(scores, self.imgIds, method)
                print(f"{method}: {sc:.3f}")
        else:
            # If for some reason BLEU returned a single value
            self.setEval(score, "Bleu")
            self.setImgToEvalImgs(scores, self.imgIds, "Bleu")
            print(f"Bleu: {score:.3f}")
        
        # Compute other metrics
        other_scorers = [
            (Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr")
        ]
        
        for scorer, method in other_scorers:
            try:
                print(f'computing {method} score...')
                score, scores = scorer.compute_score(self.gts, self.res)
                
                # Handle case where score might be a list
                if isinstance(score, list):
                    for i, sc in enumerate(score):
                        sub_method = f"{method}_{i+1}"
                        self.setEval(sc, sub_method)
                        print(f"{sub_method}: {sc:.3f}")
                else:
                    self.setEval(score, method)
                    self.setImgToEvalImgs(scores, self.imgIds, method)
                    print(f"{method}: {score:.3f}")
            except Exception as e:
                print(f"Error computing {method}: {e}")
                # Set to None if computation fails
                self.setEval(None, method)
        
        # Also set up SPICE with None so other code doesn't break
        self.setEval(None, "SPICE")
        self.eval.update({"SPICE": None})

def evaluate_from_files(groundtruth_file, prediction_file, use_coco_eval=False, skip_spice=True, use_subset=None, subset_size=None, image_ids=None):
    """
    Đánh giá các tham số dựa trên hai file đầu vào.
    
    Args:
        groundtruth_file (str): Đường dẫn đến file chứa groundtruth caption
        prediction_file (str): Đường dẫn đến file chứa kết quả caption
        use_coco_eval (bool): Whether to use COCO evaluation metrics
        skip_spice (bool): Whether to skip SPICE metric due to common Java errors
        use_subset (bool): Whether to use a subset of the ground truth data
        subset_size (int): Size of the subset to use
        image_ids (list): List of specific image IDs to evaluate (overrides subset)
        
    Returns:
        dict: Kết quả đánh giá các tham số
    """
    # Handle subset configuration from system config if not explicitly provided
    if use_subset is None and 'config' in sys.modules:
        import config
        use_subset = getattr(config, 'USE_SUBSET', False)
        subset_size = getattr(config, 'TEST_SUBSET_SIZE', 7)
    
    # Only try COCO evaluation if requested and necessary imports are available
    if use_coco_eval:
        try:
            # Try to import required libraries
            from pycocotools.coco import COCO
            from pycocoevalcap.bleu.bleu import Bleu
            from pycocoevalcap.meteor.meteor import Meteor
            from pycocoevalcap.rouge.rouge import Rouge
            from pycocoevalcap.cider.cider import Cider
            
            # Load ground truth file
            print(f"Loading ground truth COCO data from: {groundtruth_file}")
            coco = COCO(groundtruth_file)
            
            # Filter image IDs if using subset
            if image_ids:
                # Use specific image IDs if provided
                filtered_img_ids = [img_id for img_id in image_ids if img_id in coco.imgs]
                if len(filtered_img_ids) == 0:
                    print("WARNING: None of the provided image IDs are in the ground truth data!")
                    filtered_img_ids = list(coco.imgs.keys())[:subset_size]
            elif use_subset and subset_size:
                # Use a subset of image IDs
                filtered_img_ids = list(coco.imgs.keys())[:subset_size]
                print(f"Using subset of {len(filtered_img_ids)} images for evaluation")
            else:
                # Use all image IDs
                filtered_img_ids = list(coco.imgs.keys())
                print(f"Using all {len(filtered_img_ids)} images for evaluation")
            
            # Load results for either all images or subset
            print(f"Loading prediction results from: {prediction_file}")
            coco_result = coco.loadRes(prediction_file)
            
            try:
                # Create coco_eval object
                if skip_spice:
                    # Use modified evaluator without SPICE
                    print("Using modified COCO evaluator without SPICE")
                    coco_eval = ModifiedCOCOEvalCap(coco, coco_result)
                else:
                    # Try to use standard evaluator with SPICE
                    print("Using standard COCO evaluator with SPICE")
                    from pycocoevalcap.eval import COCOEvalCap
                    coco_eval = COCOEvalCap(coco, coco_result)
                
                # Evaluate only on the selected images (filtered subset)
                # Get the intersection of filtered_img_ids and what's available in the results
                available_img_ids = set(coco_result.getImgIds())
                eval_img_ids = [img_id for img_id in filtered_img_ids if img_id in available_img_ids]
                
                if len(eval_img_ids) == 0:
                    print("WARNING: No common image IDs between predictions and selected ground truth!")
                    # Fall back to standard evaluation
                    return evaluate_from_files(groundtruth_file, prediction_file, use_coco_eval=False, 
                                            skip_spice=skip_spice, use_subset=use_subset, 
                                            subset_size=subset_size, image_ids=image_ids)
                
                print(f"Evaluating on {len(eval_img_ids)} images")
                coco_eval.params['image_id'] = eval_img_ids
                
                try:
                    # Evaluate results
                    coco_eval.evaluate()
                    
                    # Return scores
                    result = {}
                    for metric, score in coco_eval.eval.items():
                        if score is not None:  # Skip None values (like SPICE when skipped)
                            result[metric] = score
                    
                    return result
                except AttributeError as e:
                    # Handle attribute errors that might happen with the COCO evaluator classes
                    print(f"Error with COCO evaluator: {e}")
                    print("Falling back to standard evaluation")
                    return evaluate_from_files(groundtruth_file, prediction_file, use_coco_eval=False, 
                                            skip_spice=skip_spice, use_subset=use_subset, 
                                            subset_size=subset_size, image_ids=image_ids)
                except subprocess.CalledProcessError as e:
                    # Handle SPICE-specific errors from Java
                    if "SPICE" in str(e):
                        print(f"SPICE evaluation failed: {e}")
                        # If SPICE failed but we didn't explicitly skip it, retry with skip_spice=True
                        if not skip_spice:
                            print("Retrying without SPICE metric...")
                            return evaluate_from_files(groundtruth_file, prediction_file, use_coco_eval=True, skip_spice=True,
                                                    use_subset=use_subset, subset_size=subset_size, image_ids=image_ids)
                        else:
                            # We already tried to skip it, so fall back to standard evaluation
                            print("Falling back to standard evaluation...")
                    else:
                        # Some other error with COCO evaluation
                        print(f"Error during COCO evaluation: {e}")
                        print("Falling back to standard evaluation...")
            except Exception as e:
                print(f"Error with COCO evaluation setup: {e}")
                print("Falling back to standard evaluation")
                return evaluate_from_files(groundtruth_file, prediction_file, use_coco_eval=False, 
                                        skip_spice=skip_spice, use_subset=use_subset, 
                                        subset_size=subset_size, image_ids=image_ids)
        except (ImportError, ModuleNotFoundError) as e:
            print(f"Could not import required COCO modules: {e}")
            print("Falling back to standard evaluation")
            use_coco_eval = False
        except Exception as e:
            print(f"Error using COCO evaluation: {e}")
            print("Falling back to standard evaluation...")
            use_coco_eval = False
    
    # Standard evaluation (fallback if COCO fails or is not available)
    print("Using standard evaluation method")
    with open(groundtruth_file, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)
    
    with open(prediction_file, 'r', encoding='utf-8') as f:
        pred_data = json.load(f)
    
    # Prepare data for metric computation
    gt_captions = {}
    pred_captions = {}
    
    # Convert groundtruth data
    for image_id, captions in gt_data.items():
        gt_captions[image_id] = captions if isinstance(captions, list) else [captions]
    
    # Convert prediction data for different formats
    if isinstance(pred_data, list):
        # If it's COCO format (list of dicts with image_id and caption)
        for item in pred_data:
            if isinstance(item, dict) and 'image_id' in item and 'caption' in item:
                image_id = item['image_id']
                caption = item['caption']
                pred_captions[image_id] = [caption]
    else:
        # Standard format with image_id as keys
        for image_id, caption in pred_data.items():
            if isinstance(caption, str):
                pred_captions[image_id] = [caption]
            elif isinstance(caption, list):
                pred_captions[image_id] = caption
            else:
                print(f"Warning: Không nhận dạng được định dạng caption cho image_id {image_id}")
    
    # Filter image IDs if using subset
    if image_ids:
        # Use specific image IDs if provided
        gt_ids = set(gt_captions.keys()).intersection(image_ids)
    elif use_subset and subset_size:
        # Use a subset of ground truth IDs - take the first subset_size entries
        gt_ids = set(list(gt_captions.keys())[:subset_size])
        print(f"Using subset of {len(gt_ids)} ground truth images for evaluation")
    else:
        # Use all ground truth IDs
        gt_ids = set(gt_captions.keys())
    
    # Check common image IDs with predictions
    pred_ids = set(pred_captions.keys())
    common_ids = gt_ids.intersection(pred_ids)
    
    if len(common_ids) == 0:
        print("WARNING: No common image_id found between the two files!")
        return {}
    
    print(f"Number of images in filtered groundtruth: {len(gt_ids)}")
    print(f"Number of images in prediction: {len(pred_ids)}")
    print(f"Number of common images for evaluation: {len(common_ids)}")
    
    if len(common_ids) < len(gt_ids) or len(common_ids) < len(pred_ids):
        print(f"Warning: {len(gt_ids) - len(common_ids)} images from groundtruth are not in prediction")
        print(f"Warning: {len(pred_ids) - len(common_ids)} images from prediction are not in groundtruth")
    
    # Filter to keep only common image IDs
    filtered_gt = {id: gt_captions[id] for id in common_ids}
    filtered_pred = {id: pred_captions[id] for id in common_ids}
    
    # Compute metrics
    print("Computing metrics...")
    try:
        # Try to compute metrics with SPICE
        scores = metrics.compute_scores(filtered_gt, filtered_pred)[0]
    except Exception as e:
        if "SPICE" in str(e) or "subprocess" in str(e):
            print(f"Error with SPICE calculation: {e}")
            print("Trying again without SPICE...")
            # Try to get metrics with disable_spice=True if it's an option in your compute_scores
            try:
                scores = metrics.compute_scores(filtered_gt, filtered_pred, disable_spice=True)[0]
            except:
                # If the function doesn't have a disable_spice param, need to handle it differently
                # Fallback to just returning other metrics
                print("Falling back to simpler metrics calculation...")
                # Create a simpler scores dict without SPICE
                scores = {
                    "BLEU": [0.0, 0.0, 0.0, 0.0],  # Placeholders
                    "METEOR": 0.0,
                    "ROUGE_L": 0.0,
                    "CIDEr": 0.0
                }
                # Try to compute each metric individually
                try:
                    from pycocoevalcap.bleu.bleu import Bleu
                    bleu_scorer = Bleu(4)
                    bleu_score, _ = bleu_scorer.compute_score(filtered_gt, filtered_pred)
                    scores["BLEU"] = bleu_score
                except Exception as e:
                    print(f"Failed to compute BLEU: {e}")
                
                try:
                    from pycocoevalcap.meteor.meteor import Meteor
                    meteor_scorer = Meteor()
                    meteor_score, _ = meteor_scorer.compute_score(filtered_gt, filtered_pred)
                    scores["METEOR"] = meteor_score
                except Exception as e:
                    print(f"Failed to compute METEOR: {e}")
                
                try:
                    from pycocoevalcap.rouge.rouge import Rouge
                    rouge_scorer = Rouge()
                    rouge_score, _ = rouge_scorer.compute_score(filtered_gt, filtered_pred)
                    scores["ROUGE_L"] = rouge_score
                except Exception as e:
                    print(f"Failed to compute ROUGE_L: {e}")
                
                try:
                    from pycocoevalcap.cider.cider import Cider
                    cider_scorer = Cider()
                    cider_score, _ = cider_scorer.compute_score(filtered_gt, filtered_pred)
                    scores["CIDEr"] = cider_score
                except Exception as e:
                    print(f"Failed to compute CIDEr: {e}")
        else:
            # Some other error
            print(f"Unexpected error in metrics calculation: {e}")
            return {}
    
    result = {}
    for metric, score in scores.items():
        if metric == "BLEU":
            result["BLEU-1"] = score[0]
            result["BLEU-2"] = score[1]
            result["BLEU-3"] = score[2]
            result["BLEU-4"] = score[3]
        elif metric != "SPICE" or (metric == "SPICE" and score is not None):
            # Only include SPICE if it's not None
            result[metric] = score
    
    return result

def process_image(image_path, feature_extractor, device):
    """Process an image for the model."""
    try:
        image = Image.open(image_path).convert('RGB')
        pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values.to(device)
        return pixel_values
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def generate_caption(model, feature_extractor, tokenizer, image_path, device, num_beams=3, max_length=24):
    """Generate a caption for an image."""
    pixel_values = process_image(image_path, feature_extractor, device)
    if pixel_values is None:
        return None
    
    with torch.no_grad():
        generated_ids = model.generate(
            pixel_values, 
            num_beams=num_beams, 
            do_sample=False,  
            max_length=max_length
        )
    
    generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text.replace("_", " ")

def compute_metrics_from_model(
    model, 
    tokenizer, 
    feature_extractor, 
    eval_dataset, 
    groundtruth_file, 
    output_dir, 
    epoch=None, 
    num_beams=3, 
    max_length=24, 
    use_coco_eval=False,
    skip_spice=True,
    use_subset=None,
    subset_size=None,
    return_raw_results=False
):
    """
    Generate captions directly from the model and compute metrics.
    
    Args:
        model: The model to evaluate
        tokenizer: Tokenizer for decoding text
        feature_extractor: Feature extractor for processing images
        eval_dataset: Dataset to evaluate on
        groundtruth_file: Path to ground truth captions
        output_dir: Directory to save predictions
        epoch: Current epoch (for naming output files)
        num_beams: Number of beams for beam search
        max_length: Maximum caption length
        use_coco_eval: Whether to use COCO evaluation
        skip_spice: Whether to skip SPICE metric
        use_subset: Whether to use a subset of the ground truth data
        subset_size: Size of the subset to use
        return_raw_results: Whether to return the raw generation results
        
    Returns:
        dict: Evaluation metrics
    """
    # Handle subset configuration from system config if not explicitly provided
    if use_subset is None and 'config' in sys.modules:
        import config
        use_subset = getattr(config, 'USE_SUBSET', False)
        subset_size = getattr(config, 'TEST_SUBSET_SIZE', 7)
    
    # Ensure model is in evaluation mode
    model.eval()
    
    # Get the compute device
    device = next(model.parameters()).device
    
    # Generate captions for all images in the eval dataset
    results = []
    image_ids = []  # Keep track of image IDs for subsetting
    
    for item in tqdm(eval_dataset, desc="Generating captions for evaluation"):
        image_id = item['image_id']
        image_path = item['image_path']
        image_ids.append(image_id)
        
        try:
            caption = generate_caption(
                model, 
                feature_extractor, 
                tokenizer, 
                image_path, 
                device, 
                num_beams=num_beams, 
                max_length=max_length
            )
            
            if caption is not None:
                results.append({
                    "image_id": image_id,
                    "caption": caption
                })
        except Exception as e:
            print(f"Error generating caption for image_id {image_id}: {str(e)}")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create prediction filename
    if epoch is not None:
        prediction_file = os.path.join(output_dir, f"predictions_epoch_{epoch}_beam_{num_beams}.json")
    else:
        prediction_file = os.path.join(output_dir, f"predictions_beam_{num_beams}.json")
    
    # Save predictions
    save_predictions_to_json(
        results, 
        None,  # image_ids are contained in results
        prediction_file,
        is_coco_format=use_coco_eval
    )
    
    # Compute metrics if groundtruth file is provided
    if groundtruth_file and os.path.exists(groundtruth_file):
        metrics_result = evaluate_from_files(
            groundtruth_file, 
            prediction_file, 
            use_coco_eval=use_coco_eval,
            skip_spice=skip_spice,
            use_subset=use_subset,
            subset_size=subset_size,
            image_ids=image_ids if use_subset else None
        )
    else:
        metrics_result = {}
        print("No groundtruth file provided for evaluation or file does not exist")
    
    # Return both metrics and raw results if requested
    if return_raw_results:
        return metrics_result, results
    
    return metrics_result

def compute_metrics(eval_preds, tokenizer, model=None, feature_extractor=None, eval_dataset=None, groundtruth_file=None, output_dir=None, epoch=None, num_beams=3, max_length=24, skip_spice=True, use_subset=None, subset_size=None):
    """
    Compute evaluation metrics for the generated captions.
    This function can work in two modes:
    1. Using predictions from the Trainer (original behavior)
    2. Generating captions directly from the model (new behavior when model is provided)
    """
    # If model and required parameters are provided, use direct generation
    if model is not None and feature_extractor is not None and eval_dataset is not None:
        return compute_metrics_from_model(
            model=model,
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            eval_dataset=eval_dataset,
            groundtruth_file=groundtruth_file,
            output_dir=output_dir if output_dir else "./eval_outputs",
            epoch=epoch,
            num_beams=num_beams,
            max_length=max_length,
            skip_spice=skip_spice,
            use_subset=use_subset,
            subset_size=subset_size
        )
    
    # Handle subset configuration from system config if not explicitly provided
    if use_subset is None and 'config' in sys.modules:
        import config
        use_subset = getattr(config, 'USE_SUBSET', False)
        subset_size = getattr(config, 'TEST_SUBSET_SIZE', 7)
    
    # Original behavior using Trainer outputs
    preds, labels = eval_preds
    
    if isinstance(preds, tuple):
        preds = preds[0]
    
    # Decode predictions
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    
    # Save predictions to a JSON file
    if output_dir is None:
        output_dir = "./eval_outputs"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Include epoch number in filename if available
    if epoch is not None:
        prediction_file = os.path.join(output_dir, f"predictions_epoch_{epoch}.json")
    else:
        prediction_file = os.path.join(output_dir, "predictions.json")
    
    image_ids = None
    save_predictions_to_json(decoded_preds, image_ids, prediction_file, groundtruth_file)
    
    # If groundtruth file is provided, evaluate metrics
    if groundtruth_file and os.path.exists(groundtruth_file):
        result = evaluate_from_files(
            groundtruth_file, 
            prediction_file, 
            skip_spice=skip_spice,
            use_subset=use_subset,
            subset_size=subset_size
        )
    else:
        result = {}
        print("No groundtruth file provided for evaluation")
    
    return result

# Add a test function to verify our implementation works
def test_evaluation(groundtruth_file, prediction_file, use_coco_eval=False, skip_spice=True):
    """
    Simple test function to verify that our evaluation works.
    
    Args:
        groundtruth_file: Path to ground truth file
        prediction_file: Path to predictions file
        use_coco_eval: Whether to use COCO evaluation
        skip_spice: Whether to skip SPICE metric
    """
    print("=" * 50)
    print(f"Testing evaluation with:")
    print(f"- Ground truth file: {groundtruth_file}")
    print(f"- Prediction file: {prediction_file}")
    print(f"- Using COCO eval: {use_coco_eval}")
    print(f"- Skipping SPICE: {skip_spice}")
    print("=" * 50)
    
    result = evaluate_from_files(
        groundtruth_file, 
        prediction_file, 
        use_coco_eval=use_coco_eval,
        skip_spice=skip_spice,
        use_subset=True, 
        subset_size=3
    )
    
    print("=" * 50)
    print("Evaluation results:")
    for metric, score in result.items():
        print(f"{metric}: {score:.4f}")
    print("=" * 50)
    
    return result

# Run the test if executed directly
if __name__ == "__main__":
    import sys
    
    # Default test files
    groundtruth_file = "../data/groundtruth_captions_val2017.json"
    prediction_file = "../output/eval/predictions_epoch_1_beam_3.json"
    
    # Allow overriding test files from command line
    if len(sys.argv) > 2:
        groundtruth_file = sys.argv[1]
        prediction_file = sys.argv[2]
    
    # Test with standard evaluation first
    test_evaluation(groundtruth_file, prediction_file, use_coco_eval=False)
    
    # Then test with COCO evaluation if pycocotools is available
    if HAS_PYCOCOTOOLS:
        test_evaluation(groundtruth_file, prediction_file, use_coco_eval=True, skip_spice=True)
