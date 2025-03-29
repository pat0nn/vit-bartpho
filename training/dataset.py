"""Custom dataset implementation for image captioning."""

import torch
from PIL import Image

class ImageCaptioningDataset(torch.utils.data.Dataset):
    def __init__(self, ds, ds_type, tokenizer, feature_extractor, max_target_length):
        """
        Initialize the dataset.
        
        Args:
            ds: The dataset dictionary
            ds_type: 'train' or 'test'
            tokenizer: The tokenizer for processing captions
            feature_extractor: The feature extractor for processing images
            max_target_length: Maximum length for captions
        """
        self.ds = ds
        self.max_target_length = max_target_length
        self.ds_type = ds_type
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor

    def __getitem__(self, idx):
        image_path = self.ds[self.ds_type]['image_path'][idx]
        caption = self.ds[self.ds_type]['caption'][idx]
        model_inputs = dict()
        model_inputs['labels'] = self.tokenization_fn(caption, self.max_target_length)
        model_inputs['pixel_values'] = self.feature_extraction_fn(image_path)
        return model_inputs

    def __len__(self):
        return len(self.ds[self.ds_type])

    # text preprocessing step
    def tokenization_fn(self, captions, max_target_length):
        """Run tokenization on captions using BARTpho tokenizer."""
        labels = self.tokenizer(
            captions,
            padding="max_length",
            max_length=max_target_length,
            truncation=True,
            return_tensors="pt"
        ).input_ids
        
        # Fix: Convert to a simple tensor without extra dimensions
        return labels.squeeze(0)  # Remove batch dimension if it exists

    # image preprocessing step
    def feature_extraction_fn(self, image_path):
        """
        Run feature extraction on images.
        If `check_image` is `True`, the examples that fails during `Image.open()` will be caught and discarded.
        Otherwise, an exception will be thrown.
        """
        image = Image.open(image_path).convert('RGB')
        encoder_inputs = self.feature_extractor(images=image, return_tensors="np")
        return encoder_inputs.pixel_values[0]
