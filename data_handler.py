# data_handler.py
# Handles data loading and preprocessing for multimodal RLHF

import os
import json
import torch
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
from torch.utils.data import Dataset, DataLoader
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

@dataclass
class MultimodalSample:
    """A single multimodal sample containing text, image, and 3D data."""
    text: str
    image_path: Optional[str] = None
    image: Optional[Image.Image] = None
    nerf_path: Optional[str] = None  # Path to NeRF data
    nerf_params: Optional[Dict] = None
    sensor_data: Optional[Dict] = None

@dataclass
class PreferenceData:
    """A pair of samples with human preference annotation."""
    chosen: MultimodalSample
    rejected: MultimodalSample
    consistency_score: Optional[float] = None  # Higher is better
    metadata: Optional[Dict] = None

class MultiModalPreferenceDataset(Dataset):
    """Dataset for RLHF training with multimodal preferences."""
    
    def __init__(self, data_path, tokenizer, transform=None):
        """
        Initialize the dataset.
        
        Args:
            data_path: Path to the JSON file containing preference data
            tokenizer: Tokenizer for text encoding
            transform: Optional transformation to apply to images
        """
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.transform = transform
        
        # Load the preference data
        with open(data_path, 'r') as f:
            self.preference_data = json.load(f)
            
        logger.info(f"Loaded {len(self.preference_data)} preference pairs")
        
    def __len__(self):
        return len(self.preference_data)
    
    def load_image(self, image_path):
        """Load an image from a path."""
        if not image_path:
            return None
        
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image
    
    def load_nerf_data(self, nerf_path):
        """Load NeRF data from a path."""
        if not nerf_path:
            return None, None
        
        # Load NeRF parameters and sensor data
        with open(nerf_path, 'r') as f:
            nerf_data = json.load(f)
        
        return nerf_data.get("params", {}), nerf_data.get("sensor_data", {})
    
    def __getitem__(self, idx):
        """Get a preference pair by index."""
        pair = self.preference_data[idx]
        
        # Process chosen sample
        chosen_text = pair["chosen"]["text"]
        chosen_image_path = pair["chosen"].get("image_path")
        chosen_image = self.load_image(chosen_image_path)
        chosen_nerf_path = pair["chosen"].get("nerf_path")
        chosen_nerf_params, chosen_sensor_data = self.load_nerf_data(chosen_nerf_path)
        
        # Process rejected sample
        rejected_text = pair["rejected"]["text"]
        rejected_image_path = pair["rejected"].get("image_path")
        rejected_image = self.load_image(rejected_image_path)
        rejected_nerf_path = pair["rejected"].get("nerf_path")
        rejected_nerf_params, rejected_sensor_data = self.load_nerf_data(rejected_nerf_path)
        
        # Tokenize text
        chosen_encodings = self.tokenizer(chosen_text, return_tensors="pt", padding="max_length", 
                                         truncation=True, max_length=512)
        rejected_encodings = self.tokenizer(rejected_text, return_tensors="pt", padding="max_length", 
                                          truncation=True, max_length=512)
        
        # Create sample objects
        chosen_sample = MultimodalSample(
            text=chosen_text,
            image_path=chosen_image_path,
            image=chosen_image,
            nerf_path=chosen_nerf_path,
            nerf_params=chosen_nerf_params,
            sensor_data=chosen_sensor_data
        )
        
        rejected_sample = MultimodalSample(
            text=rejected_text,
            image_path=rejected_image_path,
            image=rejected_image,
            nerf_path=rejected_nerf_path,
            nerf_params=rejected_nerf_params,
            sensor_data=rejected_sensor_data
        )
        
        # Create preference data
        preference = PreferenceData(
            chosen=chosen_sample,
            rejected=rejected_sample,
            consistency_score=pair.get("consistency_score"),
            metadata=pair.get("metadata", {})
        )
        
        return {
            "chosen_input_ids": chosen_encodings.input_ids.squeeze(0),
            "chosen_attention_mask": chosen_encodings.attention_mask.squeeze(0),
            "chosen_image": chosen_image,
            "chosen_nerf_params": chosen_nerf_params,
            "chosen_sensor_data": chosen_sensor_data,
            
            "rejected_input_ids": rejected_encodings.input_ids.squeeze(0),
            "rejected_attention_mask": rejected_encodings.attention_mask.squeeze(0),
            "rejected_image": rejected_image,
            "rejected_nerf_params": rejected_nerf_params,
            "rejected_sensor_data": rejected_sensor_data,
            
            "consistency_score": torch.tensor(preference.consistency_score if preference.consistency_score is not None else 0.0)
        }


def create_dataloaders(
    train_path, 
    val_path,
    test_path,
    tokenizer,
    batch_size=8,
    transform=None
):
    """Create data loaders for training, validation, and testing."""
    # Create datasets
    train_dataset = MultiModalPreferenceDataset(train_path, tokenizer, transform)
    val_dataset = MultiModalPreferenceDataset(val_path, tokenizer, transform) 
    test_dataset = MultiModalPreferenceDataset(test_path, tokenizer, transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset


def preprocess_batch(batch, device):
    """Preprocess a batch for model input."""
    # Move tensors to device
    batch["chosen_input_ids"] = batch["chosen_input_ids"].to(device)
    batch["chosen_attention_mask"] = batch["chosen_attention_mask"].to(device)
    
    batch["rejected_input_ids"] = batch["rejected_input_ids"].to(device)
    batch["rejected_attention_mask"] = batch["rejected_attention_mask"].to(device)
    
    if "consistency_score" in batch:
        batch["consistency_score"] = batch["consistency_score"].to(device)
    
    return batch


# Example data preparation function 
def prepare_sample_data(output_dir, size=20):
    """Generate a small sample dataset for testing purposes."""
    import random
    
    os.makedirs(output_dir, exist_ok=True)
    
    sample_data = []
    
    for i in range(size):
        # Generate random preference pair
        chosen = {
            "text": f"This is a sample chosen text {i} with good alignment between modalities.",
            "image_path": None,  # Would point to actual image in real data
            "nerf_path": None,   # Would point to actual NeRF data in real data
        }
        
        rejected = {
            "text": f"This is a sample rejected text {i} with poor alignment between modalities.",
            "image_path": None,
            "nerf_path": None,
        }
        
        pair = {
            "chosen": chosen,
            "rejected": rejected,
            "consistency_score": random.uniform(0.7, 1.0)
        }
        
        sample_data.append(pair)
    
    # Save splits
    train_size = int(0.7 * size)
    val_size = int(0.15 * size)
    test_size = size - train_size - val_size
    
    with open(os.path.join(output_dir, "train_preferences.json"), "w") as f:
        json.dump(sample_data[:train_size], f, indent=2)
    
    with open(os.path.join(output_dir, "val_preferences.json"), "w") as f:
        json.dump(sample_data[train_size:train_size+val_size], f, indent=2)
    
    with open(os.path.join(output_dir, "test_preferences.json"), "w") as f:
        json.dump(sample_data[train_size+val_size:], f, indent=2)
    
    logger.info(f"Created sample data: {train_size} train, {val_size} val, {test_size} test")
    
    return os.path.join(output_dir, "train_preferences.json"), \
           os.path.join(output_dir, "val_preferences.json"), \
           os.path.join(output_dir, "test_preferences.json")


if __name__ == "__main__":
    # Test data preparation
    prepare_sample_data("./sample_data", size=50)
    print("Sample data created successfully. Run models.py to test model components.")