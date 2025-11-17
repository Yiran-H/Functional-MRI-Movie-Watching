# -*- coding: utf-8 -*-
"""
extract_alexnet_static.py
Extract static AlexNet features from video frames for fMRI analysis.

Based on: OSF - Ubiquitous cortical sensitivity to visual information 
during naturalistic audiovisual movie viewing

Author: Generated for fMRI Video Analysis Project
Date: 2025-11-03
"""

import os
import argparse
import csv
import time
from PIL import Image
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from tqdm import tqdm


class AlexNetFeatureExtractor:
    """
    Extract features from all layers of AlexNet using global average pooling.
    
    Architecture:
    - Conv1: (64, 55, 55) -> global avg pool -> 64-dim
    - Conv2: (192, 27, 27) -> global avg pool -> 192-dim
    - Conv3: (384, 13, 13) -> global avg pool -> 384-dim
    - Conv4: (256, 13, 13) -> global avg pool -> 256-dim
    - Conv5: (256, 13, 13) -> global avg pool -> 256-dim
    - FC6: (4096,) -> 4096-dim
    - FC7: (4096,) -> 4096-dim
    - FC8: (1000,) -> 1000-dim
    
    Total: 10,344 dimensions
    """
    
    def __init__(self, device='cuda'):
        """
        Initialize AlexNet with ImageNet pretrained weights.
        
        Args:
            device: 'cuda' or 'cpu'
        """
        self.device = device
        print(f"🔧 Loading AlexNet (pretrained on ImageNet)...")
        
        # Load pretrained AlexNet
        alexnet = models.alexnet(pretrained=True)
        alexnet = alexnet.to(device)
        alexnet.eval()
        
        self.model = alexnet
        self.features_dict = {}
        
        # Define layer names and their positions
        self.layer_info = {
            'conv1': ('features', 0),   # Conv2d(3, 64, kernel_size=11, stride=4)
            'conv2': ('features', 3),   # Conv2d(64, 192, kernel_size=5)
            'conv3': ('features', 6),   # Conv2d(192, 384, kernel_size=3)
            'conv4': ('features', 8),   # Conv2d(384, 256, kernel_size=3)
            'conv5': ('features', 10),  # Conv2d(256, 256, kernel_size=3)
            'fc6': ('classifier', 1),   # Linear(9216, 4096)
            'fc7': ('classifier', 4),   # Linear(4096, 4096)
            'fc8': ('classifier', 6),   # Linear(4096, 1000)
        }
        
        # Expected dimensions after global average pooling
        self.expected_dims = {
            'conv1': 64,
            'conv2': 192,
            'conv3': 384,
            'conv4': 256,
            'conv5': 256,
            'fc6': 4096,
            'fc7': 4096,
            'fc8': 1000
        }
        
        # ImageNet normalization (standard)
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        print(f"✅ AlexNet loaded on {device}")
        print(f"📊 Total feature dimensions: {sum(self.expected_dims.values())}")
        
    def _get_activation(self, name):
        """Create hook function to save layer activation."""
        def hook(model, input, output):
            self.features_dict[name] = output.detach()
        return hook
    
    def extract_features(self, image):
        """
        Extract features from all AlexNet layers.
        
        Args:
            image: PIL Image
            
        Returns:
            dict: {layer_name: feature_vector (numpy array)}
        """
        # Register forward hooks
        hooks = []
        for layer_name, (module_name, layer_idx) in self.layer_info.items():
            if module_name == 'features':
                layer = self.model.features[layer_idx]
            else:  # classifier
                layer = self.model.classifier[layer_idx]
            
            hook = layer.register_forward_hook(self._get_activation(layer_name))
            hooks.append(hook)
        
        # Preprocess image
        img_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        # Forward pass
        self.features_dict = {}
        with torch.no_grad():
            _ = self.model(img_tensor)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Process features
        processed_features = {}
        for layer_name, feat_tensor in self.features_dict.items():
            if len(feat_tensor.shape) == 4:  # Conv layers: (B, C, H, W)
                # Global average pooling
                feat_flat = feat_tensor.mean(dim=[2, 3]).squeeze(0)
            else:  # FC layers: (B, D)
                feat_flat = feat_tensor.squeeze(0)
            
            # Convert to numpy
            feat_numpy = feat_flat.cpu().numpy()
            
            # Verify dimension
            expected_dim = self.expected_dims[layer_name]
            assert feat_numpy.shape[0] == expected_dim, \
                f"Layer {layer_name}: expected {expected_dim}, got {feat_numpy.shape[0]}"
            
            processed_features[layer_name] = feat_numpy
        
        return processed_features


def extract_frames_from_video(video_path, fps=1.0):
    """
    Extract frames from video at specified fps.
    
    Args:
        video_path: Path to video file
        fps: Frames per second to sample
        
    Returns:
        list: [(frame_index, timestamp, PIL_image), ...]
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps
    
    print(f"📹 Video info:")
    print(f"   FPS: {video_fps:.2f}")
    print(f"   Total frames: {total_frames}")
    print(f"   Duration: {duration:.2f}s")
    print(f"   Sampling at: {fps} fps")
    
    frame_interval = int(video_fps / fps)
    frames = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            timestamp = frame_count / video_fps
            # BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            frames.append((len(frames), timestamp, pil_image))
        
        frame_count += 1
    
    cap.release()
    print(f"✅ Extracted {len(frames)} frames")
    return frames


def generate_csv_header():
    """
    Generate CSV header with all feature column names.
    
    Returns:
        list: Column names
    """
    header = ['frame_index', 'timestamp_sec']
    
    # Define layer order (same as extraction order)
    layers = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8']
    dims = [64, 192, 384, 256, 256, 4096, 4096, 1000]
    
    for layer_name, dim in zip(layers, dims):
        for i in range(dim):
            header.append(f'{layer_name}_{i}')
    
    return header


def main():
    parser = argparse.ArgumentParser(
        description='Extract static AlexNet features from video frames'
    )
    parser.add_argument('--video', required=True, help='Input video file')
    parser.add_argument('--output', required=True, help='Output CSV file')
    parser.add_argument('--fps', type=float, default=1.0, 
                        help='Frames per second to sample (default: 1.0)')
    parser.add_argument('--device', default='cuda', 
                        help='Device to use: cuda or cpu (default: cuda)')
    args = parser.parse_args()
    
    # Check CUDA availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    print("=" * 80)
    print("🧠 ALEXNET STATIC FEATURE EXTRACTION")
    print("=" * 80)
    print(f"Video: {args.video}")
    print(f"Output: {args.output}")
    print(f"Sampling: {args.fps} fps")
    print(f"Device: {args.device}")
    print("=" * 80)
    print()
    
    # Check video file
    if not os.path.exists(args.video):
        raise FileNotFoundError(f"Video not found: {args.video}")
    
    # Step 1: Initialize AlexNet
    print("=" * 80)
    print("Step 1/3: Loading AlexNet model...")
    print("=" * 80)
    
    extractor = AlexNetFeatureExtractor(device=args.device)
    
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 VRAM Total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    print()
    
    # Step 2: Extract frames
    print("=" * 80)
    print("Step 2/3: Extracting frames from video...")
    print("=" * 80)
    
    frames = extract_frames_from_video(args.video, args.fps)
    print()
    
    # Step 3: Extract features
    print("=" * 80)
    print("Step 3/3: Extracting AlexNet features...")
    print("=" * 80)
    
    results = []
    total_time = 0
    
    for frame_idx, timestamp, image in tqdm(frames, desc="Processing frames"):
        t0 = time.time()
        
        # Extract features
        features = extractor.extract_features(image)
        
        dt = time.time() - t0
        total_time += dt
        
        # Combine all features in the correct order
        layer_order = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8']
        all_features = []
        for layer_name in layer_order:
            all_features.extend(features[layer_name].tolist())
        
        # Store result
        results.append({
            'frame_index': frame_idx,
            'timestamp': timestamp,
            'features': all_features
        })
        
        # Print progress every 10 frames
        if (frame_idx + 1) % 10 == 0:
            avg_time = total_time / (frame_idx + 1)
            remaining = len(frames) - (frame_idx + 1)
            eta = avg_time * remaining
            if torch.cuda.is_available():
                vram_used = torch.cuda.memory_allocated(0) / 1024**3
                print(f"  [{frame_idx+1}/{len(frames)}] Avg: {avg_time:.2f}s/frame | "
                      f"ETA: {eta:.0f}s | VRAM: {vram_used:.1f}GB")
    
    avg_time = total_time / len(frames) if frames else 0
    print(f"\n✅ Feature extraction complete!")
    print(f"   Average time: {avg_time:.2f}s/frame")
    print(f"   Total time: {total_time/60:.1f} minutes")
    print()
    
    # Step 4: Save to CSV
    print("=" * 80)
    print("Step 4/4: Saving to CSV...")
    print("=" * 80)
    
    header = generate_csv_header()
    
    with open(args.output, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        
        for result in results:
            row = [
                result['frame_index'],
                f"{result['timestamp']:.3f}"
            ] + result['features']
            writer.writerow(row)
    
    print(f"✅ CSV saved: {args.output}")
    
    # Print statistics
    print()
    print("=" * 80)
    print("📊 STATISTICS")
    print("=" * 80)
    print(f"Total frames processed: {len(results)}")
    print(f"Feature dimensions per frame: {len(results[0]['features'])}")
    print(f"CSV columns: {len(header)}")
    print(f"File size: {os.path.getsize(args.output) / 1024**2:.1f} MB")
    print()
    
    print("=" * 80)
    print("✅✅✅ ALL DONE! ✅✅✅")
    print("=" * 80)


if __name__ == "__main__":
    main()
