import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from dataset import AudioDataset
from preprocess import AUDIO_CONFIG
from collections import Counter

def inspect_data():
    if not os.path.exists('data/train_list.txt'):
        print("Error: data/train_list.txt not found.")
        return

    print("--- Data Inspection ---")
    dataset = AudioDataset('data/train_list.txt', train=True)
    print(f"Dataset Size: {len(dataset)}")
    
    # Check Class Distribution
    labels = []
    for line in dataset.lines:
        try:
            _, label = line.split('\t')
            labels.append(int(label))
        except:
            pass
    
    counts = Counter(labels)
    print(f"Class Distribution: {counts}")
    
    # Get a sample
    print("\n--- Sample Inspection ---")
    idx = 0
    mel, label_a, label_b, lam = dataset[idx]
    
    print(f"Sample {idx}:")
    print(f"  Mel Shape: {mel.shape}")
    print(f"  Label A: {label_a}, Label B: {label_b}, Lam: {lam}")
    print(f"  Mel Min: {mel.min():.4f}, Max: {mel.max():.4f}, Mean: {mel.mean():.4f}")
    
    # Plot
    plt.figure(figsize=(10, 4))
    plt.imshow(mel.squeeze().numpy(), origin='lower', aspect='auto')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f"Mel Spectrogram (Class {label_a})")
    plt.tight_layout()
    plt.savefig('sample_spectrogram.png')
    print("Saved sample_spectrogram.png")

if __name__ == "__main__":
    inspect_data()
