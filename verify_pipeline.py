
import torch
import os
from torch.utils.data import DataLoader
from dataset import AudioDataset, AUDIO_CONFIG
import sys

def verify():
    print("--- Starting Verification ---")
    
    # Check if data list exists
    if not os.path.exists("data/train_list.txt"):
        print("Error: data/train_list.txt not found. Please run prepare_deepship_data.py first.")
        # Create dummy for testing logic if needed?
        # No, better to fail fast.
        return

    # 1. Initialize Dataset
    print("\n1. Initializing AudioDataset...")
    # Use mini list if available, else standard
    list_path = "data/train_list_mini.txt" if os.path.exists("data/train_list_mini.txt") else "data/train_list.txt"
    dataset = AudioDataset(list_path, train=True)
    print(f"Dataset length: {len(dataset)}")
    
    if len(dataset) == 0:
        print("Dataset empty!")
        return

    # 2. Check Item 0
    print("\n2. Checking __getitem__(0)...")
    try:
        item = dataset[0]
        mel, label_a, label_b, lam = item
        print(f"Item 0 shapes: Mel={mel.shape}, LabelA={label_a}, LabelB={label_b}, Lam={lam}")
        
        # Check Mel Shape
        # Should be (n_mels, time_steps)
        # target_length_frames = 3 * 16000 = 48000
        # hop_length = 160 -> frames = 48000 / 160 = 300 + 1 (center=True/False?)
        # T.MelSpectrogram uses center=True by default?
        # Let's see shapes.
    except Exception as e:
        print(f"Error getting item 0: {e}")
        import traceback
        traceback.print_exc()

    # 3. DataLoader Test
    print("\n3. Testing DataLoader iteration (Batch Size 4)...")
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    try:
        mixup_detected = False
        for batch_idx, (mels, labels_a, labels_b, lams) in enumerate(loader):
            print(f"Batch {batch_idx}: Mels={mels.shape}, La={labels_a.shape}, Lb={labels_b.shape}, Lam={lams}")
            
            if (lams < 1.0).any():
                print("  -> Mixup active in this batch!")
                mixup_detected = True
                
            if batch_idx >= 4 and mixup_detected:
                break
            if batch_idx >= 10: # Stop after 10 batches anyway
                break
                
        if not mixup_detected:
             print("Warning: Mixup not detected in first 10 batches (p=0.4). This is possible but unlikely.")
        else:
             print("Success: Mixup verified.")

    except Exception as e:
        print(f"Error in DataLoader: {e}")
        import traceback
        traceback.print_exc()

    # 4. Model Forward Pass Test (Optional, requires model code)
    print("\n4. Testing Model Forward Pass (Mock)...")
    try:
        # Mock model to test shape compatibility
        model = torch.nn.Conv2d(1, 4, kernel_size=3, padding=1)
        output = model(mels) # Use last batch mels
        print(f"Model output shape (Mock): {output.shape}")
        
        # Calculate Loss (Mock)
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(torch.randn(4, 4), labels_a) * lams.mean() # just syntax check
        print("Loss calculation syntax valid.")
        
    except Exception as e:
        print(f"Error in Model Test: {e}")

    print("\n--- Verification Complete ---")

if __name__ == "__main__":
    verify()
