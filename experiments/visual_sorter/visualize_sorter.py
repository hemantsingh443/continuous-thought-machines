import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os

from .visual_sorter_dataset import VisualSorterDataset
from .sorter_model import ContinuousThoughtMachine as SorterModel

def visualize_prediction(checkpoint_path, data_root, sample_idx=0):
    """
    Loads a trained model and visualizes its prediction on a single sample.
    """
    # --- 1. Load Checkpoint and Recreate Model ---
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=False) # Load to CPU
    args = checkpoint['args']
    
    # Re-create the model with the same architecture as during training
    model = SorterModel(
        iterations=args.iterations,
        d_model=args.d_model,
        d_input=args.d_input,
        heads=args.heads,
        n_synch_out=args.n_synch_out,
        n_synch_action=args.n_synch_action,
        synapse_depth=args.synapse_depth,
        memory_length=args.memory_length,
        deep_nlms=True,
        memory_hidden_dims=args.memory_hidden_dims,
        do_layernorm_nlm=False,
        backbone_type=args.backbone_type,
        positional_embedding_type='none',
        out_dims=2,
        prediction_reshaper=[args.iterations, 2],
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval() # Set the model to evaluation mode
    print("Model loaded successfully.")

    # --- 2. Load a Sample from the Validation Set ---
    val_dataset = VisualSorterDataset(data_dir=f"{data_root}/val", max_seq_len=args.iterations)
    image_tensor, target_coords, seq_len = val_dataset[sample_idx]

    # --- 3. Get Model Prediction ---
    with torch.no_grad(): # Disable gradient calculations for inference
        # Add a batch dimension (B, C, H, W) as the model expects a batch
        input_batch = image_tensor.unsqueeze(0)
        
        # Get the sequence of predictions from the model
        predictions_history, _, _ = model(input_batch) # Shape (B, 2, T)
        
        # We'll use the prediction from the final thought step
        final_prediction = predictions_history[0, :, -1] # Shape (2,) for the first item in batch
        
        # The model outputs a sequence, so let's get the full sequence
        # from the final thought step across all output slots
        predicted_sequence = predictions_history[0].permute(1, 0) # Shape (T, 2)

    # --- 4. Prepare Data for Plotting ---
    # Convert tensors to numpy arrays
    image = image_tensor.squeeze().numpy()
    
    # Filter out padded coordinates (-1.0)
    true_coords = target_coords[:seq_len].numpy()
    predicted_coords = predicted_sequence[:seq_len].detach().numpy()
    
    # Un-normalize coordinates to pixel space
    img_size = image.shape[0]
    true_coords_pixels = true_coords * img_size
    predicted_coords_pixels = predicted_coords * img_size

    # --- 5. Visualize ---
    plt.figure(figsize=(8, 8))
    plt.imshow(image, cmap='gray')
    
    # Plot Ground Truth (Green Circles)
    plt.scatter(true_coords_pixels[:, 0], true_coords_pixels[:, 1], 
                s=200, facecolors='none', edgecolors='lime', linewidth=2, label='Ground Truth')
    
    # Plot Model Prediction (Red Crosses)
    plt.scatter(predicted_coords_pixels[:, 0], predicted_coords_pixels[:, 1], 
                s=250, c='red', marker='x', linewidth=2, label='Model Prediction')
                
    # Draw lines to show the sorting order
    for i in range(len(true_coords_pixels) - 1):
        plt.plot(*zip(true_coords_pixels[i], true_coords_pixels[i+1]), 'lime', linestyle='--', alpha=0.7)
        plt.plot(*zip(predicted_coords_pixels[i], predicted_coords_pixels[i+1]), 'red', linestyle='--', alpha=0.7)

    plt.title(f'Visual Sorter Prediction (Sample {sample_idx})')
    plt.legend()
    plt.axis('off')
    
    output_path = os.path.join(os.path.dirname(checkpoint_path), f'prediction_sample_{sample_idx}.png')
    plt.savefig(output_path)
    print(f"\nVisualization saved to: {output_path}")
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize a trained CTM sorter model.")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the model checkpoint .pt file.')
    parser.add_argument('--data_root', type=str, default='visual_sorter_data', help='Root directory of the dataset.')
    parser.add_argument('--sample_idx', type=int, default=0, help='Index of the validation sample to visualize.')
    
    vis_args = parser.parse_args()
    
    visualize_prediction(vis_args.checkpoint, vis_args.data_root, vis_args.sample_idx)