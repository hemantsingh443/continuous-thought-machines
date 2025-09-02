import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_style('darkgrid')
import torch
import torch.nn as nn
from torch.nn.parameter import UninitializedParameter
from tqdm.auto import tqdm

# project-specific imports (package-relative for -m execution)
from .visual_sorter_dataset import VisualSorterDataset
from .sorter_model import ContinuousThoughtMachine as SorterModel

# Utilities from the original repo
from utils.housekeeping import set_seed
from utils.schedulers import WarmupCosineAnnealingLR

def parse_args():
    parser = argparse.ArgumentParser(description="Train CTM for Visual Sorting")

    # Model Architecture
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--d_input', type=int, default=128)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--iterations', type=int, default=10, help='Number of thought steps. Should match max_seq_len for V1.')
    parser.add_argument('--backbone_type', type=str, default='resnet18-1', help='Use a smaller resnet for this simpler task.')
    parser.add_argument('--synapse_depth', type=int, default=4)
    parser.add_argument('--n_synch_out', type=int, default=128)
    parser.add_argument('--n_synch_action', type=int, default=128)
    parser.add_argument('--memory_length', type=int, default=10)
    parser.add_argument('--memory_hidden_dims', type=int, default=32)

    # Training
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--training_iterations', type=int, default=50001)
    parser.add_argument('--warmup_steps', type=int, default=2000)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    
    # Logging and Data
    parser.add_argument('--log_dir', type=str, default='logs/visual_sorter')
    parser.add_argument('--data_root', type=str, default='visual_sorter_data')
    parser.add_argument('--save_every', type=int, default=5000)
    parser.add_argument('--track_every', type=int, default=1000)
    parser.add_argument('--n_test_batches', type=int, default=50, help='Batches for validation.')
    
    # System
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()
    return args

def train_step(model, batch, criterion, device):
    images, target_coords, _ = batch
    images = images.to(device)
    target_coords = target_coords.to(device) # Shape: (B, max_seq_len, 2)

    # Forward pass
    # Model returns predictions of shape (B, 2, T)
    predictions, _, _ = model(images)
    
    # Reshape predictions to match target: (B, T, 2)
    predicted_sequence = predictions.permute(0, 2, 1)

    # Create a mask to ignore padded values (-1.0) in the target
    mask = (target_coords != -1.0).float()

    # Calculate the raw, un-reduced loss
    raw_loss = criterion(predicted_sequence, target_coords)

    # Apply the mask
    masked_loss = raw_loss * mask
    
    # Calculate the mean loss only over non-padded elements
    # Add a small epsilon to avoid division by zero if a batch has no valid targets
    loss = masked_loss.sum() / (mask.sum() + 1e-8)
    
    return loss

@torch.inference_mode()
def evaluate(model, loader, criterion, device, n_batches):
    model.eval()
    total_loss = 0
    total_samples = 0
    for i, batch in enumerate(loader):
        loss = train_step(model, batch, criterion, device)
        total_loss += loss.item() * batch[0].size(0)
        total_samples += batch[0].size(0)
        if i + 1 >= n_batches:
            break
    model.train()
    return total_loss / total_samples

if __name__=='__main__':
    args = parse_args()
    set_seed(args.seed, True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # --- Data Loading ---
    train_dataset = VisualSorterDataset(data_dir=os.path.join(args.data_root, 'train'), max_seq_len=args.iterations)
    val_dataset = VisualSorterDataset(data_dir=os.path.join(args.data_root, 'val'), max_seq_len=args.iterations)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # --- Model, Optimizer, Scheduler ---
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
        out_dims=2, # Not used by our projector, but good to set
        prediction_reshaper=[args.iterations, 2],
    ).to(args.device)

    # Safely count only initialized parameters (Lazy modules are excluded)
    initialized_params = sum(p.numel() for p in model.parameters() if not isinstance(p, UninitializedParameter))
    print(f'Total initialized params: {initialized_params/1e6:.2f}M')

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = WarmupCosineAnnealingLR(optimizer, args.warmup_steps, args.training_iterations)
    criterion = nn.MSELoss(reduction='none')

    # --- Training Loop ---
    iters = []
    train_losses = []
    val_losses = []
    
    iterator = iter(train_loader)
    pbar = tqdm(range(args.training_iterations))
    
    for i in pbar:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)

        loss = train_step(model, batch, criterion, args.device)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        pbar.set_description(f"Iter {i} | Loss: {loss.item():.4f} | LR: {scheduler.get_last_lr()[0]:.1e}")

        # --- Evaluation and Logging ---
        if i % args.track_every == 0:
            train_loss_eval = evaluate(model, train_loader, criterion, args.device, args.n_test_batches)
            val_loss_eval = evaluate(model, val_loader, criterion, args.device, args.n_test_batches)
            
            iters.append(i)
            train_losses.append(train_loss_eval)
            val_losses.append(val_loss_eval)

            print(f"\nIter {i}: Train Loss = {train_loss_eval:.4f}, Val Loss = {val_loss_eval:.4f}\n")
            
            # Plotting
            plt.figure(figsize=(10, 5))
            plt.plot(iters, train_losses, label=f'Train Loss: {train_loss_eval:.4f}')
            plt.plot(iters, val_losses, label=f'Validation Loss: {val_loss_eval:.4f}')
            plt.xlabel('Iterations')
            plt.ylabel('MSE Loss')
            plt.title('Training and Validation Loss')
            plt.legend()
            plt.ylim(bottom=0)
            plt.savefig(os.path.join(args.log_dir, 'loss_curve.png'))
            plt.close()

        # --- Save Checkpoint ---
        if i % args.save_every == 0 and i > 0:
            checkpoint_path = os.path.join(args.log_dir, f'checkpoint_iter_{i}.pt')
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'iteration': i,
                'args': args
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")