# train_with_tensorboard.py (COMPLETE)
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from diffalign import DiffAlign
from dataset import GeomPairsDataset, collate_fn

# Setup
device = torch.device('cuda')
model = DiffAlign().to(device)
optimizer = AdamW(model.parameters(), lr=1e-4)

# TensorBoard writer
writer = SummaryWriter('runs/experiment_1')

# Load data
dataset = GeomPairsDataset('data/training_pairs_filtered.pkl')
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

print(f"Training on {device}")
print(f"Dataset: {len(dataset)} pairs")
print(f"Batches per epoch: {len(dataloader)}")

num_epochs = 50
T = model.num_timesteps

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    
    for batch_idx, (query_batch, ref_batch) in enumerate(dataloader):
        query_batch = query_batch.to(device)
        ref_batch = ref_batch.to(device)
        
        # Random timestep
        t = torch.randint(0, T, (query_batch.num_graphs,), device=device)
        
        # Forward diffusion
        noise = torch.randn_like(query_batch.pos)
        sqrt_alpha = model.sqrt_alphas_cumprod[t[query_batch.batch]].unsqueeze(-1)
        sqrt_one_minus_alpha = model.sqrt_one_minus_alphas_cumprod[t[query_batch.batch]].unsqueeze(-1)
        
        x_0 = query_batch.pos
        x_t = sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise
        
        # Predict
        query_batch.pos = x_t
        v_pred_merged = model(query_batch, ref_batch, t)
        
        # Extract query predictions
        from diffalign.models.epsnet.diffusion import merge_graphs_in_batch
        merged = merge_graphs_in_batch(query_batch, ref_batch, device=device)
        query_mask = (merged.graph_idx % 2 == 0)
        v_pred = v_pred_merged[query_mask]
        
        # v-target
        alpha_t = model.sqrt_alphas_cumprod[t[query_batch.batch]].unsqueeze(-1)
        sigma_t = model.sqrt_one_minus_alphas_cumprod[t[query_batch.batch]].unsqueeze(-1)
        v_target = alpha_t * noise - sigma_t * x_0
        
        # Loss
        loss = torch.nn.functional.mse_loss(v_pred, v_target)
        
        # Log batch loss to TensorBoard
        global_step = epoch * len(dataloader) + batch_idx
        writer.add_scalar('Loss/train_batch', loss.item(), global_step)
        
        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(dataloader)
    
    # Log epoch metrics
    writer.add_scalar('Loss/train_epoch', avg_loss, epoch)
    writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    # Save checkpoints
    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), f'checkpoints/model_epoch_{epoch+1}.pt')

print("Training complete")
torch.save(model.state_dict(), 'trained_model_tensorboard.pt')
writer.close()
