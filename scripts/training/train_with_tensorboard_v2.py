import os
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from diffalign import DiffAlign
from dataset import GeomPairsDataset, collate_fn
from diffalign.models.epsnet.diffusion import merge_graphs_in_batch

# --- Configuration ---
CHECKPOINT_DIR = 'checkpoints'
EPOCH_50_WEIGHTS = os.path.join(CHECKPOINT_DIR, 'model_epoch_50.pt')
RESUME_CHECKPOINT = os.path.join(CHECKPOINT_DIR, 'last_state.pt')

LOG_DIR = 'runs/experiment_1'
DATA_PATH = 'data/training_pairs_filtered.pkl'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
TARGET_EPOCHS = 200
PATIENCE = 15 

# --- Initialization ---
model = DiffAlign().to(DEVICE)
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
writer = SummaryWriter(LOG_DIR)

start_epoch = 0
best_val_loss = float('inf')
epochs_without_improvement = 0

# --- Strict Resume Logic ---
if os.path.exists(RESUME_CHECKPOINT):
    print(f"[*] Found existing session. Resuming from: {RESUME_CHECKPOINT}")
    checkpoint = torch.load(RESUME_CHECKPOINT, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))
elif os.path.exists(EPOCH_50_WEIGHTS):
    print(f"[*] Found Epoch 50 milestone. Loading: {EPOCH_50_WEIGHTS}")
    model.load_state_dict(torch.load(EPOCH_50_WEIGHTS, map_location=DEVICE))
    start_epoch = 50
    print(f"[*] Initializing training from Epoch {start_epoch}")
else:
    raise FileNotFoundError("Could not find model_epoch_50.pt. Please check the file path.")

# --- Data Loading ---
full_dataset = GeomPairsDataset(DATA_PATH)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

def run_step(query_batch, ref_batch, model, device):
    query_batch, ref_batch = query_batch.to(device), ref_batch.to(device)
    t = torch.randint(0, model.num_timesteps, (query_batch.num_graphs,), device=device)
    
    batch_idx_expanded = t[query_batch.batch].unsqueeze(-1)
    sqrt_alpha = model.sqrt_alphas_cumprod[batch_idx_expanded]
    sqrt_one_minus_alpha = model.sqrt_one_minus_alphas_cumprod[batch_idx_expanded]
    
    noise = torch.randn_like(query_batch.pos)
    x_0 = query_batch.pos
    x_t = sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise
    
    query_batch.pos = x_t
    v_pred_merged = model(query_batch, ref_batch, t)
    
    merged = merge_graphs_in_batch(query_batch, ref_batch, device=device)
    query_mask = (merged.graph_idx % 2 == 0)
    v_pred = v_pred_merged[query_mask]
    
    v_target = sqrt_alpha * noise - sqrt_one_minus_alpha * x_0
    return torch.nn.functional.mse_loss(v_pred, v_target)

# --- Training Loop ---
print(f"Targeting {TARGET_EPOCHS} epochs on {DEVICE}...")

for epoch in range(start_epoch, TARGET_EPOCHS):
    # 1. Training
    model.train()
    train_loss = 0
    for batch in train_loader:
        loss = run_step(*batch, model, DEVICE)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    avg_train_loss = train_loss / len(train_loader)
    
    # 2. Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            loss = run_step(*batch, model, DEVICE)
            val_loss += loss.item()
    
    avg_val_loss = val_loss / len(val_loader)
    
    # 3. Logging & Terminal Output
    print(f"Epoch {epoch+1}/{TARGET_EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
    writer.add_scalar('Loss/Train', avg_train_loss, epoch)
    writer.add_scalar('Loss/Validation', avg_val_loss, epoch)

    # 4. Checkpointing
    current_state = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_loss': best_val_loss
    }
    
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_without_improvement = 0
        torch.save(current_state, os.path.join(CHECKPOINT_DIR, 'best_model.pt'))
    else:
        epochs_without_improvement += 1
    
    torch.save(current_state, RESUME_CHECKPOINT)
    
    if (epoch + 1) % 10 == 0:
        torch.save(current_state, os.path.join(CHECKPOINT_DIR, f'model_epoch_{epoch+1}.pt'))

    if epochs_without_improvement >= PATIENCE:
        print(f"Early stopping at epoch {epoch+1}. No progress for {PATIENCE} epochs.")
        break

writer.close()