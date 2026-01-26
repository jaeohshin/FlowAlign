# test_model.py (fixed)
import torch
from diffalign import DiffAlign
from dataset import GeomPairsDataset
from torch_geometric.data import Batch
import numpy as np

# Load model
device = torch.device('cuda')
model = DiffAlign().to(device)
model.load_state_dict(torch.load('trained_model_real.pt'))
model.eval()

print("Model loaded successfully")

# Load test data
dataset = GeomPairsDataset('data/training_pairs.pkl')
query_data, ref_data = dataset[0]  # Single Data object

# Create batch from single item
query_batch = Batch.from_data_list([query_data]).to(device)
ref_batch = Batch.from_data_list([ref_data]).to(device)

print(f"\nQuery atoms: {query_batch.num_nodes}")
print(f"Reference atoms: {ref_batch.num_nodes}")

# Generate alignment (no UFF for now)
print("\nGenerating alignment...")
with torch.no_grad():
    generated_pos, _ = model.DDPM_Sampling_UFF(
        query_batch,
        ref_batch,
        cfg_scale=1.0,
        uff_guidance_scale=0.0,
        query_mols=None,
        pocket_mols=None,
    )

print(f"Generated coordinates shape: {generated_pos.shape}")

# Calculate RMSD
query_original = query_data.pos.numpy()
ref_coords = ref_data.pos.numpy()
generated = generated_pos.cpu().numpy()

def rmsd(coords1, coords2):
    return np.sqrt(np.mean((coords1 - coords2)**2))

rmsd_original = rmsd(query_original, ref_coords)
rmsd_generated = rmsd(generated, ref_coords)

print(f"\nRMSD (original query → reference): {rmsd_original:.3f} Å")
print(f"RMSD (generated → reference):      {rmsd_generated:.3f} Å")
print(f"Improvement: {rmsd_original - rmsd_generated:.3f} Å")

if rmsd_generated < rmsd_original:
    print("\n✓ Model improved alignment")
else:
    print("\n✗ Model didn't improve (may need more training or data)")