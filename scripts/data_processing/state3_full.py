# scripts/data_processing/stage3_full_msgpack.py
"""
Stage 3: Extract training pairs from GEOM-Drugs (MSGPACK format)
Modified to work with drugs_crude.msgpack instead of HDF5
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

import torch
import pickle
import msgpack
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from datetime import datetime
import gc

# Define paths
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA = DATA_DIR / "processed"
CHECKPOINTS = DATA_DIR / "checkpoints"

PROCESSED_DATA.mkdir(parents=True, exist_ok=True)
CHECKPOINTS.mkdir(parents=True, exist_ok=True)

# MSGPACK file path
GEOM_MSGPACK = DATA_DIR / "drugs_crude.msgpack"
OUTPUT_PATH = PROCESSED_DATA / "training_pairs_full_dataset.pkl"

CONFIG = {
    'target_similar_pairs': 65000,
    'target_conformer_pairs': 20000,
    'max_candidates': 100000,  # Reduced for msgpack (slower to read)
    'batch_size': 500,
    'checkpoint_every': 5000,
    'tanimoto_min': 0.55,
    'tanimoto_max': 0.85,
    'shape_score_min': 0.5,
}

print("="*80)
print(f"STAGE 3: FULL GEOM-DRUGS PAIR EXTRACTION (MSGPACK)")
print(f"Started: {datetime.now()}")
print(f"Target: {CONFIG['target_similar_pairs']} similar + {CONFIG['target_conformer_pairs']} conformer pairs")
print(f"GEOM-Drugs msgpack: {GEOM_MSGPACK}")
print("="*80)

# Check if file exists
if not GEOM_MSGPACK.exists():
    print(f"ERROR: GEOM-Drugs msgpack not found at {GEOM_MSGPACK}")
    sys.exit(1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")


def compute_2d_tanimoto(smiles_i, smiles_j):
    """Compute 2D Tanimoto similarity"""
    try:
        mol_i = Chem.MolFromSmiles(smiles_i)
        mol_j = Chem.MolFromSmiles(smiles_j)
        
        if mol_i is None or mol_j is None:
            return 0.0
        
        fp_i = AllChem.GetMorganFingerprintAsBitVect(mol_i, radius=2, nBits=2048)
        fp_j = AllChem.GetMorganFingerprintAsBitVect(mol_j, radius=2, nBits=2048)
        
        return DataStructs.TanimotoSimilarity(fp_i, fp_j)
    except:
        return 0.0


def kabsch_align(coords_mobile, coords_target):
    """Align using Kabsch algorithm"""
    center_mobile = coords_mobile.mean(dim=0, keepdim=True)
    center_target = coords_target.mean(dim=0, keepdim=True)
    
    coords_mobile_centered = coords_mobile - center_mobile
    coords_target_centered = coords_target - center_target
    
    H = coords_mobile_centered.T @ coords_target_centered
    U, S, Vt = torch.linalg.svd(H)
    R = Vt.T @ U.T
    
    if torch.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    coords_aligned = coords_mobile_centered @ R + center_target
    return coords_aligned


def compute_3d_shape_score(coords_i, coords_j):
    """Compute shape similarity"""
    try:
        coords_i_aligned = kabsch_align(coords_i, coords_j)
        rmsd = torch.sqrt(((coords_i_aligned - coords_j)**2).sum() / len(coords_i))
        score = torch.exp(-rmsd / 2.0).item()
        return score
    except:
        return 0.0


# =============================================================================
# STAGE 1: LOAD AND INDEX MSGPACK
# =============================================================================

print("\n" + "="*80)
print("[1/5] LOADING MSGPACK AND BUILDING INDEX")
print("="*80)

index_path = PROCESSED_DATA / "molecule_index_msgpack.pkl"

if index_path.exists():
    print(f"Loading existing index from {index_path}")
    with open(index_path, 'rb') as f:
        molecule_index = pickle.load(f)
    print(f"Loaded {len(molecule_index)} molecules")
else:
    print("Reading msgpack file (this may take a while)...")
    
    molecule_index = []
    
    with open(GEOM_MSGPACK, 'rb') as f:
        unpacker = msgpack.Unpacker(f, raw=False, max_buffer_size=2**31-1)
        
        for idx, mol_data in enumerate(tqdm(unpacker, desc="Reading msgpack")):
            try:
                # msgpack structure: each entry is a molecule dict
                smiles = mol_data.get('smiles', '')
                if not smiles:
                    continue
                
                # Get conformers
                conformers = mol_data.get('conformers', [])
                if len(conformers) == 0:
                    continue
                
                molecule_index.append({
                    'mol_id': f"mol_{idx}",
                    'smiles': smiles,
                    'conformers': conformers,  # Store all conformers
                    'num_conformers': len(conformers),
                })
                
            except Exception as e:
                continue
    
    print(f"Indexed {len(molecule_index)} molecules")
    
    # Save index
    with open(index_path, 'wb') as f:
        pickle.dump(molecule_index, f)
    print(f"Saved index to {index_path}")


# =============================================================================
# STAGE 2: FIND CANDIDATE PAIRS (2D TANIMOTO)
# =============================================================================

print("\n" + "="*80)
print("[2/5] FINDING CANDIDATE PAIRS (2D TANIMOTO)")
print(f"Filter: Tanimoto ∈ [{CONFIG['tanimoto_min']}, {CONFIG['tanimoto_max']}]")
print("="*80)

candidates_path = CHECKPOINTS / "candidate_pairs_msgpack.pkl"

if candidates_path.exists():
    print(f"Loading existing candidates from {candidates_path}")
    with open(candidates_path, 'rb') as f:
        candidate_pairs = pickle.load(f)
    print(f"Loaded {len(candidate_pairs)} candidate pairs")
else:
    print("Computing 2D Tanimoto similarities...")
    
    candidate_pairs = []
    batch_size = CONFIG['batch_size']
    
    molecule_index_sorted = sorted(molecule_index, key=lambda x: x['smiles'])
    
    for i in tqdm(range(0, len(molecule_index_sorted), batch_size), desc="Batches"):
        if len(candidate_pairs) >= CONFIG['max_candidates']:
            print(f"\nReached target of {CONFIG['max_candidates']} candidates")
            break
        
        batch = molecule_index_sorted[i:i+batch_size]
        
        for mol_i in batch:
            for mol_j in molecule_index_sorted[i:]:
                if mol_i['smiles'] >= mol_j['smiles']:
                    continue
                
                tanimoto = compute_2d_tanimoto(mol_i['smiles'], mol_j['smiles'])
                
                if CONFIG['tanimoto_min'] <= tanimoto <= CONFIG['tanimoto_max']:
                    candidate_pairs.append({
                        'mol_i': mol_i,  # Store full molecule data
                        'mol_j': mol_j,
                        'tanimoto_2d': tanimoto,
                    })
        
        if (i // batch_size) % 10 == 0 and len(candidate_pairs) > 0:
            checkpoint_path = CHECKPOINTS / f"candidates_checkpoint_{len(candidate_pairs)}.pkl"
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(candidate_pairs, f)
            print(f"\n  Checkpoint: {len(candidate_pairs)} candidates saved")
        
        gc.collect()
    
    print(f"\nFound {len(candidate_pairs)} candidate pairs")
    
    with open(candidates_path, 'wb') as f:
        pickle.dump(candidate_pairs, f)
    print(f"Saved candidates to {candidates_path}")


# =============================================================================
# STAGE 3: APPLY 3D SHAPE FILTER
# =============================================================================

print("\n" + "="*80)
print("[3/5] APPLYING 3D SHAPE FILTER")
print(f"Filter: Shape score > {CONFIG['shape_score_min']}")
print(f"Target: {CONFIG['target_similar_pairs']} pairs")
print("="*80)

training_pairs = []
failed_pairs = 0

for idx, pair in enumerate(tqdm(candidate_pairs, desc="Processing pairs")):
    if len(training_pairs) >= CONFIG['target_similar_pairs']:
        print(f"\nReached target of {CONFIG['target_similar_pairs']} training pairs!")
        break
    
    try:
        mol_i = pair['mol_i']
        mol_j = pair['mol_j']
        
        # Get first conformer from each
        conformers_i = mol_i['conformers']
        conformers_j = mol_j['conformers']
        
        if len(conformers_i) == 0 or len(conformers_j) == 0:
            failed_pairs += 1
            continue
        
        # Convert to torch tensors
        coords_i = torch.tensor(conformers_i[0]['coords'], device=device, dtype=torch.float32)
        coords_j = torch.tensor(conformers_j[0]['coords'], device=device, dtype=torch.float32)
        
        # Compute shape score
        shape_score = compute_3d_shape_score(coords_i, coords_j)
        
        if shape_score > CONFIG['shape_score_min']:
            training_pairs.append({
                'smiles_i': mol_i['smiles'],
                'smiles_j': mol_j['smiles'],
                'coords_i': coords_i.cpu().numpy(),
                'coords_j': coords_j.cpu().numpy(),
                'tanimoto_2d': pair['tanimoto_2d'],
                'shape_score': shape_score,
            })
        
    except Exception as e:
        failed_pairs += 1
        continue
    
    if len(training_pairs) % CONFIG['checkpoint_every'] == 0 and len(training_pairs) > 0:
        checkpoint_path = CHECKPOINTS / f"training_pairs_checkpoint_{len(training_pairs)}.pkl"
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(training_pairs, f)
        print(f"\n  Checkpoint: {len(training_pairs)} training pairs saved")
    
    if idx % 1000 == 0:
        torch.cuda.empty_cache()

print(f"\nCreated {len(training_pairs)} training pairs from similar molecules")
print(f"Failed: {failed_pairs} pairs")


# =============================================================================
# STAGE 4: ADD CONFORMER PAIRS
# =============================================================================

print("\n" + "="*80)
print("[4/5] ADDING CONFORMER PAIRS")
print(f"Target: {CONFIG['target_conformer_pairs']} pairs")
print("="*80)

conformer_pairs = []

molecules_with_conformers = [m for m in molecule_index if m['num_conformers'] >= 2]
print(f"Found {len(molecules_with_conformers)} molecules with 2+ conformers")

n_sample = min(10000, len(molecules_with_conformers))
sampled_molecules = np.random.choice(molecules_with_conformers, size=n_sample, replace=False)

for mol_info in tqdm(sampled_molecules, desc="Extracting conformer pairs"):
    if len(conformer_pairs) >= CONFIG['target_conformer_pairs']:
        break
    
    try:
        conformers = mol_info['conformers']
        n_conformers = len(conformers)
        
        if n_conformers < 2:
            continue
        
        n_pairs = min(3, n_conformers - 1)
        for _ in range(n_pairs):
            i, j = np.random.choice(n_conformers, size=2, replace=False)
            
            conformer_pairs.append({
                'smiles_i': mol_info['smiles'],
                'smiles_j': mol_info['smiles'],
                'coords_i': conformers[i]['coords'],
                'coords_j': conformers[j]['coords'],
                'tanimoto_2d': 1.0,
                'shape_score': 1.0,
            })
            
            if len(conformer_pairs) >= CONFIG['target_conformer_pairs']:
                break
            
    except:
        continue

print(f"Added {len(conformer_pairs)} conformer pairs")


# =============================================================================
# STAGE 5: SAVE
# =============================================================================

print("\n" + "="*80)
print("[5/5] FINALIZING DATASET")
print("="*80)

all_pairs = training_pairs + conformer_pairs
print(f"\nTotal training pairs: {len(all_pairs)}")
print(f"  - Similar molecule pairs: {len(training_pairs)}")
print(f"  - Conformer pairs: {len(conformer_pairs)}")

final_output = PROCESSED_DATA / f"training_pairs_msgpack_{len(all_pairs)}.pkl"
with open(final_output, 'wb') as f:
    pickle.dump(all_pairs, f)
print(f"\nSaved to: {final_output}")

with open(OUTPUT_PATH, 'wb') as f:
    pickle.dump(all_pairs, f)
print(f"Saved to: {OUTPUT_PATH}")

summary = {
    'total_pairs': len(all_pairs),
    'similar_molecule_pairs': len(training_pairs),
    'conformer_pairs': len(conformer_pairs),
    'total_molecules_indexed': len(molecule_index),
    'candidate_pairs_evaluated': len(candidate_pairs),
    'failed_pairs': failed_pairs,
    'config': CONFIG,
    'date': datetime.now().isoformat(),
}

summary_path = PROCESSED_DATA / "stage3_msgpack_summary.pkl"
with open(summary_path, 'wb') as f:
    pickle.dump(summary, f)

print("\n" + "="*80)
print("STAGE 3 COMPLETE!")
print("="*80)
print(f"Finished: {datetime.now()}")
print(f"\nDataset Statistics:")
print(f"  Total pairs: {len(all_pairs):,}")
print(f"  Similar molecules: {len(training_pairs):,}")
print(f"  Conformer pairs: {len(conformer_pairs):,}")
print(f"\nFiles saved:")
print(f"  Dataset: {OUTPUT_PATH}")
print(f"  Summary: {summary_path}")
print("="*80)