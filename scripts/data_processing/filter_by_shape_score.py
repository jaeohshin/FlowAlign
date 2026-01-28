import pickle
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign
import numpy as np
from tqdm import tqdm

print("Loading unfiltered pairs...")
pairs = pickle.load(open('data/processed/training_pairs_unfiltered.pkl', 'rb'))
print(f"Loaded {len(pairs)} pairs")

print("\nFiltering by 3D shape score > 0.5...")
filtered_pairs = []

for pair in tqdm(pairs):
    try:
        # Get molecules with 3D coordinates
        mol_q = pair['mol_query']
        mol_r = pair['mol_ref']
        
        # Set conformer coordinates
        conf_q = Chem.Conformer(mol_q.GetNumAtoms())
        conf_r = Chem.Conformer(mol_r.GetNumAtoms())
        
        for i, coord in enumerate(pair['query_coords']):
            conf_q.SetAtomPosition(i, coord)
        
        for i, coord in enumerate(pair['ref_coords']):
            conf_r.SetAtomPosition(i, coord)
        
        mol_q.AddConformer(conf_q, assignId=True)
        mol_r.AddConformer(conf_r, assignId=True)
        
        # Align query to reference
        rmsd = rdMolAlign.AlignMol(mol_q, mol_r)
        
        # Calculate shape score (inverse of RMSD, normalized)
        # Lower RMSD = higher shape score
        # Paper uses "3D shape score > 0.5" - we approximate this
        max_rmsd = 10.0  # Maximum acceptable RMSD
        shape_score = 1.0 - (rmsd / max_rmsd)
        
        if shape_score > 0.5:
            pair['shape_score'] = shape_score
            pair['alignment_rmsd'] = rmsd
            filtered_pairs.append(pair)
        
    except Exception as e:
        continue

print(f"\n✓ Filtered: {len(pairs)} → {len(filtered_pairs)} pairs")
print(f"✓ Retention rate: {100*len(filtered_pairs)/len(pairs):.1f}%")

# Save filtered dataset
output_path = 'data/processed/training_pairs_filtered_shape.pkl'
pickle.dump(filtered_pairs, open(output_path, 'wb'))
print(f"✓ Saved to {output_path}")

# Statistics
if filtered_pairs:
    shape_scores = [p['shape_score'] for p in filtered_pairs]
    rmsds = [p['alignment_rmsd'] for p in filtered_pairs]
    print(f"\nShape score: mean={np.mean(shape_scores):.3f}, min={np.min(shape_scores):.3f}")
    print(f"Alignment RMSD: mean={np.mean(rmsds):.3f}, max={np.max(rmsds):.3f}")
