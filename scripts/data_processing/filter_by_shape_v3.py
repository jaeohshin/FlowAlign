import pickle
from rdkit import Chem
from rdkit.Chem import rdShapeHelpers
import numpy as np
from tqdm import tqdm

print("Loading pairs...")
pairs = pickle.load(open('data/processed/training_pairs_unfiltered.pkl', 'rb'))
print(f"Loaded {len(pairs)} pairs")

print("\nFiltering by shape Tanimoto > 0.5...")
filtered_pairs = []

for pair in tqdm(pairs):
    try:
        # Use the already-stored molecules with Hs
        mol_q = pair['mol_query']
        mol_r = pair['mol_ref']
        
        # Verify coordinates match molecule
        if mol_q.GetNumAtoms() != len(pair['query_coords']):
            continue
        if mol_r.GetNumAtoms() != len(pair['ref_coords']):
            continue
        
        # Create conformers
        conf_q = Chem.Conformer(mol_q.GetNumAtoms())
        conf_r = Chem.Conformer(mol_r.GetNumAtoms())
        
        for i, coord in enumerate(pair['query_coords']):
            conf_q.SetAtomPosition(i, tuple(coord))
        
        for i, coord in enumerate(pair['ref_coords']):
            conf_r.SetAtomPosition(i, tuple(coord))
        
        mol_q.RemoveAllConformers()
        mol_r.RemoveAllConformers()
        mol_q.AddConformer(conf_q, assignId=True)
        mol_r.AddConformer(conf_r, assignId=True)
        
        # Calculate shape Tanimoto
        shape_tanimoto = rdShapeHelpers.ShapeTanimotoDist(mol_q, mol_r)
        shape_similarity = 1.0 - shape_tanimoto
        
        if shape_similarity > 0.5:
            pair['shape_score'] = shape_similarity
            filtered_pairs.append(pair)
        
    except Exception as e:
        continue

print(f"\n✓ Filtered: {len(pairs)} → {len(filtered_pairs)} pairs")
print(f"✓ Retention rate: {100*len(filtered_pairs)/len(pairs):.1f}%")

output_path = 'data/processed/training_pairs_filtered_shape.pkl'
pickle.dump(filtered_pairs, open(output_path, 'wb'))
print(f"✓ Saved to {output_path}")

if filtered_pairs:
    scores = [p['shape_score'] for p in filtered_pairs]
    print(f"\nShape scores: mean={np.mean(scores):.3f}, min={np.min(scores):.3f}, max={np.max(scores):.3f}")
