import pickle
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign, rdShapeHelpers
import numpy as np
from tqdm import tqdm

print("Loading pairs...")
pairs = pickle.load(open('data/processed/training_pairs_unfiltered.pkl', 'rb'))
print(f"Loaded {len(pairs)} pairs")

print("\nFiltering by shape Tanimoto > 0.5...")
filtered_pairs = []

for pair in tqdm(pairs):
    try:
        # Create molecules WITHOUT adding Hs (use original)
        mol_q = Chem.MolFromSmiles(pair['smiles_query'])
        mol_r = Chem.MolFromSmiles(pair['smiles_ref'])
        
        # Create conformers with provided coordinates
        conf_q = Chem.Conformer(len(pair['query_coords']))
        conf_r = Chem.Conformer(len(pair['ref_coords']))
        
        for i, coord in enumerate(pair['query_coords']):
            conf_q.SetAtomPosition(i, tuple(coord))
        
        for i, coord in enumerate(pair['ref_coords']):
            conf_r.SetAtomPosition(i, tuple(coord))
        
        mol_q.AddConformer(conf_q, assignId=True)
        mol_r.AddConformer(conf_r, assignId=True)
        
        # Calculate shape Tanimoto (0-1, higher is better)
        shape_tanimoto = rdShapeHelpers.ShapeTanimotoDist(mol_q, mol_r)
        # ShapeTanimotoDist returns distance, convert to similarity
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
