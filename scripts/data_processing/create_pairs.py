# create_pairs.py (fixed)
import pickle
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import torch

print("Loading molecules...")
molecules = pickle.load(open('data/subset_100.pkl', 'rb'))
print(f"Loaded {len(molecules)} molecules")

pairs = []

for smiles, mol_data in molecules:
    conformers = mol_data['conformers']
    
    if len(conformers) < 2:
        continue
    
    # Use first conformer as reference
    ref_conf = conformers[0]
    
    # Create up to 5 pairs per molecule
    for query_conf in conformers[1:6]:
        # Build mol from SMILES
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)  # Add hydrogens
        
        # Extract coordinates from xyz [atomic_num, x, y, z]
        ref_xyz = np.array(ref_conf['xyz'])
        query_xyz = np.array(query_conf['xyz'])
        
        ref_coords = ref_xyz[:, 1:4]  # Take x,y,z columns
        query_coords = query_xyz[:, 1:4]
        
        # Store pair
        pairs.append({
            'smiles': smiles,
            'ref_coords': torch.FloatTensor(ref_coords),
            'query_coords': torch.FloatTensor(query_coords),
            'mol': mol,
            'ref_energy': ref_conf['totalenergy'],
            'query_energy': query_conf['totalenergy']
        })
        
        if len(pairs) >= 500:
            break
    
    if len(pairs) >= 500:
        break
    
    if len(pairs) % 50 == 0:
        print(f"Created {len(pairs)} pairs...")

print(f"\nTotal pairs: {len(pairs)}")
pickle.dump(pairs, open('data/training_pairs.pkl', 'wb'))
print("Saved to data/training_pairs.pkl")