# create_hybrid_pairs.py
import pickle
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import numpy as np
from tqdm import tqdm

print("="*60)
print("Creating HYBRID training pairs")
print("="*60)

# Load molecules
molecules = pickle.load(open('data/molecules_20k.pkl', 'rb'))
print(f"\n✓ Loaded {len(molecules)} molecules")

# Compute fingerprints
print("\nComputing fingerprints...")
fingerprints = []
for smiles, mol_data in tqdm(molecules):
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    fingerprints.append(fp)

# STRATEGY 1: Similar molecules [0.55, 0.85]
print("\n[1/3] Finding similar molecule pairs [0.55, 0.85]...")
similar_pairs_idx = []

for i in tqdm(range(len(fingerprints))):
    for j in range(i+1, min(i+500, len(fingerprints))):
        sim = DataStructs.TanimotoSimilarity(fingerprints[i], fingerprints[j])
        if 0.55 <= sim <= 0.85:
            similar_pairs_idx.append((i, j, sim))

print(f"✓ Found {len(similar_pairs_idx)} similar pairs")

# STRATEGY 2: Conformer pairs
print("\n[2/3] Creating conformer pairs...")
conformer_pairs_idx = []

for idx, (smiles, mol_data) in enumerate(tqdm(molecules)):
    conformers = mol_data.get('conformers', [])
    if len(conformers) < 2:
        continue
    
    # Pair conformer 0 with conformers 1-5
    for conf_idx in range(1, min(6, len(conformers))):
        conformer_pairs_idx.append((idx, conf_idx))
        
        if len(conformer_pairs_idx) >= 5000:
            break
    
    if len(conformer_pairs_idx) >= 5000:
        break

print(f"✓ Found {len(conformer_pairs_idx)} conformer pairs")

# Build training pairs
print("\n[3/3] Building training pairs...")
all_pairs = []

# Process similar molecule pairs
print("  Processing similar molecules...")
for i, j, sim in tqdm(similar_pairs_idx):
    try:
        smiles_i, data_i = molecules[i]
        smiles_j, data_j = molecules[j]
        
        coords_i = np.array(data_i['conformers'][0]['xyz'])[:, 1:4]
        coords_j = np.array(data_j['conformers'][0]['xyz'])[:, 1:4]
        
        if len(coords_i) < 5 or len(coords_j) < 5:
            continue
        if len(coords_i) > 100 or len(coords_j) > 100:
            continue
        
        mol_i = Chem.AddHs(Chem.MolFromSmiles(smiles_i))
        mol_j = Chem.AddHs(Chem.MolFromSmiles(smiles_j))
        
        pair = {
            'smiles_query': smiles_i,
            'smiles_ref': smiles_j,
            'query_coords': coords_i,
            'ref_coords': coords_j,
            'mol_query': mol_i,
            'mol_ref': mol_j,
            'similarity': sim,
            'type': 'similar_molecule'
        }
        all_pairs.append(pair)
    except Exception as e:
        continue

# Process conformer pairs
print("  Processing conformers...")
for mol_idx, conf_idx in tqdm(conformer_pairs_idx):
    try:
        smiles, mol_data = molecules[mol_idx]
        conformers = mol_data['conformers']
        
        coords_ref = np.array(conformers[0]['xyz'])[:, 1:4]
        coords_query = np.array(conformers[conf_idx]['xyz'])[:, 1:4]
        
        if len(coords_ref) < 5 or len(coords_ref) > 100:
            continue
        
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        
        pair = {
            'smiles_query': smiles,
            'smiles_ref': smiles,
            'query_coords': coords_query,
            'ref_coords': coords_ref,
            'mol_query': mol,
            'mol_ref': mol,
            'similarity': 1.0,
            'type': 'conformer'
        }
        all_pairs.append(pair)
    except Exception as e:
        continue

# Summary
print("\n" + "="*60)
print(f"✓ COMPLETE - Created {len(all_pairs)} training pairs")
print(f"  - Similar molecules: {sum(1 for p in all_pairs if p['type']=='similar_molecule')}")
print(f"  - Conformer pairs:   {sum(1 for p in all_pairs if p['type']=='conformer')}")
print("="*60)

# Save
output_path = 'data/training_pairs_hybrid.pkl'
pickle.dump(all_pairs, open(output_path, 'wb'))
print(f"\n✓ Saved to {output_path}")

# Save stats
with open('data/hybrid_stats.txt', 'w') as f:
    f.write(f"Total pairs: {len(all_pairs)}\n")
    f.write(f"Similar molecules: {sum(1 for p in all_pairs if p['type']=='similar_molecule')}\n")
    f.write(f"Conformer pairs: {sum(1 for p in all_pairs if p['type']=='conformer')}\n")