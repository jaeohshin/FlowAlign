# create_pairs_3000.py
import pickle
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import numpy as np
from tqdm import tqdm

print("Loading 20000 molecules...")
molecules = pickle.load(open('data/molecules_20000.pkl', 'rb'))
print(f"Loaded {len(molecules)} molecules")

# Compute fingerprints
print("Computing fingerprints...")
fingerprints = []
for smiles, mol_data in tqdm(molecules):
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    fingerprints.append(fp)

# Find similar pairs (paper constraints)
print("Finding similar pairs [0.55, 0.85]...")
pairs = []

for i in tqdm(range(len(fingerprints))):
    for j in range(i+1, min(i+200, len(fingerprints))):  # Check next 200
        sim = DataStructs.TanimotoSimilarity(fingerprints[i], fingerprints[j])
        
        if 0.55 <= sim <= 0.85:
            pairs.append((i, j, sim))
            
        if len(pairs) >= 5000:
            break
    if len(pairs) >= 5000:
        break

print(f"Found {len(pairs)} candidate pairs")

# Create training pairs (same as before)
# ... rest of processing ...