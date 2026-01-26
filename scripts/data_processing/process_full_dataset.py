# process_full_dataset_streaming.py (MEMORY EFFICIENT)
import msgpack
import pickle
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import numpy as np
from tqdm import tqdm
import os
from datetime import datetime

print("="*60)
print(f"Starting STREAMING dataset processing: {datetime.now()}")
print("="*60)

OUTPUT_DIR = 'data/processed'
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_PAIRS = 5000
MIN_SIMILARITY = 0.55
MAX_SIMILARITY = 0.85

# STEP 1: Extract subset of molecules WITHOUT loading everything
print("\n[1/4] Extracting molecules (STREAMING - low memory)...")

molecules = []
fingerprints = []

with open('data/drugs_crude.msgpack', 'rb') as f:
    unpacker = msgpack.Unpacker(f, raw=False, max_buffer_size=2**31-1)
    
    try:
        for item in unpacker:
            if isinstance(item, dict):
                # Process molecules one by one
                count = 0
                for smiles, mol_data in item.items():
                    if count >= 2000:  # Only take 2000 molecules
                        break
                    
                    try:
                        mol = Chem.MolFromSmiles(smiles)
                        if mol is None:
                            continue
                        
                        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
                        molecules.append((smiles, mol_data))
                        fingerprints.append(fp)
                        count += 1
                        
                        if count % 100 == 0:
                            print(f"Loaded {count} molecules...", end='\r')
                    except:
                        continue
                
                break  # Only process first batch
    except Exception as e:
        print(f"Extraction stopped: {e}")

print(f"\n✓ Extracted {len(molecules)} molecules")

# STEP 2: Find similar pairs
print("\n[2/4] Finding similar pairs...")
candidate_pairs = []

for i in tqdm(range(len(fingerprints))):
    if len(candidate_pairs) >= TARGET_PAIRS * 2:
        break
    
    for j in range(i+1, min(i+50, len(fingerprints))):
        similarity = DataStructs.TanimotoSimilarity(fingerprints[i], fingerprints[j])
        
        if MIN_SIMILARITY <= similarity <= MAX_SIMILARITY:
            candidate_pairs.append((i, j, similarity))
        
        if len(candidate_pairs) >= TARGET_PAIRS * 2:
            break

print(f"✓ Found {len(candidate_pairs)} candidate pairs")

# STEP 3: Create training pairs
print("\n[3/4] Creating training pairs...")
final_pairs = []

for i, j, sim in tqdm(candidate_pairs):
    if len(final_pairs) >= TARGET_PAIRS:
        break
    
    try:
        smiles_i, data_i = molecules[i]
        smiles_j, data_j = molecules[j]
        
        if 'conformers' not in data_i or 'conformers' not in data_j:
            continue
        
        conf_i = data_i['conformers'][0]
        conf_j = data_j['conformers'][0]
        
        coords_i = np.array(conf_i['xyz'])[:, 1:4]
        coords_j = np.array(conf_j['xyz'])[:, 1:4]
        
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
            'similarity': sim
        }
        
        final_pairs.append(pair)
        
        # Checkpoint every 500
        if len(final_pairs) % 500 == 0:
            checkpoint_path = f'{OUTPUT_DIR}/checkpoint_{len(final_pairs)}.pkl'
            pickle.dump(final_pairs, open(checkpoint_path, 'wb'))
            print(f"\n✓ Saved checkpoint: {len(final_pairs)} pairs")
    
    except:
        continue

# STEP 4: Save
print(f"\n[4/4] Saving final dataset...")
final_path = f'{OUTPUT_DIR}/training_pairs_{len(final_pairs)}.pkl'
pickle.dump(final_pairs, open(final_path, 'wb'))

print("\n" + "="*60)
print(f"✓ COMPLETE: {datetime.now()}")
print(f"✓ Created {len(final_pairs)} pairs")
print(f"✓ File: {final_path}")
print("="*60)