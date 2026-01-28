import msgpack
import pickle
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import numpy as np
from tqdm import tqdm
import os

OUTPUT_DIR = 'data/processed'
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_PAIRS = 100000
MIN_SIMILARITY = 0.55
MAX_SIMILARITY = 0.85

print("Loading molecules and finding pairs...")

all_molecules = []
all_fingerprints = []
all_pairs = []

with open('data/drugs_crude.msgpack', 'rb') as f:
    unpacker = msgpack.Unpacker(f, raw=False, max_buffer_size=2**31-1)
    
    for batch_idx, item in enumerate(unpacker):
        if not isinstance(item, dict):
            continue
        
        print(f"\nBatch {batch_idx+1}: {len(item)} entries")
        
        batch_molecules = []
        batch_fingerprints = []
        
        # Load this batch
        for smiles, mol_data in item.items():
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue
                
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
                batch_molecules.append((smiles, mol_data))
                batch_fingerprints.append(fp)
            except:
                continue
        
        print(f"Loaded {len(batch_molecules)} molecules from this batch")
        
        # Compare new molecules against ALL previous molecules
        print("Comparing with previous molecules...")
        for i in tqdm(range(len(batch_molecules))):
            # Compare with all previous molecules
            for j in range(len(all_fingerprints)):
                similarity = DataStructs.TanimotoSimilarity(batch_fingerprints[i], all_fingerprints[j])
                
                if MIN_SIMILARITY <= similarity <= MAX_SIMILARITY:
                    try:
                        smiles_i, data_i = batch_molecules[i]
                        smiles_j, data_j = all_molecules[j]
                        
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
                            'similarity': similarity,
                            'type': 'similar_pair'
                        }
                        
                        all_pairs.append(pair)
                    except:
                        continue
            
            # Also compare within current batch
            for j in range(i+1, len(batch_fingerprints)):
                similarity = DataStructs.TanimotoSimilarity(batch_fingerprints[i], batch_fingerprints[j])
                
                if MIN_SIMILARITY <= similarity <= MAX_SIMILARITY:
                    try:
                        smiles_i, data_i = batch_molecules[i]
                        smiles_j, data_j = batch_molecules[j]
                        
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
                            'similarity': similarity,
                            'type': 'similar_pair'
                        }
                        
                        all_pairs.append(pair)
                    except:
                        continue
        
        # Add batch to accumulated molecules
        all_molecules.extend(batch_molecules)
        all_fingerprints.extend(batch_fingerprints)
        
        print(f"Total molecules: {len(all_molecules)}, Total pairs: {len(all_pairs)}")
        
        # Save checkpoint
        if len(all_pairs) % 10000 < len(batch_molecules):
            checkpoint_path = f'{OUTPUT_DIR}/pairs_checkpoint_{len(all_pairs)}.pkl'
            pickle.dump(all_pairs, open(checkpoint_path, 'wb'))
            print(f"Saved checkpoint")
        
        # Stop if enough
        if len(all_pairs) >= TARGET_PAIRS:
            break
        
        # Memory limit: stop loading more molecules after 30K
        if len(all_molecules) >= 30000:
            print("Reached 30K molecules, stopping to avoid OOM")
            break

# Save final
print(f"\nSaving...")
final_path = f'{OUTPUT_DIR}/training_pairs_unfiltered.pkl'
pickle.dump(all_pairs, open(final_path, 'wb'))

print(f"\n✓ Created {len(all_pairs)} pairs from {len(all_molecules)} molecules")
print(f"✓ Saved to {final_path}")
