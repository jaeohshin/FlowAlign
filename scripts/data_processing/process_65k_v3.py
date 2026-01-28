cat > scripts/data_processing/process_65k_chunks.py << 'EOF'
import msgpack
import pickle
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import numpy as np
from tqdm import tqdm
import os

OUTPUT_DIR = 'data/processed'
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_PAIRS = 100000  # Generate extra, filter later
MIN_SIMILARITY = 0.55
MAX_SIMILARITY = 0.85
CHUNK_SIZE = 10000  # Process 10K molecules at a time

print("Processing in chunks to avoid OOM...")

all_pairs = []
total_molecules_processed = 0

with open('data/drugs_crude.msgpack', 'rb') as f:
    unpacker = msgpack.Unpacker(f, raw=False, max_buffer_size=2**31-1)
    
    molecules_chunk = []
    fingerprints_chunk = []
    
    for batch_idx, item in enumerate(unpacker):
        if not isinstance(item, dict):
            continue
        
        print(f"\nBatch {batch_idx+1}: {len(item)} entries")
        
        for smiles, mol_data in item.items():
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue
                
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
                molecules_chunk.append((smiles, mol_data))
                fingerprints_chunk.append(fp)
                
                # Process chunk when it reaches CHUNK_SIZE
                if len(molecules_chunk) >= CHUNK_SIZE:
                    print(f"Processing chunk of {len(molecules_chunk)} molecules...")
                    
                    # Find pairs within this chunk
                    for i in tqdm(range(len(fingerprints_chunk))):
                        for j in range(i+1, len(fingerprints_chunk)):
                            similarity = DataStructs.TanimotoSimilarity(fingerprints_chunk[i], fingerprints_chunk[j])
                            
                            if MIN_SIMILARITY <= similarity <= MAX_SIMILARITY:
                                try:
                                    smiles_i, data_i = molecules_chunk[i]
                                    smiles_j, data_j = molecules_chunk[j]
                                    
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
                                        'similarity': sim,
                                        'type': 'similar_pair'
                                    }
                                    
                                    all_pairs.append(pair)
                                except:
                                    continue
                    
                    total_molecules_processed += len(molecules_chunk)
                    print(f"Total pairs so far: {len(all_pairs)}")
                    
                    # Save checkpoint
                    if len(all_pairs) % 10000 == 0 and len(all_pairs) > 0:
                        checkpoint_path = f'{OUTPUT_DIR}/pairs_checkpoint_{len(all_pairs)}.pkl'
                        pickle.dump(all_pairs, open(checkpoint_path, 'wb'))
                        print(f"Saved checkpoint: {len(all_pairs)} pairs")
                    
                    # Clear chunk to free memory
                    molecules_chunk = []
                    fingerprints_chunk = []
                    
                    # Stop if we have enough
                    if len(all_pairs) >= TARGET_PAIRS:
                        print(f"\nReached {TARGET_PAIRS} pairs!")
                        break
                        
            except:
                continue
        
        if len(all_pairs) >= TARGET_PAIRS:
            break

# Save final
print(f"\nSaving final dataset...")
final_path = f'{OUTPUT_DIR}/training_pairs_unfiltered.pkl'
pickle.dump(all_pairs, open(final_path, 'wb'))

print(f"\n✓ Created {len(all_pairs)} pairs (unfiltered)")
print(f"✓ Saved to {final_path}")
print("Next: Filter by 3D shape score")
EOF

python scripts/data_processing/process_65k_chunks.py