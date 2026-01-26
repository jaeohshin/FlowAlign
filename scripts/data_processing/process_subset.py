# process_subset.py
import msgpack
import pickle
from rdkit import Chem

print("Reading msgpack in streaming mode...")

pairs = []
max_molecules = 1000  # Process only first 1000 molecules

with open('data/drugs_crude.msgpack', 'rb') as f:
    unpacker = msgpack.Unpacker(f, raw=False, max_buffer_size=2**31-1)
    
    mol_count = 0
    for item in unpacker:
        if mol_count >= max_molecules:
            break
        
        # Process molecule
        # Create pairs from conformers
        mol_count += 1
        
        if mol_count % 100 == 0:
            print(f"Processed {mol_count} molecules...")

print(f"Created {len(pairs)} pairs")
pickle.dump(pairs, open('data/subset_pairs.pkl', 'wb'))
