# extract_more_molecules.py
import msgpack
import pickle
from rdkit import Chem
from rdkit.Chem import AllChem

print("Extracting 2000-3000 molecules from msgpack...")

molecules = []
target = 20000

with open('data/drugs_crude.msgpack', 'rb') as f:
    unpacker = msgpack.Unpacker(f, raw=False, max_buffer_size=2**31-1)
    
    batch_count = 0
    try:
        for item in unpacker:
            if isinstance(item, dict):
                batch_count += 1
                print(f"\nProcessing batch {batch_count}...")
                
                for smiles, mol_data in item.items():
                    if len(molecules) >= target:
                        break
                    
                    try:
                        mol = Chem.MolFromSmiles(smiles)
                        if mol and 'conformers' in mol_data:
                            molecules.append((smiles, mol_data))
                            
                            if len(molecules) % 100 == 0:
                                print(f"  Loaded {len(molecules)} molecules...", end='\r')
                    except:
                        pass
                
                if len(molecules) >= target:
                    break
                
                # If batch too small, continue to next batch
                print(f"  Batch {batch_count} done: {len(molecules)} total molecules")
    
    except Exception as e:
        print(f"\nStopped: {e}")

print(f"\n✓ Extracted {len(molecules)} molecules")

# Save for reuse
pickle.dump(molecules, open('data/molecules_3000.pkl', 'wb'))
print("✓ Saved to data/molecules_3000.pkl")