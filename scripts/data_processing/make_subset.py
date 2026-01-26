# make_subset.py
import msgpack
import pickle

print("Creating subset from msgpack...")

molecules = []
count = 0

with open('data/drugs_crude.msgpack', 'rb') as f:
    unpacker = msgpack.Unpacker(f, raw=False, max_buffer_size=2147483647)
    
    try:
        for item in unpacker:
            if isinstance(item, dict):
                # This is the full dict with all molecules
                print(f"Found dict with {len(item)} molecules")
                # Take first 100
                for smiles, mol_data in list(item.items())[:100]:
                    if 'conformers' in mol_data and len(mol_data['conformers']) >= 2:
                        molecules.append((smiles, mol_data))
                        count += 1
                break
    except Exception as e:
        print(f"Error: {e}")

print(f"\nSaving {len(molecules)} molecules...")
pickle.dump(molecules, open('data/subset_100.pkl', 'wb'))
print("Done!")
