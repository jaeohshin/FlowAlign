# inspect_conformers.py
import pickle

molecules = pickle.load(open('data/subset_100.pkl', 'rb'))

# Look at first molecule
smiles, mol_data = molecules[0]

print(f"SMILES: {smiles}")
print(f"\nMolecule data keys: {mol_data.keys()}")
print(f"\nNumber of conformers: {len(mol_data['conformers'])}")

# Look at first conformer
first_conf = mol_data['conformers'][0]
print(f"\nConformer keys: {first_conf.keys()}")

# Print some values to understand structure
for key in first_conf.keys():
    val = first_conf[key]
    print(f"\n{key}: type={type(val)}, shape/len={len(val) if hasattr(val, '__len__') else 'N/A'}")
    if isinstance(val, (list, tuple)) and len(val) > 0:
        print(f"  First element: {val[0]}")