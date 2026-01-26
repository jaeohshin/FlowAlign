# filter_pairs.py
import pickle
from rdkit import Chem

pairs = pickle.load(open('data/training_pairs_hybrid.pkl', 'rb'))
print(f"Original: {len(pairs)} pairs")

# Keep only pairs with matching atom counts
filtered = []
for pair in pairs:
    mol_q = pair['mol_query']
    mol_r = pair['mol_ref']
    
    if mol_q.GetNumAtoms() == mol_r.GetNumAtoms():
        filtered.append(pair)

print(f"Filtered: {len(filtered)} pairs (matching atom counts)")

# Breakdown
conformer = sum(1 for p in filtered if p['type'] == 'conformer')
similar = sum(1 for p in filtered if p['type'] == 'similar_molecule')
print(f"  - Conformer pairs: {conformer}")
print(f"  - Similar molecules: {similar}")

pickle.dump(filtered, open('data/training_pairs_filtered.pkl', 'wb'))
print("✓ Saved to data/training_pairs_filtered.pkl")
