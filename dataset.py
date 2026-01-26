# dataset.py (UPDATED for hybrid format)
import pickle
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
from rdkit import Chem

class GeomPairsDataset(Dataset):
    def __init__(self, pairs_path='data/training_pairs_filtered.pkl'):
        self.pairs = pickle.load(open(pairs_path, 'rb'))
        print(f"Loaded {len(self.pairs)} training pairs")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        
        # Handle both old format (mol) and new format (mol_query, mol_ref)
        if 'mol' in pair:
            # Old format (same molecule)
            mol = pair['mol']
        else:
            # New format (could be different molecules)
            mol = pair['mol_query']  # Use query molecule structure
        
        # Build graph features
        atom_types = torch.tensor([atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=torch.long)
        
        edges = []
        edge_types = []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edges.extend([[i, j], [j, i]])
            bt = bond.GetBondTypeAsDouble()
            edge_types.extend([bt, bt])
        
        edge_index = torch.tensor(edges, dtype=torch.long).t() if edges else torch.zeros((2, 0), dtype=torch.long)
        edge_type = torch.tensor(edge_types, dtype=torch.float) if edge_types else torch.zeros(0)
        
        # Convert coordinates to tensor
        query_coords = torch.tensor(pair['query_coords'], dtype=torch.float32)
        ref_coords = torch.tensor(pair['ref_coords'], dtype=torch.float32)
        
        # Query and reference data
        query_data = Data(
            atom_type=atom_types,
            edge_index=edge_index,
            edge_type=edge_type,
            pos=query_coords
        )
        
        ref_data = Data(
            atom_type=atom_types,
            edge_index=edge_index,
            edge_type=edge_type,
            pos=ref_coords
        )
        
        return query_data, ref_data


def collate_fn(batch):
    """Collate function for DataLoader"""
    queries, refs = zip(*batch)
    return Batch.from_data_list(queries), Batch.from_data_list(refs)
