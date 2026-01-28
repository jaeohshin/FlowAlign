# scripts/evaluation/eval_abl1.py

import torch
import numpy as np
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import rdShapeHelpers
import pandas as pd

# Your existing utilities
from diffalign.utils.chem import (
    mol_to_graph_data_obj,
    set_rdmol_positions, 
    get_best_rmsd
)
from diffalign.models.epsnet.flow_matching import DiffAlign

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================================
# PART A: Load & Convert
# ============================================================================

def load_ligand_from_pdb(pdb_path):
    """Load ligand from PDB and infer bonds"""
    try:
        mol = Chem.MolFromPDBFile(str(pdb_path), removeHs=False)
        if mol is None:
            mol = Chem.MolFromPDBFile(str(pdb_path), removeHs=True)
        
        # CRITICAL: PDB files often don't have bond info
        # Infer bonds from 3D coordinates
        if mol is not None and mol.GetNumBonds() == 0:
            print(f"    No bonds in PDB, inferring from coordinates...")
            from rdkit.Chem import AllChem
            # Determine connectivity from 3D structure
            mol = Chem.Mol(mol)  # Make a copy
            AllChem.ConnectMol(mol)  # Infer bonds
            # Or use:
            # mol = Chem.MolFromPDBFile(str(pdb_path), sanitize=False, removeHs=False)
            # Chem.SanitizeMol(mol)
        
        return mol
    except Exception as e:
        print(f"    Failed to load {pdb_path}: {e}")
        return None

def mol_to_flowalign_input(mol):
    """Convert RDKit mol to your DiffAlign graph format"""
    try:
        graph_data = mol_to_graph_data_obj(mol)
        
        if graph_data is None:
            return None
        
        # Your DiffAlign expects these fields
        return {
            'atom_type': graph_data.atom_type,
            'edge_index': graph_data.edge_index,
            'edge_type': graph_data.edge_type,
            'pos': graph_data.pos
        }
    except Exception as e:
        print(f"    Graph conversion failed: {e}")
        return None

# ============================================================================
# PART B: Metrics
# ============================================================================

def calculate_tanimoto_combo(mol1, mol2):
    """Shape similarity (simplified - just shape, no pharmacophore)"""
    try:
        score = rdShapeHelpers.ShapeTanimotoDist(mol1, mol2)
        return score
    except:
        return 0.0

# ============================================================================
# PART C: Run DiffAlign
# ============================================================================

# Add debug prints in run_flowalign function:

def run_flowalign(model, query_mol, ref_mol, n_samples=30):
    """Generate aligned poses using your DiffAlign model"""
    from torch_geometric.data import Data, Batch
    
    # Convert to graph format
    query_graph = mol_to_flowalign_input(query_mol)
    ref_graph = mol_to_flowalign_input(ref_mol)
    
    # Check if conversion succeeded
    if query_graph is None or ref_graph is None:
        return None
    
    # Create PyTorch Geometric Data objects
    query_data = Data(
        atom_type=query_graph['atom_type'],
        edge_index=query_graph['edge_index'],
        edge_type=query_graph['edge_type'],
        pos=query_graph['pos']
    )
    ref_data = Data(
        atom_type=ref_graph['atom_type'],
        edge_index=ref_graph['edge_index'],
        edge_type=ref_graph['edge_type'],
        pos=ref_graph['pos']
    )
    
    # Create batches
    query_batch = Batch.from_data_list([query_data]).to(DEVICE)
    ref_batch = Batch.from_data_list([ref_data]).to(DEVICE)
    
    # Sample
    with torch.no_grad():
        try:
            all_samples = []
            for _ in range(n_samples):
                # Returns (coordinates, something_else)
                coords, _ = model.FlowMatching_Sampling_UFF(
                    query_batch=query_batch,
                    reference_batch=ref_batch,
                    num_steps=50,
                    cfg_scale=5.0,
                    query_mols=[query_mol],
                    pocket_mols=None,
                    uff_guidance_scale=1.0,
                    debug_log=False
                )
                coords_cpu = coords.cpu().numpy()
                all_samples.append(coords_cpu)
            
            # Convert back to RDKit mols
            samples = []
            for coords in all_samples:
                mol_copy = set_rdmol_positions(query_mol, coords)
                samples.append(mol_copy)
            
            return samples
            
        except Exception as e:
            print(f"    Sampling failed: {e}")
            import traceback
            traceback.print_exc()
            return None

# ============================================================================
# PART D: Evaluate one case
# ============================================================================

def evaluate_one_case(query_id, ref_id, data_dir, model):
    """Run one cross-docking test"""
    try:
        # Load structures
        query_lig = load_ligand_from_pdb(data_dir / f"{query_id}_LIG.pdb")
        ref_lig = load_ligand_from_pdb(data_dir / f"{ref_id}_LIG.pdb")
        crystal_lig = load_ligand_from_pdb(data_dir / f"{query_id}_LIG.pdb")
        
        # Check if loading succeeded
        if query_lig is None or ref_lig is None or crystal_lig is None:
            return None
        
        # Generate samples
        samples = run_flowalign(model, query_lig, ref_lig, n_samples=30)
        
        # Check if sampling succeeded
        if samples is None or len(samples) == 0:
            return None
        
        # Select Top-1 by TanimotoCombo
        scores = [calculate_tanimoto_combo(s, ref_lig) for s in samples]
        best_idx = np.argmax(scores)
        best_sample = samples[best_idx]
        
        # Calculate RMSD using YOUR utility function
        rmsd = get_best_rmsd(best_sample, crystal_lig)
        
        return {
            'query_id': query_id,
            'ref_id': ref_id,
            'rmsd': rmsd,
            'tanimoto': scores[best_idx],
            'success_1A': rmsd < 1.0,
            'success_2A': rmsd < 2.0,
            'success_3A': rmsd < 3.0
        }
    except Exception as e:
        print(f"    Exception: {e}")
        return None

# ============================================================================
# PART E: Main benchmark
# ============================================================================

def run_benchmark(model_path, data_dir, n_structures=10):
    """Run ABL1 benchmark"""
    
    print(f"Loading model from {model_path}")
    
    # Load model
    checkpoint = torch.load(model_path)
    model = DiffAlign()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model = model.to(DEVICE)
    
    print(f"Model loaded on {DEVICE}")
    
    # Get PDB IDs
    data_dir = Path(data_dir)
    all_files = sorted(data_dir.glob("*_LIG.pdb"))
    pdb_ids = [f.stem.split('_')[0] for f in all_files][:n_structures]
    
    print(f"\nTesting {len(pdb_ids)} ABL1 structures")
    print(f"PDB IDs: {pdb_ids}\n")
    
    # Run all pairs
    results = []
    total = len(pdb_ids) * (len(pdb_ids) - 1)
    
    for ref_id in pdb_ids:
        for query_id in pdb_ids:
            if query_id == ref_id:
                continue
            
            done = len(results)
            print(f"[{done+1}/{total}] {query_id} → {ref_id}", end=" ... ")
            
            try:
                result = evaluate_one_case(query_id, ref_id, data_dir, model)
                if result is not None:
                    results.append(result)
                    print(f"RMSD: {result['rmsd']:.2f} Å")
                else:
                    print(f"SKIPPED")
            except Exception as e:
                print(f"ERROR: {e}")
    
    # Save results
    df = pd.DataFrame(results)
    output_file = "abl1_benchmark_results.csv"
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")
    
    # Summary
    print("\n" + "="*60)
    print("ABL1 BENCHMARK RESULTS")
    print("="*60)
    
    if len(df) == 0:
        print("ERROR: No successful test cases!")
        return df
    
    print(f"Total successful cases: {len(df)} / {total}")
    print(f"RMSD < 1Å: {df['success_1A'].mean()*100:.1f}%")
    print(f"RMSD < 2Å: {df['success_2A'].mean()*100:.1f}%")
    print(f"RMSD < 3Å: {df['success_3A'].mean()*100:.1f}%")
    print(f"Mean RMSD: {df['rmsd'].mean():.2f} ± {df['rmsd'].std():.2f} Å")
    print("="*60)
    
    # Compare with paper
    print("\nDiffAlign paper (all 4,035 complexes):")
    print("RMSD < 1Å: 6.0%")
    print("RMSD < 2Å: 18.4%")
    print("RMSD < 3Å: 27.9%")
    
    return df

# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    results = run_benchmark(
        model_path="/data/work/FlowAlign/checkpoints/best_model.pt",
        data_dir="/data/work/FlowAlign/disco/ABL1/PDB_Structures",
        n_structures=10
    )
    