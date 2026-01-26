#!/bin/bash
# reorganize.sh - Run this to clean up your directory

echo "Reorganizing DiffAlign project structure..."

# Create new directories
mkdir -p data/{raw,processed,checkpoints}
mkdir -p models
mkdir -p scripts/{data_processing,training,evaluation}
mkdir -p notebooks
mkdir -p experiments/{runs,results}
mkdir -p docs
mkdir -p config

# Move data files
echo "Moving data files..."
mv data/molecules_20k.pkl data/processed/ 2>/dev/null
mv data/training_pairs*.pkl data/processed/ 2>/dev/null
mv data/*.h5 data/raw/ 2>/dev/null

# Move model weights
echo "Moving model weights..."
mv trained_model*.pt models/ 2>/dev/null
mv checkpoints/*.pt models/ 2>/dev/null

# Move processing scripts
echo "Moving scripts..."
mv create_*.py scripts/data_processing/ 2>/dev/null
mv extract_*.py scripts/data_processing/ 2>/dev/null
mv filter_*.py scripts/data_processing/ 2>/dev/null
mv process_*.py scripts/data_processing/ 2>/dev/null
mv make_subset.py scripts/data_processing/ 2>/dev/null

# Move training scripts (we'll unify these later)
mv train*.py scripts/training/ 2>/dev/null

# Move evaluation scripts
mv test_model.py scripts/evaluation/ 2>/dev/null
mv evaluate_*.py scripts/evaluation/ 2>/dev/null

# Move notebooks
mv *.ipynb notebooks/ 2>/dev/null

# Move documentation
mv daily_summary.txt docs/ 2>/dev/null
mv summary.py docs/ 2>/dev/null

# Move TensorBoard runs
mv runs experiments/ 2>/dev/null

# Move logs
mkdir -p experiments/logs
mv *.log experiments/logs/ 2>/dev/null

# Keep diffalign/ package as is
# Keep README, LICENSE, env.yml at root

# Clean up old checkpoints directory if empty
rmdir checkpoints 2>/dev/null

echo "Done! Your project is now organized."
echo ""
echo "Next steps:"
echo "1. Review the new structure"
echo "2. Update import paths in your scripts"
echo "3. Create a unified training script"
