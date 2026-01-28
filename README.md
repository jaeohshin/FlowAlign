# FlowAlign

Flow matching implementation based on DiffAlign.

## Directory Structure
```
FlowAlign/
├── data/
│   ├── drugs_crude.msgpack          # GEOM-Drugs dataset (64GB)
│   └── processed/
│       ├── training_pairs_unfiltered.pkl  # Pairs with 2D similarity filter
│       └── training_pairs_filtered_shape.pkl  # + 3D shape filter
├── models/
│   ├── common.py                    # Shared model components
│   └── epsnet/
│       ├── diffusion.py            # DiffAlign (original)
│       └── flow.py                 # FlowAlign (modified)
├── utils/
│   ├── chem.py                     # Molecular graph utilities
│   └── datasets.py                 # PyTorch datasets
├── scripts/
│   └── data_processing/
│       ├── process_65k_rolling.py  # Generate pairs (2D filter)
│       └── filter_by_shape_v3.py   # 3D shape score filter
├── checkpoints_24k/                # Trained model weights
├── disco/ABL1/                     # Evaluation benchmark
├── train_24k.py                    # Training script
└── eval_24k_abl1.py               # Evaluation script
```

## Original Work
This project is derived from [DiffAlign](https://github.com/kim-iljung/DiffAlign) 
by Iljung Kim, Keehyoung Joo, and Yung-Kyun Noh.

**Original Paper:** 
Kim, I., Joo, K., & Noh, Y. (2024). DiffAlign: Diffusion-Based Molecular Alignment 
with Pocket-Aware Guidance. MLSB Workshop.

## Modifications
- Converted diffusion model to flow matching
- Simplified sampling (10 steps vs 50-100)
- Modified training objective for velocity prediction

## License
MIT License - See LICENSE file (includes original DiffAlign copyright)


### 3. **Keep the original LICENSE file**
Copyright (c) 2024 Iljung Kim (original DiffAlign)
Copyright (c) 2026 Jaeoh Shin (flow matching modifications)

