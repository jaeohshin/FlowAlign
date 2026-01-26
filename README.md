# DiffAlign-FlowMatching

Flow matching implementation based on DiffAlign.

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
```

### 3. **Keep the original LICENSE file**
Make sure `LICENSE` file includes the original copyright:
```
Copyright (c) 2024 Iljung Kim (original DiffAlign)
Copyright (c) 2026 Jaeoh Shin (flow matching modifications)

MIT License (see `LICENSE`).
