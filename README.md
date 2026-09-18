# RealWaste project proposal

This archive supports the proposal **RealWaste: Image-Based Waste Material Classification**.

Team:

- Mohammed Moallem, ID 202283440
- Youssef Abdelaziz, ID s202278640

## What the model will do

Input: one color photograph of waste.

Output: probabilities for nine material classes and the most likely class. For example, a bottle photograph may produce `Plastic: 0.94`.

## Files

- `RealWaste_Proposal.pdf`: final three-page proposal.
- `analysis.py`: dataset inspection, plots, duplicate check, and fixed split creation.
- `requirements.txt`: Python dependencies.
- `outputs/class_distribution.png`: proposal distribution plot.
- `outputs/sample_structure.png`: inspected UCI file entries.

## Dataset

Download RealWaste from:

https://archive.ics.uci.edu/dataset/908/realwaste

Extract the archive. The class folders should be inside a directory named `RealWaste`.

UCI lists CC BY 4.0, while the creators' GitHub repository states CC BY-NC-SA 4.0. This project follows the stricter CC BY-NC-SA 4.0 terms.

## Run the inspection

```bash
python -m pip install -r requirements.txt
python analysis.py --data-dir /path/to/realwaste-main/RealWaste
```

The script creates:

- `outputs/image_inventory.csv`
- `outputs/class_distribution.png`
- `outputs/sample_grid.png`
- `outputs/split_manifest.csv`
- `outputs/exact_duplicates.csv`, only if exact duplicates are found

The split is fixed before training: 70% training, 15% validation, and 15% testing, using random seed 42 and preserving class proportions.

## Planned evaluation

The single validation metric is macro F1. It evaluates every material class separately and gives all nine classes equal importance.

