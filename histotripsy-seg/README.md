# Development of a Convolutional Neural Network to Segment Ultrasound Images of Histotripsy Ablation

Deep-learning pipeline (PyTorch U-Net) that segments the ablation zone in B-mode ultrasound
images acquired during/after histotripsy treatment. Histotripsy produces a transient bubble cloud
(hyperechoic) and leaves a homogenized, typically hypoechoic lesion; automatic segmentation enables
fast, operator-independent estimation of ablation size and position.

## Structure
```
configs/default.yaml             hyperparameters
src/model.py                     U-Net
src/dataset.py                   dataset + augmentation
src/losses.py                    Dice+BCE loss, Dice/IoU metrics
src/train.py                     training loop
src/evaluate.py                  Dice / IoU / area-error evaluation
src/predict.py                   inference + overlay export
scripts/make_synthetic_data.py   synthetic data for smoke tests
tests/                           unit tests
```

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Data format
```
data/<dataset>/images/xxx.png   # grayscale B-mode frames
data/<dataset>/masks/xxx.png    # binary masks (nonzero = ablation zone), same filename
```
For real data, split by subject/treatment session (not by frame) to avoid leakage.

## Quick start (synthetic data)
```bash
python scripts/make_synthetic_data.py --out data/synthetic --n 200
python -m src.train --config configs/default.yaml
python -m src.evaluate --ckpt runs/exp1/best.pt --data_dir data/synthetic
python -m src.predict  --ckpt runs/exp1/best.pt --input data/synthetic/images --out preds
pytest -q
```

## Notes
- Ground-truth masks should come from expert annotation (ideally validated against histology/MRI/gross pathology).
- Research code only; not for clinical use.
