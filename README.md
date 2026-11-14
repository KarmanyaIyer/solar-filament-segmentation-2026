# Solar Filament Segmentation Challenge 2026: consensus U-Nets + YOLO11-seg instance fusion

Solution code for the Kaggle competition
[Solar Filament Segmentation Challenge 2026](https://www.kaggle.com/competitions/filament-segmentation-2026).
The method is described in `report/report.pdf`; `notebooks/demo.ipynb` reproduces the submission end to end.

## Method in one paragraph
An ensemble of U-Nets is trained on a *soft consensus* target (per pixel, the fraction of annotators that
marked it) and predicts a filament probability map: ResNet-34 and EfficientNet-B3 encoders on images
downsampled to 1024x1024, plus a ResNet-34 on the native 2048x2048 images, 5 folds each. Two YOLO11-seg
detectors (YOLO11s and YOLO11m), trained with every annotator's label set as a separate sample, propose
filament instances with a confidence that behaves like an agreement score; their predictions are merged by mask
IoU with the confidence averaged over detectors. Fusion: each merged instance is grown by a few pixels and
intersected with the U-Net foreground; instances are painted in confidence order (so masks are disjoint) and
dropped when small or weakly supported by the U-Net. Masks are written as COCO RLE.

## Layout
| path | content |
|---|---|
| `src/common.py` | data loading, RLE helpers, the competition metric (panoptic quality), fold split |
| `src/prep.py` | builds the image cache (1024px, or 2048px with `RES=2048`) with soft consensus targets |
| `src/train.py` | U-Net training (one fold per call) |
| `src/infer.py` | U-Net inference with flip TTA (out-of-fold and test) |
| `src/yolo_data.py`, `src/yolo_run.py` | YOLO11-seg dataset export, training and prediction dump |
| `src/yolo_ens.py` | merges the instance predictions of several detectors |
| `src/fuse.py` | fusion of YOLO instances and U-Net probabilities (+ parameter grid search on OOF) |
| `src/postproc.py`, `src/tune.py` | U-Net-only instance extraction and its parameter search (baseline) |
| `src/evaluate.py` | rubric-style evaluation: PQ, IoU/Dice distributions, 1:n / n:1 relations, bootstrap CI |
| `src/make_sub.py` | command-line submission writer and validator |
| `configs/final.json` | ensemble members and fusion parameters of the final submission |
| `weights/` | trained weights: U-Nets (fp16, 5 folds per member); YOLO models trained on all but 30 observations (`*_all.pt`, the 30 select the best epoch) and without fold 0 / fold 1 (`*_f0.pt`, `*_f1.pt`, used only for the out-of-fold evaluation) |
| `notebooks/demo.ipynb` | inference, optional retraining, out-of-fold evaluation, examples |
| `report/report.pdf` | 4-page report |

## Quick start
```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
# put the competition data at data/MAGFiLO_1.0_Kaggle_2026 (kaggle competitions download -c filament-segmentation-2026)
jupyter notebook notebooks/demo.ipynb
```
The whole notebook (test inference plus the out-of-fold evaluation) ran in about 20 minutes on an RTX 4060 laptop GPU (CPU works, much slower). Retraining everything
from scratch (set `TRAIN_FROM_SCRATCH = True` in the notebook) takes roughly 11 GPU-hours on an RTX 4060-class GPU.

## Reproducing the parameter selection (command line)
The notebook reproduces the submission from the stored weights. The fusion parameters were chosen on
out-of-fold predictions of folds 0 and 1 with these steps (each fold model only predicts images it never saw):
```bash
# U-Net out-of-fold and test probability maps (one call per member; RES=2048 for the native member)
python src/infer.py --runs "runs/r34_f{fold}" --enc resnet34 --out oof/r34
python src/infer.py --runs "runs/r34_f{fold}" --enc resnet34 --out test_probs/r34 --test
# fold detectors dump their validation-fold predictions, the full-data detectors their test predictions
python src/yolo_run.py --fold 0 --model yolo11s-seg.pt --batch 4 --tag y11s_f0 --ds yolo_ds0   # likewise fold 1, yolo11m-seg (batch 2), fold -1
python src/yolo_ens.py --dirs yolo_pred/y11s_f0/val,yolo_pred/y11m_f0/val --out yolo_pred/ens_f0/val --match 0.3   # likewise f1, all/test
# grid search of the fusion parameters on both folds
python src/fuse.py --oof oof/r34,oof/effb3,oof/r34n --yolo yolo_pred/ens_f0/val,yolo_pred/ens_f1/val \
    --grid '{"yconf":[0.15,0.2,0.25],"thr":[0.35,0.4,0.45],"grow":[2],"min_area":[200],"min_uconf":[0.75,0.8,0.85]}'
# rubric-style evaluation (PQ, IoU/Dice histograms, 1:n / n:1) of the chosen parameters
python src/eval_oof.py --oof oof/r34,oof/effb3,oof/r34n --yolo yolo_pred/ens_f0/val,yolo_pred/ens_f1/val \
    --fuse '{"yconf":0.2,"thr":0.4,"grow":2,"min_area":200,"min_uconf":0.8}'
```
The YOLO fold models keep the epoch with the best validation score on their held-out fold (Ultralytics
`best.pt`), so their out-of-fold numbers are slightly optimistic.

## Data and rules
Only the competition training data, ImageNet-pretrained encoders (torchvision / timm via
segmentation_models_pytorch) and COCO-pretrained YOLO11 weights (Ultralytics) are used. No test image is
used for training or tuning, and no external filament annotations are used.

## License
Copyright (c) 2026 Karmanya Iyer. AGPL-3.0 (required because the instance branch uses Ultralytics YOLO, which is
AGPL-3.0). See `LICENSE`.

Third-party components: Ultralytics YOLO11 (AGPL-3.0), segmentation_models_pytorch (MIT), timm (Apache-2.0),
PyTorch / torchvision (BSD-3-Clause), pycocotools (BSD-2-Clause).
