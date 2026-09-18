# Autoencoder depth study — 2000 epochs

Open `index.html` for the report, curves and final-checkpoint reconstruction videos.

## Run the notebooks

Python 3.10 or later. Install PyTorch appropriate for your CUDA hardware, then:

```sh
python -m pip install -r requirements.txt
python download_data.py
jupyter lab
```

Start Jupyter from this folder. Select one of the five notebooks and Run All.
Default mode loads the published final checkpoint, without retraining. Set CHECKPOINT_KIND="best" to inspect the best validation checkpoint.
The full dataset is available in the v1.0 GitHub Release. Run `python download_data.py` to download, verify and extract it into data/balls_128/. This is the default notebook data path. Alternatively set AE_DATASET_ROOT to your existing dataset. Expected structure:

```
DATASET_ROOT/
  train/0/{actions.npy,frames.npy,metadata.json}
  train/1/...
  test/0/...
```

Original split: 257 train / 193 test clips. Frames: uint8 [128,128,128,3] in time/height/width/channel order. Metadata uses case.instruction. Numeric sample folders are sorted numerically by BallMovingDataset. Dataset and DataLoader construction are included in source/dataloader.py and each notebook.

Set RUN_TRAINING=True for a fresh 2000-epoch run. Outputs go to runs/fresh_training/; published results remain unchanged. Repeating training overwrites that runs directory's same-named outputs. To reserve GPU0, launch with `CUDA_VISIBLE_DEVICES=1 jupyter lab`; the notebooks do not select a physical GPU. CPU execution is possible but full video training is very slow and memory-intensive.

## Contents

- source/: downloaded dataset and display utility definitions.
- checkpoint/: five best and five final state dictionaries.
- results/: full loss JSON, plots, final-weight reconstruction HTML.
- collection_summary/: combined curves and final/best metrics.
- Original server notebooks remain in the local original_notebooks/ backup; portable notebooks are published at repository root.

Model definitions and optimization logic are unchanged. Portability changes only cover paths, device setup, output directories, CPU-safe RNG checkpointing and an inference/training switch.

Validation: original train/test counts and one actual batch read successfully; all 10 checkpoint state dictionaries loaded strictly and passed finite CPU forward checks on 32³ inputs. Full 2000-epoch training was not rerun for this packaging change. Published reconstruction HTML uses train samples 100 and 200, not the held-out validation set.
