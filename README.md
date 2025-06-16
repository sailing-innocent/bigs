# BIGS: Bayesian Inference Gaussian Splatting

This is the official implementaiton for BIGS(Bayesian Inference Gaussian Splatting)

- [Development Guide](./doc/dev.md)
- [Data Preparation](./doc/data.md)
- [Research Tasks](./doc/task.md)

## Scripts

## Vanilla Train and Render as Preprocess

- v_arguments.py/v_full_eval.py/v_metrics.py/v_render.py/v_train.py: the same script for vanilla gaussian splatting training as https://github.com/graphdeco-inria/gaussian-splatting
- assume we place our MipNeRF360 dataset into `data/datasets/mip360` and we would like to train `bicycle` scene as follows:
- simple train `uv run v_train.py -s data/datasets/mip360/bicycle`
- simple render `uv run v_render.py -s data/datasets/mip360/bicycle -m data/mid/reprod/bicycle`
- simple metrics `uv run v_metrics.py -m data/mid/reprod/bicycle`
