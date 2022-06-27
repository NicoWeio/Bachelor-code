## TODOs
- MAE → EMD (Wasserstein distance)?
- sum per-sample probabilities instead of most predicted labels
- put it into DSEA
  - https://stackoverflow.com/questions/66374709/adding-custom-weights-to-training-data-in-pytorch
- hyperparameter search?

## Requirements
- plotly
  - for passing interactive plots wandb.ai

## Quirks
- `AttributeError: module 'setuptools._distutils' has no attribute 'version'`
  - `pip install setuptools==59.5.0`

## Literature
- https://raschka-research-group.github.io/coral-pytorch/tutorials/pytorch_lightning/ordinal-corn_cement
- [https://mail.sebastianraschka.com/pdf/slides/2022-02_rework-coral-lightning.pdf](Easy-to-understand slides explaining CORN/CORAL)
