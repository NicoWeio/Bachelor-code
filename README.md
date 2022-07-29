## Structure
- [`a_data_selection.py`](a_data_selection.py)
  - reads the full MC data
  - drops some rows
  - replaces NaNs with an extreme value
  - selects columns according to [`a_data_selection_features.csv`](a_data_selection_features.csv)
    - 90% reduction in file size
    - → subsequent analysis is faster
  - outputs to [`build_large/data.csv`](build_large/data.csv)
- [`b_prepare_data.py`](b_prepare_data.py)
  - drops energies outside of a certain range
  - discretizes the energies
  - applies `StandardScaler`
  - does not write to disk; instead, it's intended to be invoked by other Python files
- (TODO)
  - the following files will be renamed in the future…

## TODOs
- MAE → EMD (Wasserstein distance)?
- hyperparameter search?

## Requirements
- plotly
  - for passing interactive plots wandb.ai

## Quirks (& fixes)
- `AttributeError: module 'setuptools._distutils' has no attribute 'version'`
  - `pip install setuptools==59.5.0`

## Code references
- [`class LogisticAT`](https://github.com/fabianp/mord/blob/ef578a79bf8374d84b77f246454b06d81a620630/mord/threshold_based.py#L167)
- https://github.com/mirkobunse/CherenkovDeconvolution.py/blob/master/cherenkovdeconvolution/methods/dsea.py
- https://github.com/janjaek/dsea_mord/blob/master/dsea_best_features.ipynb

## Literature
- https://raschka-research-group.github.io/coral-pytorch/tutorials/pytorch_lightning/ordinal-corn_cement
- [https://mail.sebastianraschka.com/pdf/slides/2022-02_rework-coral-lightning.pdf](Easy-to-understand slides explaining CORN/CORAL)
