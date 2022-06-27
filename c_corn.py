import time

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torchmetrics
from coral_pytorch.dataset import corn_label_from_logits
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

import wandb
# ours ↓
from b_prepare_data import get_data
from ca_corn_functions import corn_loss, corn_proba_from_logits
from x_config import *


def run_id():
    return f"{time.strftime('%Y%m%d')}_{NROWS}_{'_'.join(map(str, HIDDEN_UNITS))}"


wandb_run = wandb.init(project="dsea-corn")
wandb.config.batch_size = BATCH_SIZE
wandb.config.num_epochs = NUM_EPOCHS
# learning rate is already present for some reason
wandb.config.num_samples = NROWS
wandb.config.num_bins = NUM_BINS


# ███ Load data ███
print("Loading data…")
X, y = get_data(dummy=False, to_numpy=False, nrows=NROWS)
y = y.astype(np.int64)  # convert category → int64
data_features, data_labels = X, y  # TODO

print('Number of features:', data_features.shape[1])
print('Number of examples:', data_features.shape[0])
print('Labels:', np.unique(data_labels.values))
print('Label distribution:', np.bincount(data_labels))


# ███ Performance baseline ███
avg_prediction = np.median(data_labels.values)  # median minimizes MAE
baseline_mae = np.mean(np.abs(data_labels.values - avg_prediction))
print(f'Baseline MAE: {baseline_mae:.2f}')


# ███ Dataset ███
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, feature_array, label_array, dtype=np.float32):
        self.features = feature_array.astype(dtype)
        self.labels = label_array

    def __getitem__(self, index):
        inputs = self.features[index]
        label = self.labels[index]
        return inputs, label

    def __len__(self):
        return self.features.shape[0]


# ███ DataModule ███
class DataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        self.data_features = data_features
        self.data_labels = data_labels

        # Split into
        # 70% train, 10% validation, 20% testing

        # split into ((train, valid), test)
        X_temp, X_test, y_temp, y_test = train_test_split(
            self.data_features.values,
            self.data_labels.values,
            test_size=0.2,
            random_state=1,
            # stratify=self.data_labels.values # TODO
        )

        # split into (train, valid)
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_temp,
            y_temp,
            test_size=0.1,
            random_state=1,
            # stratify=y_temp # TODO
        )

        # Standardize features
        sc = StandardScaler()
        X_train_std = sc.fit_transform(X_train)
        X_valid_std = sc.transform(X_valid)
        X_test_std = sc.transform(X_test)

        self.train = MyDataset(X_train_std, y_train)
        self.valid = MyDataset(X_valid_std, y_valid)
        self.test = MyDataset(X_test_std, y_test)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=BATCH_SIZE,
                          num_workers=NUM_WORKERS,
                          drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.valid, batch_size=BATCH_SIZE,
                          num_workers=NUM_WORKERS)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=BATCH_SIZE,
                          num_workers=NUM_WORKERS)


torch.manual_seed(1)
data_module = DataModule()


# ███ Regular PyTorch Module ███
class MultiLayerPerceptron(torch.nn.Module):
    def __init__(self, input_size, hidden_units, num_classes):
        super().__init__()

        # num_classes is used by the corn loss function
        self.num_classes = num_classes

        # Initialize MLP layers
        all_layers = []
        for hidden_unit in hidden_units:
            all_layers.append(torch.nn.Linear(input_size, hidden_unit))
            all_layers.append(torch.nn.Dropout(0.2))  # NEW
            all_layers.append(torch.nn.ReLU())
            input_size = hidden_unit

        # CORN output layer -------------------------------------------
        # Regular classifier would use num_classes instead of
        # num_classes-1 below
        output_layer = torch.nn.Linear(hidden_units[-1], num_classes-1)
        # -------------------------------------------------------------

        all_layers.append(output_layer)
        self.model = torch.nn.Sequential(*all_layers)

    def forward(self, x):
        x = self.model(x)
        return x


# ███ LightningModule ███
# LightningModule that receives a PyTorch model as input
class LightningMLP(pl.LightningModule):
    def __init__(self, model, learning_rate):
        super().__init__()

        self.learning_rate = learning_rate
        # The inherited PyTorch module
        self.model = model

        # Save settings and hyperparameters to the log directory
        # but skip the model parameters
        self.save_hyperparameters(ignore=['model'])

        # Set up attributes for computing the MAE
        self.train_mae = torchmetrics.MeanAbsoluteError()
        self.valid_mae = torchmetrics.MeanAbsoluteError()
        self.test_mae = torchmetrics.MeanAbsoluteError()

    # Defining the forward method is only necessary
    # if you want to use a Trainer's .predict() method (optional)
    def forward(self, x):
        return self.model(x)

    # A common forward step to compute the loss and labels
    # this is used for training, validation, and testing below
    def _shared_step(self, batch):
        features, true_labels = batch
        logits = self(features)

        # Use CORN loss --------------------------------------
        # A regular classifier uses:
        # loss = torch.nn.functional.cross_entropy(logits, y)
        loss = corn_loss(logits, true_labels,
                         num_classes=self.model.num_classes)
        # ----------------------------------------------------

        # CORN logits to labels ------------------------------
        # A regular classifier uses:
        # predicted_labels = torch.argmax(logits, dim=1)
        predicted_labels = corn_label_from_logits(logits)
        # ----------------------------------------------------

        return loss, true_labels, predicted_labels

    def training_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)
        self.log("train_loss", loss)
        self.train_mae(predicted_labels, true_labels)
        self.log("train_mae", self.train_mae, on_epoch=True, on_step=False)
        return loss  # this is passed to the optimzer for training

    def validation_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)
        self.log("valid_loss", loss)
        self.valid_mae(predicted_labels, true_labels)
        self.log("valid_mae", self.valid_mae,
                 on_epoch=True, on_step=False, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)
        self.test_mae(predicted_labels, true_labels)
        self.log("test_mae", self.test_mae, on_epoch=True, on_step=False)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


# ███ Training ███
pytorch_model = MultiLayerPerceptron(
    input_size=data_features.shape[1],
    hidden_units=HIDDEN_UNITS,
    num_classes=np.bincount(data_labels).shape[0])

lightning_model = LightningMLP(
    model=pytorch_model,
    learning_rate=LEARNING_RATE)


callbacks = [
    RichProgressBar(refresh_rate_per_second=1),
    ModelCheckpoint(save_top_k=1, mode="min", monitor="valid_mae"),  # save top 1 model
]
csv_logger = CSVLogger(save_dir="logs/", name="mlp-corn-cement")
wandb_logger = WandbLogger(project="dsea-corn")


trainer = pl.Trainer(
    max_epochs=NUM_EPOCHS,
    callbacks=callbacks,
    accelerator="auto",  # Uses GPUs or TPUs if available
    devices="auto",  # Uses all available GPUs/TPUs if applicable
    logger=[csv_logger, wandb_logger],
    deterministic=True,
    log_every_n_steps=10)

start_time = time.time()
trainer.fit(model=lightning_model, datamodule=data_module)

runtime = (time.time() - start_time)/60
print(f"Training took {runtime:.2f} min in total.")


# ███ Evaluation ███
# load the best model from the checkpoint
lightning_model = LightningMLP.load_from_checkpoint(
    trainer.checkpoint_callback.best_model_path,
    model=pytorch_model
)
lightning_model.eval()


# Evaluate the model on the test set
all_labels = []
all_predicted_labels = []
all_predicted_probas = []
for batch in data_module.test_dataloader():
    features, labels = batch
    all_labels.append(labels)
    logits = lightning_model(features)
    # ↓ https://github.com/Raschka-research-group/coral-pytorch/blob/6b85e287118476095bac85d6f3dabc6ffb89a326/coral_pytorch/dataset.py#L123
    all_predicted_labels.append(corn_label_from_logits(logits))
    all_predicted_probas.append(corn_proba_from_logits(logits))

all_labels = torch.cat(all_labels)
all_predicted_labels = torch.cat(all_predicted_labels)
all_predicted_probas = torch.cat(all_predicted_probas)


# Export for evaluation
eval_df = pd.DataFrame({
    'labels': all_labels,
    'predicted_labels': all_predicted_labels,
    'predicted_probas': all_predicted_probas.tolist()
})
# print("Saving eval CSV…")
# eval_df.to_csv('build_large/eval.csv', index=False)
print("Saving eval HDF5…")
eval_df.to_hdf('build_large/eval.hdf5', key='eval', index=False)

# import code
# code.interact(local=locals())
