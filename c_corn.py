import matplotlib.pyplot as plt
from pytorch_lightning.loggers import WandbLogger
import time
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import b_prepare_data
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
import numpy as np
import torch
from coral_pytorch.losses import corn_loss
from coral_pytorch.dataset import corn_label_from_logits
import pytorch_lightning as pl
import torchmetrics
# ours ↓
from b_prepare_data import get_data

# ███ General settings and hyperparameters ███
BATCH_SIZE = 128
# NUM_EPOCHS = 200
NUM_EPOCHS = 5
LEARNING_RATE = 0.005
NUM_WORKERS = 0
# NUM_WORKERS = 40 #TEST
# NROWS = None
NROWS = 50000
DATA_BASEPATH = "./data"
NUM_BINS = 10

# ███ Load data ███
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
    def __init__(self, data_path='./'):
        super().__init__()
        self.data_path = data_path

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
data_module = DataModule(data_path=DATA_BASEPATH)


# ███ Regular PyTorch Module ███
class MultiLayerPerceptron(torch.nn.Module):
    def __init__(self, input_size, hidden_units, num_classes):
        super().__init__()

        # num_classes is used by the corn loss function
        self.num_classes = num_classes

        # Initialize MLP layers
        all_layers = []
        for hidden_unit in hidden_units:
            layer = torch.nn.Linear(input_size, hidden_unit)
            all_layers.append(layer)
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
    hidden_units=(40, 20),
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
    # progress_bar_refresh_rate=50,  # recommended for notebooks
    # enable_progress_bar=False,
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
for batch in data_module.test_dataloader():
    features, labels = batch
    all_labels.append(labels)
    logits = lightning_model(features)
    predicted_labels = corn_label_from_logits(logits)
    all_predicted_labels.append(predicted_labels)

all_labels = torch.cat(all_labels)
all_predicted_labels = torch.cat(all_predicted_labels)


plt.figure(figsize=(10, 6))

plt.bar(np.arange(NUM_BINS), np.bincount(all_labels), alpha=0.7, color='red', label='true class')
plt.bar(np.arange(NUM_BINS), np.bincount(all_predicted_labels, minlength=NUM_BINS), alpha=0.7, color='royalblue', label='predicted class')

plt.legend()
plt.grid()
plt.xlabel('Class')
plt.ylabel('pdf')
plt.xticks(np.arange(NUM_BINS))
plt.savefig('build/corn__hist_log.pdf')
plt.show()

# import code
# code.interact(local=locals())
