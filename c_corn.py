import time

import numpy as np
import pytorch_lightning as pl
import torch
import torchmetrics
from coral_pytorch.dataset import corn_label_from_logits
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from torch.utils.data import DataLoader

import wandb
from ca_corn_functions import corn_loss, corn_proba_from_logits


# ███ Performance baseline ███
def baseline_mae(y):
    avg_prediction = np.median(y)  # median minimizes MAE
    baseline_mae = np.mean(np.abs(y - avg_prediction))
    return baseline_mae

# ███ Dataset ███
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, feature_array, label_array, weight_array, dtype=np.float32):
        self.features = feature_array.astype(dtype)
        self.labels = label_array
        self.weights = weight_array if weight_array is not None else np.ones(self.features.shape[0])

    def __getitem__(self, index):
        inputs = self.features[index]
        label = self.labels[index]
        weight = self.weights[index]
        return inputs, label, weight

    def __len__(self):
        return self.features.shape[0]


class MyDatasetPredict(torch.utils.data.Dataset):
    def __init__(self, feature_array, dtype=np.float32):
        self.features = feature_array.astype(dtype)

    def __getitem__(self, index):
        return self.features[index]

    def __len__(self):
        return self.features.shape[0]


# ███ Regular PyTorch Module ███
class CornMlp(torch.nn.Module):
    def __init__(self, input_size, hidden_units, num_classes):
        super().__init__()

        # num_classes is used by the corn loss function
        self.num_classes = num_classes

        # Initialize MLP layers
        all_layers = []
        for hidden_unit in hidden_units:
            all_layers.append(torch.nn.Linear(input_size, hidden_unit))
            # all_layers.append(torch.nn.Dropout(0.2))  # NEW // possible cause for stupid loss
            all_layers.append(torch.nn.LeakyReLU()) # TODO: TEST
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
        # self.train_chi2 = torchmetrics.TODO()

    # Defining the forward method is only necessary
    # if you want to use a Trainer's .predict() method (optional)
    def forward(self, x):
        return self.model(x)

    # A common forward step to compute the loss and labels
    # this is used for training, validation, and testing below
    def _shared_step(self, batch):
        features, true_labels, sample_weights = batch
        logits = self(features)

        # Use CORN loss --------------------------------------
        # A regular classifier uses:
        # loss = torch.nn.functional.cross_entropy(logits, y)
        loss = corn_loss(
            logits, true_labels,
            num_classes=self.model.num_classes,
            # weights=torch.zeros_like(true_labels), # TODO: Test
            weights=sample_weights,
        )
        # ----------------------------------------------------

        # CORN logits to labels ------------------------------
        # A regular classifier uses:
        # predicted_labels = torch.argmax(logits, dim=1)
        predicted_labels = corn_label_from_logits(logits)
        # ----------------------------------------------------

        return loss, true_labels, predicted_labels

    def training_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)
        self.log("train/loss", loss)
        self.train_mae(predicted_labels, true_labels)
        self.log("train/mae", self.train_mae, on_epoch=True, on_step=False, prog_bar=True) # bottleneck?
        # self.train_chi2(predicted_labels, true_labels)
        # self.log("train_chi2", self.train_chi2, on_epoch=True, on_step=False, prog_bar=True) # bottleneck?
        return loss  # this is passed to the optimzer for training

    # validation_step and test_step removed

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


class CornClassifier():
    def __init__(self, input_size, num_classes):
        pytorch_model = CornMlp(
            # input_size=data_module.data_features.shape[1],
            # num_classes=np.bincount(data_module.data_labels).shape[0],
            # ---
            # input_size=X.shape[1],
            # num_classes=np.bincount(y).shape[0],
            # ---
            input_size=input_size,
            num_classes=num_classes,
            # ---
            hidden_units=wandb.config.hidden_units,
        )

        self.lightning_model = LightningMLP(
            model=pytorch_model,
            learning_rate=wandb.config.learning_rate,
        )

        callbacks = [
            RichProgressBar(refresh_rate_per_second=1),
            # TODO: We use train_mae, not valid_mae, because don't have validation data here. Is this correct?
            ModelCheckpoint(save_top_k=1, mode="min", monitor="train_mae"),  # save top 1 model
        ]
        # NOTE: We don't use the prefix kwarg, because the separator '-' is hard-coded, but we want '/'.
        # There is still room for improvement, as we have 'trainer/global_step' and 'epoch' instead of 'train/global_step' and 'train/epoch'.
        wandb_logger = WandbLogger()  # init happens somewhere else

        self.trainer = pl.Trainer(
            max_epochs=wandb.config.num_epochs,
            callbacks=callbacks,
            accelerator='auto',  # Uses GPUs or TPUs if available
            # accelerator='cpu', # restrict to CPU for testing
            # devices='auto',  # Uses all available GPUs/TPUs if applicable
            devices=1,  # use only one GPU: this preserves my sanity
            logger=[
                # csv_logger,
                wandb_logger,
            ],
            enable_model_summary=False,  # annoying when run in DSEA
            deterministic=True,
            log_every_n_steps=1,
        )

    def fit(self, X, y, sample_weight=None):
        """Receives training data only. The wrapper / DSEA should handle all the rest."""
        # TODO: pass sample_weight to the model

        class DataModule(pl.LightningDataModule):
            def __init__(self):
                super().__init__()

            def prepare_data(self):
                pass

            def setup(self, stage=None):
                # self.data_features = X
                # self.data_labels = y

                self.train = MyDataset(X, y, sample_weight)

            def train_dataloader(self):
                return DataLoader(
                    self.train, batch_size=wandb.config.batch_size,
                    num_workers=wandb.config.num_workers,
                    drop_last=True,
                )

        torch.manual_seed(1)
        data_module = DataModule()
        # data_module.setup()

        # reset epoch counter
        self.trainer.fit_loop.epoch_progress.reset_on_epoch()

        # print baseline MAE
        print(f'Baseline MAE: {baseline_mae(y):.2f}')

        start_time = time.time()
        self.trainer.fit(
            model=self.lightning_model,
            datamodule=data_module,
        )

        runtime = (time.time() - start_time)/60
        print(f"Training took {runtime:.2f} min in total.")

        # TODO: remember the best model etc…

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Input: (samples, features)
        Output: (samples, classes)
        """
        X_dataset = MyDatasetPredict(X)

        X_dataloader = DataLoader(
            X_dataset, batch_size=wandb.config.batch_size,
            num_workers=wandb.config.num_workers,
            drop_last=False,  # This is important for evaluation
        )

        # wrong; yields predicted_labels, obviously, which isn't our class_probas
        # y_pred_batches = self.trainer.predict(
        #     model=self.lightning_model,
        #     dataloaders=X_dataloader,
        # )

        all_predicted_labels = []
        for features in X_dataloader:
            logits = self.lightning_model(features)
            predicted_labels = corn_proba_from_logits(logits)
            all_predicted_labels.append(predicted_labels)

        y_pred = torch.cat(all_predicted_labels)

        return y_pred.detach().numpy()


if __name__ == "__main__":
    print("Not anymore.")
