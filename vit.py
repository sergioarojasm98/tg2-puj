import os
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from sklearn.model_selection import train_test_split
from timm.models.vision_transformer import VisionTransformer
import torchmetrics
from pytorch_lightning.callbacks import ModelCheckpoint

# 1. Preparación de los datos.
print("Iniciando etapa de preparación de datos...")
folder_paths = [
    "/HDDmedia/srojas/input-data",
    "/HDDmedia/srojas/output-lsb",
    "/HDDmedia/srojas/output-dct",
    "/HDDmedia/srojas/output-dwt",
]
labels = [0, 1, 1, 1]
all_image_paths = []
all_image_labels = []

for folder_path, label in zip(folder_paths, labels):
    for image_file in os.listdir(folder_path):
        full_path = os.path.join(folder_path, image_file)
        try:
            Image.open(full_path).close()  # Abre y cierra inmediatamente para verificar
            all_image_paths.append(full_path)
            all_image_labels.append(label)
        except Exception as e:
            print(f"Error al abrir la imagen: {full_path}. Error: {e}")
    print(f"Carpeta {folder_path} procesada.")


train_paths, temp_paths, train_labels, temp_labels = train_test_split(
    all_image_paths, all_image_labels, test_size=0.3, random_state=42
)
validation_paths, test_paths, validation_labels, test_labels = train_test_split(
    temp_paths, temp_labels, test_size=0.5, random_state=42
)


# 2. Creación del modelo.
class SteganographyClassifier(pl.LightningModule):
    def __init__(self, learning_rate=1e-3):
        super(SteganographyClassifier, self).__init__()
        self.vit = VisionTransformer(
            img_size=(360, 480),
            patch_size=12,
            in_chans=3,
            num_classes=2,
            embed_dim=256,
            depth=5,
            num_heads=4,
        )
        self.learning_rate = learning_rate
        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()
        self.auc = torchmetrics.AUC()
        self.precision = torchmetrics.Precision()
        self.recall = torchmetrics.Recall()
        self.f1_score = torchmetrics.F1Score()
        self.mcc = torchmetrics.MatthewsCorrelationCoefficient()

    def forward(self, x):
        return self.vit(x)

    def _shared_evaluation(self, logits, y, prefix=""):
        loss = F.cross_entropy(logits, y)
        self.log(f"{prefix}loss", loss)
        self.log(f"{prefix}accuracy", self.val_acc(logits, y))
        self.log(f"{prefix}auc", self.auc(logits, y))
        self.log(f"{prefix}precision", self.precision(logits, y))
        self.log(f"{prefix}recall", self.recall(logits, y))
        self.log(f"{prefix}f1", self.f1_score(logits, y))
        self.log(f"{prefix}mcc", self.mcc(logits, y))
        return loss

    def validation_step(self, batch, batch_idx):
        print("Iniciando etapa de validación...")
        x, y = batch
        logits = self(x)
        loss = self._shared_evaluation(logits, y, prefix="val_")

        # Registra la tasa de aprendizaje actual
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", lr)
        print("Etapa de validación completada.")
        return {"loss": loss}

    def test_step(self, batch, batch_idx):
        print("Iniciando etapa de prueba...")
        x, y = batch
        logits = self(x)
        loss = self._shared_evaluation(logits, y, prefix="test_")
        print("Etapa de prueba completada.")
        return {"loss": loss}

    def on_epoch_start(self):
        self.avg_train_acc = torch.tensor(0.0, device=self.device)
        self.batch_count = 0

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.avg_train_acc += self.train_acc(logits, y)
        self.batch_count += 1
        if (batch_idx + 1) % 100 == 0:  # Cada 100 batches
            self.log("avg_train_accuracy", self.avg_train_acc / self.batch_count)
            self.avg_train_acc = torch.tensor(0.0, device=self.device)
            self.batch_count = 0
        return loss

    def training_epoch_end(self, training_step_outputs):
        # Si quedan batches por registrar después de terminar la época:
        if self.batch_count > 0:
            self.log("avg_train_accuracy", self.avg_train_acc / self.batch_count)
            self.avg_train_acc = torch.tensor(0.0, device=self.device)
            self.batch_count = 0

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        # Define el scheduler
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=3, min_lr=0.0001
            ),
            "monitor": "val_loss",
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}


# 3. Definición del módulo de datos.
class CustomImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


class SteganographyDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_paths,
        train_labels,
        validation_paths,
        validation_labels,
        test_paths=None,
        test_labels=None,
        batch_size=32,
    ):
        super(SteganographyDataModule, self).__init__()
        self.batch_size = batch_size
        self.transform = transforms.Compose(
            [
                transforms.Resize((360, 480)),
                transforms.ToTensor(),
            ]
        )
        self.train_dataset = CustomImageDataset(
            train_paths, train_labels, transform=self.transform
        )
        self.val_dataset = CustomImageDataset(
            validation_paths, validation_labels, transform=self.transform
        )
        if test_paths and test_labels:
            self.test_dataset = CustomImageDataset(
                test_paths, test_labels, transform=self.transform
            )
        else:
            self.test_dataset = None

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        if self.test_dataset:
            return DataLoader(
                self.test_dataset, batch_size=self.batch_size, shuffle=False
            )


# 4. Entrenamiento y validación del modelo.
def main():
    print("Inicializando modelo y módulo de datos...")
    model = SteganographyClassifier()
    dm = SteganographyDataModule(
        train_paths,
        train_labels,
        validation_paths,
        validation_labels,
        test_paths,
        test_labels,
    )

    # Guardar el mejor modelo
    script_name = os.path.basename(__file__).split(".")[0]
    base_path = "/home/srojas/tg2/Models"
    model_path = os.path.join(base_path, f"Best_Model_{script_name}.ckpt")

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=base_path,
        filename=f"Best_Model_{script_name}",
        save_top_k=1,
        mode="min",
    )

    trainer = pl.Trainer(
        max_epochs=5,
        gpus=5,  # Número de GPUs
        strategy="dp",  # Entrenamiento en paralelo de datos
        precision=32,
        accelerator="gpu",
        callbacks=[checkpoint_callback],
        progress_bar_refresh_rate=20,
    )

    print("Modelo y módulo de datos inicializados.")

    print("Comenzando entrenamiento...")
    trainer.fit(model, dm)
    print("Entrenamiento completado.")

    print("Cargando el mejor modelo guardado...")
    loaded_model = SteganographyClassifier.load_from_checkpoint(
        checkpoint_path=model_path
    )
    print("Comenzando pruebas con el modelo cargado...")
    trainer.test(loaded_model, datamodule=dm)
    print("Pruebas completadas.")


if __name__ == "__main__":
    main()
