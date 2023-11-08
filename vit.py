import os
import requests
import traceback
import configparser
import random
import itertools
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc
import torch.optim as optim
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from vit_pytorch.parallel_vit import ViT
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryPrecision,
    BinaryRecall,
    BinaryF1Score,
    BinaryMatthewsCorrCoef,
)
import matplotlib.pyplot as plt

SCRIPT_NAME = os.path.basename(__file__).split(".")[0]
EPOCHS = 20

# ======================== INICIO - TELEGRAM ========================= #


def send_telegram_message(message):
    config = configparser.ConfigParser()
    config.read("/home/srojas/tg2/Stuff/config.ini")
    apiToken = config.get("Telegram", "apiToken")
    chatID = config.get("Telegram", "chatID")

    apiURL = f"https://api.telegram.org/bot{apiToken}/sendMessage"
    try:
        response = requests.post(apiURL, json={"chat_id": chatID, "text": message})
        print(message)
    except Exception as e:
        print(e)


# ======================== FIN - TELEGRAM ========================= #

send_telegram_message(
    "[ViT] El programa {} ha empezado a ejecutarse.".format(os.path.basename(__file__))
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instancia las métricas
accuracy = BinaryAccuracy().to(device)
auroc = BinaryAUROC().to(device)
precision = BinaryPrecision().to(device)
recall = BinaryRecall().to(device)
f1 = BinaryF1Score().to(device)
mcc = BinaryMatthewsCorrCoef().to(device)

# 1. Preparación de los datos.
send_telegram_message("[ViT] Iniciando etapa de preparación de datos...")

folder_paths = [
    "/HDDmedia/srojas/input-data",  # Hay 510612 imagenes
    "/HDDmedia/srojas/output-lsb",  # Hay 638265 imagenes
    "/HDDmedia/srojas/output-dct",  # Hay 638265 imagenes
    "/HDDmedia/srojas/output-dwt",  # Hay 638265 imagenes
]

labels = [0, 1, 1, 1]
all_image_paths = []
all_image_labels = []

sample_size = len(os.listdir(folder_paths[0])) // 3

for folder_path, label in zip(folder_paths, labels):
    if label == 0:
        # Si es la clase 0 (input-data), simplemente añadimos todas las imágenes
        for image_file in os.listdir(folder_path):
            full_path = os.path.join(folder_path, image_file)
            all_image_paths.append(full_path)
            all_image_labels.append(label)
    else:
        # Si es clase 1 (output-lsb, output-dct o output-dwt), tomamos una muestra aleatoria
        image_files_sample = random.sample(os.listdir(folder_path), sample_size)
        for image_file in image_files_sample:
            full_path = os.path.join(folder_path, image_file)
            all_image_paths.append(full_path)
            all_image_labels.append(label)


# División de los datos en entrenamiento, validación y pruebas
train_paths, temp_paths, train_labels, temp_labels = train_test_split(
    all_image_paths, all_image_labels, test_size=0.3, random_state=42
)

validation_paths, test_paths, validation_labels, test_labels = train_test_split(
    temp_paths, temp_labels, test_size=0.5, random_state=42
)

unique_labels, counts = np.unique(all_image_labels, return_counts=True)
class_counts = [train_labels.count(i) for i in unique_labels]
class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
samples_weights = class_weights[train_labels]
sampler = WeightedRandomSampler(
    weights=samples_weights, num_samples=len(samples_weights), replacement=True
)


class CustomDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


transform = transforms.Compose(
    [
        transforms.Resize((360, 480)),
        transforms.ToTensor(),
    ]
)

train_dataset = CustomDataset(train_paths, train_labels, transform=transform)
validation_dataset = CustomDataset(
    validation_paths, validation_labels, transform=transform
)

train_loader = DataLoader(
    train_dataset, batch_size=96, sampler=sampler, num_workers=24, pin_memory=True
)
validation_loader = DataLoader(
    validation_dataset, batch_size=96, shuffle=False, num_workers=24, pin_memory=True
)

test_dataset = CustomDataset(test_paths, test_labels, transform=transform)
test_loader = DataLoader(
    test_dataset, batch_size=96, shuffle=False, num_workers=24, pin_memory=True
)


def plot_roc_curve(y_true, y_pred_prob):
    # Calcular la curva ROC
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    # Calcular el AUC
    roc_auc = auc(fpr, tpr)

    # Trazar la curva ROC
    plt.figure(figsize=(10, 7))
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})"
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.legend(loc="lower right")

    file_path = os.path.join("/home/srojas/tg2/Resultados/", f"ROC_{SCRIPT_NAME}.pdf")
    plt.savefig(file_path)
    plt.close()  # Cierre de la figura

    send_telegram_message(f"[ViT] La curva ROC se guardó en {file_path}")


def calculate_far_frr(y_true, y_pred_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    far = fpr
    frr = 1 - tpr
    return far, frr, thresholds


def save_frr_far_data_as_csv(y_true, y_pred_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    far_values = fpr
    frr_values = 1 - tpr

    data = {"Thresholds": thresholds, "FAR": far_values, "FRR": frr_values}

    file_path = os.path.join(
        "/home/srojas/tg2/Resultados/CSV_Files/", f"FRR_FAR_Data_{SCRIPT_NAME}.csv"
    )

    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)
    send_telegram_message(f"[CNN] Los datos de FAR y FRR se guardaron en {file_path}")


def plot_far_frr(y_true, y_pred_prob):
    save_frr_far_data_as_csv(y_true, y_pred_prob)
    far, frr, thresholds = calculate_far_frr(y_true, y_pred_prob)

    idx = np.argmin(np.abs(far - frr))
    eer = (far[idx] + frr[idx]) / 2
    threshold_eer = thresholds[idx]

    plt.figure(figsize=(10, 7))
    plt.plot(thresholds, far, color="blue", lw=2, label=f"FAR")
    plt.plot(thresholds, frr, color="red", lw=2, label=f"FRR")
    plt.axvline(threshold_eer, color="k", linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Accuracy")
    plt.ylabel("Percentage")
    plt.title("FAR vs. FRR")
    plt.legend(loc="lower right")

    file_path = os.path.join(
        "/home/srojas/tg2/Resultados/", f"FAR_FRR_{SCRIPT_NAME}.pdf"
    )
    plt.savefig(file_path)
    plt.close()

    send_telegram_message(f"[ViT] La gráfica de FAR vs FRR se guardó en {file_path}")

    return eer, threshold_eer


def plot_classification_distribution(y_true, y_pred_prob):
    # Convertir a arrays de NumPy
    y_true = np.array(y_true).ravel()
    y_pred_prob = np.array(y_pred_prob).ravel()

    data = pd.DataFrame({"Classification_Value": y_pred_prob, "Class": y_true})

    # Clasificación positiva y negativa
    positive_class = y_pred_prob[y_true == 1]
    negative_class = y_pred_prob[y_true == 0]

    plt.figure(figsize=(10, 7))
    plt.hist(
        positive_class,
        bins=50,
        color="blue",
        label="Clase Positiva",
        alpha=0.7,
        density=True,
    )
    plt.hist(
        negative_class,
        bins=50,
        color="red",
        label="Clase Negativa",
        alpha=0.7,
        density=True,
    )
    plt.xlabel("Valor de Clasificación")
    plt.ylabel("Frecuencia")
    plt.title(f"Distribución del Valor de Clasificación")
    plt.legend()

    file_path = os.path.join(
        "/home/srojas/tg2/Resultados/CSV_Files/",
        f"Distribucion_{SCRIPT_NAME}.csv",
    )
    data.to_csv(file_path, index=False)

    send_telegram_message(
        f"[ViT] Los datos de la distribucion del valor de clasificación se guardaron en {file_path}"
    )

    file_path = os.path.join(
        "/home/srojas/tg2/Resultados/", f"Distribucion_{SCRIPT_NAME}.pdf"
    )
    plt.savefig(file_path)
    plt.close()

    send_telegram_message(
        f"[ViT] La grafica de distribucion del valor de clasificación se guardo en {file_path}"
    )


def plot_classification_density(y_true, y_pred_prob):
    y_pred_prob = np.array(y_pred_prob).ravel()
    y_true = np.array(y_true).ravel()

    # Convertir a un DataFrame de pandas
    data = pd.DataFrame({"Classification_Value": y_pred_prob, "Class": y_true})

    # Configurar el estilo de Seaborn
    sns.set_style("white")

    # Crear un objeto de figura y ejes para tener un control preciso
    fig, ax = plt.subplots(figsize=(10, 7))

    # Gráfico de la clase positiva
    sns.kdeplot(
        data=data[data["Class"] == 1],
        x="Classification_Value",
        fill=True,
        color="blue",
        label="Clase Positiva",
        alpha=0.7,
        ax=ax,
    )

    # Gráfico de la clase negativa
    sns.kdeplot(
        data=data[data["Class"] == 0],
        x="Classification_Value",
        fill=True,
        color="red",
        label="Clase Negativa",
        alpha=0.7,
        ax=ax,
    )

    ax.set_xlabel("Valor de Clasificación")
    ax.set_ylabel("Densidad")
    ax.set_title(f"Densidad del Valor de Clasificación")

    if data["Classification_Value"].mean() > 0.5:
        legend_loc = "upper left"
    else:
        legend_loc = "upper right"

    ax.legend(loc=legend_loc)

    # Guardar CSV de los datos de la gráfica
    file_path = os.path.join(
        "/home/srojas/tg2/Resultados/CSV_Files/", f"Densidad_{SCRIPT_NAME}.csv"
    )
    data.to_csv(file_path, index=False)

    send_telegram_message(
        f"[ViT] Los datos de la densidad del valor de clasificación se guardaron en {file_path}"
    )

    file_path = os.path.join(
        "/home/srojas/tg2/Resultados/", f"Densidad_{SCRIPT_NAME}.pdf"
    )
    plt.savefig(file_path)
    plt.close()

    send_telegram_message(
        f"[ViT] La densidad del valor de clasificación se guardó en {file_path}"
    )


def plot_confusion_matrix(cm, classes):
    plt.figure(figsize=(10, 8))  # Incrementa el tamaño de la figura

    # Invierte el orden de la matriz y las clases para mejor visualizacion
    cm = cm[::-1, ::-1]
    classes = classes[::-1]

    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = "d"
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    plt.ylabel("Actual Values")
    plt.xlabel("Predicted Values")

    file_path = os.path.join(
        "/home/srojas/tg2/Resultados/", f"Confusion_Matrix_{SCRIPT_NAME}.pdf"
    )
    plt.savefig(file_path)
    plt.close()

    send_telegram_message(f"[ViT] La matriz de confusión se guardo en {file_path}")


# Definición del modelo
v = ViT(
    image_size = 480,
    patch_size = 30,
    num_classes = 1,  
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1,
    num_parallel_branches = 6  
)

if torch.cuda.device_count() > 1:
    v = torch.nn.DataParallel(v, device_ids=[0, 1, 2, 3, 4, 5])

v = v.to(device)

# Definir el optimizador
optimizer = optim.Adam(v.parameters(), lr=0.001)

best_val_loss = float("inf")
best_val_metrics = {}
best_train_metrics = {}


def train_and_validate(epochs):
    global best_val_loss
    global best_val_metrics
    global best_train_metrics

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(1, epochs + 1):
        send_telegram_message(f"[ViT] Inicio de la epoca {epoch}/{EPOCHS}")

        # Resetear las métricas al inicio de cada época para el entrenamiento
        accuracy.reset()
        auroc.reset()
        precision.reset()
        recall.reset()
        f1.reset()

        # Entrenamiento
        v.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = v(data)
            loss = F.binary_cross_entropy_with_logits(output.squeeze(), target.float())
            loss.backward()
            optimizer.step()

            preds = torch.sigmoid(output).squeeze()
            # Actualizar métricas (sin calcularlas aún)
            accuracy.update(preds, target)
            auroc.update(preds, target)
            precision.update(preds, target)
            recall.update(preds, target)
            f1.update(preds, target)

        # Guardar métricas temporales de entrenamiento
        temp_train_metrics = {
            "Loss": loss.item(),
            "Accuracy": accuracy.compute().item(),
            "AUROC": auroc.compute().item(),
            "Precision": precision.compute().item(),
            "Recall": recall.compute().item(),
            "F1 Score": f1.compute().item(),
        }

        message = (
            f"-- TRAINING (EPOCH {epoch}) --\n"
            f"Loss: {loss.item()}\n"
            f"Accuracy: {accuracy.compute()}\n"
            f"AUROC: {auroc.compute()}\n"
            f"Precision: {precision.compute()}\n"
            f"Recall: {recall.compute()}\n"
            f"F1 Score: {f1.compute()}"
        )
        send_telegram_message(message)

        train_losses.append(loss.item())
        train_accuracies.append(accuracy.compute().item())

        accuracy.reset()
        auroc.reset()
        precision.reset()
        recall.reset()
        f1.reset()

        # Validación
        v.eval()
        total_loss = 0.0
        with torch.no_grad():
            for data, target in validation_loader:
                data, target = data.to(device), target.to(device)
                output = v(data)
                preds = torch.sigmoid(output).squeeze()
                loss_val = F.binary_cross_entropy_with_logits(
                    output.squeeze(), target.float()
                )
                total_loss += loss_val.item()
                # Actualizar métricas (sin calcularlas aún)
                accuracy.update(preds, target)
                auroc.update(preds, target)
                precision.update(preds, target)
                recall.update(preds, target)
                f1.update(preds, target)

        avg_loss = total_loss / len(validation_loader)
        val_losses.append(avg_loss)  # Esta línea se mueve aquí
        val_accuracies.append(accuracy.compute().item())

        message = (
            f"-- VALIDATION (EPOCH {epoch}) -- \n"
            f"Loss: {avg_loss:.6f}\n"
            f"Accuracy: {accuracy.compute()}\n"
            f"AUROC: {auroc.compute()}\n"
            f"Precision: {precision.compute()}\n"
            f"Recall: {recall.compute()}\n"
            f"F1 Score: {f1.compute()}"
        )
        send_telegram_message(message)

        # Comprobar si esta es la mejor pérdida y actualizar todo si es necesario
        if avg_loss < best_val_loss:
            best_val_loss = avg_loss
            best_train_metrics = temp_train_metrics
            best_val_metrics = {  # Actualizar las métricas de validación
                "Loss": avg_loss,
                "Accuracy": accuracy.compute().item(),
                "AUROC": auroc.compute().item(),
                "Precision": precision.compute().item(),
                "Recall": recall.compute().item(),
                "F1 Score": f1.compute().item(),
            }

            # Guardar el mejor modelo
            base_path = "/home/srojas/tg2/Models"
            model_path = os.path.join(base_path, f"Best_Model_{SCRIPT_NAME}.pth")
            torch.save(v.state_dict(), model_path)
            send_telegram_message(
                f"[ViT] Guardado el mejor modelo con pérdida de validación: {avg_loss:.6f} en {model_path}"
            )

    # Al final de todas las épocas, enviar las métricas del mejor modelo
    if best_val_metrics:
        train_msg = "-- BEST TRAINING METRICS --\n" + "\n".join(
            [f"{k}: {v:.6f}" for k, v in best_train_metrics.items()]
        )
        val_msg = "-- BEST VALIDATION METRICS --\n" + "\n".join(
            [f"{k}: {v:.6f}" for k, v in best_val_metrics.items()]
        )
        send_telegram_message(train_msg)
        send_telegram_message(val_msg)
        # Gráfica de precisión y pérdida por época
        file_path = os.path.join(
            "/home/srojas/tg2/Resultados/", f"Accuracy_Loss_{SCRIPT_NAME}.pdf"
        )
        plt.figure(figsize=(12, 4))

        # Gráfica de precisión
        plt.subplot(1, 2, 1)
        plt.plot(train_accuracies, label="Train Accuracy")
        plt.plot(val_accuracies, label="Validation Accuracy")
        plt.title("Model Accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.legend(loc="upper left")

        # Gráfica de pérdida
        plt.subplot(1, 2, 2)
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.title("Model Loss")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.legend(loc="upper left")

        plt.tight_layout()
        plt.savefig(file_path)
        plt.close()

        send_telegram_message(
            f"[ViT] La gráfica Accuracy vs Loss se guardó en {file_path}"
        )

    else:
        send_telegram_message(
            "[ViT] No se encontró una mejora en la pérdida de validación a lo largo de las épocas."
        )


# Test del modelo
def test():
    base_path = "/home/srojas/tg2/Models"
    model_path = os.path.join(base_path, f"Best_Model_{SCRIPT_NAME}.pth")
    v.load_state_dict(torch.load(model_path))
    v.eval()
    accuracy.reset()
    precision.reset()
    recall.reset()
    f1.reset()
    mcc.reset()
    auroc.reset()

    all_targets = []
    all_preds = []

    total_loss = 0.0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = v(data)
            preds = torch.sigmoid(output).squeeze()
            loss_test = F.binary_cross_entropy_with_logits(
                output.squeeze(), target.float()
            )
            total_loss += loss_test.item()

            # Actualizar métricas
            accuracy.update(preds, target)
            precision.update(preds, target)
            recall.update(preds, target)
            f1.update(preds, target)
            mcc.update(preds, target)
            auroc.update(preds, target)  # Añadido el update para auroc

            all_targets.extend(target.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # Calcular el loss promedio
    avg_loss = total_loss / len(test_loader)

    # Calcular AUROC
    auc_value = auroc.compute()

    message = (
        f"-- TEST -- \n"
        f"Loss: {avg_loss:.6f}\n"
        f"Accuracy: {accuracy.compute()}\n"
        f"Precision: {precision.compute()}\n"
        f"Recall: {recall.compute()}\n"
        f"F1 Score: {f1.compute()}\n"
        f"MCC: {mcc.compute()}\n"
        f"AUROC: {auc_value}"
    )
    send_telegram_message(message)

    predicted_labels = np.where(np.array(all_preds) > 0.5, 1, 0).flatten()

    # Calcular la matriz de confusión
    cm = confusion_matrix(all_targets, predicted_labels)

    # Crear gráficos
    plot_roc_curve(all_targets, all_preds)
    plot_classification_distribution(all_targets, all_preds)
    plot_classification_density(all_targets, all_preds)
    plot_confusion_matrix(cm, classes=[0, 1])
    eer, threshold_eer = plot_far_frr(all_targets, all_preds)

    send_telegram_message(f"EER: {eer:.2f} at threshold: {threshold_eer:.2f}\n")


def run():
    try:
        send_telegram_message("[ViT] Iniciando etapa de entrenamiento y validacion...")
        train_and_validate(EPOCHS)
        send_telegram_message("[ViT] Iniciando etapa de prueba...")
        test()

    except Exception as e:
        error_message = str(e) + "\n\n" + traceback.format_exc()
        send_telegram_message(f"[ViT] Error durante la ejecucion:\n{error_message}")

    send_telegram_message(
        "[ViT] El programa {} ha terminado de ejecutarse.".format(
            os.path.basename(__file__)
        )
    )


if __name__ == "__main__":
    run()
