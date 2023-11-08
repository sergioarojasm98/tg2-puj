import os
import io
import sys
import random
import requests
import configparser
import traceback
import itertools
import numpy as np
import seaborn as sns
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC, Precision, Recall
from tensorflow_addons.metrics import MatthewsCorrelationCoefficient
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LambdaCallback,
    ReduceLROnPlateau,
)
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, roc_curve, auc
from scipy import interpolate
from scipy.stats import pearsonr
from scipy.optimize import brentq
import matplotlib.pyplot as plt

SCRIPT_NAME = os.path.basename(__file__).split(".")[0]
MODEL_NAME = "CNN_Test_7"
MODEL_PATH = os.path.join("/home/srojas/tg2/Models", f"Best_Model_{MODEL_NAME}.h5")

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
    "[CNN] El programa {} ha empezado a ejecutarse.".format(os.path.basename(__file__))
)

# ======================== INICIO - MIRRORED STRATEGY ========================= #

import tensorflow as tf

# Obtener la lista de dispositivos GPU
gpus = tf.config.experimental.list_physical_devices("GPU")

if gpus:
    try:
        for gpu in gpus:
            # Configurando el límite de memoria para cada GPU
            tf.config.experimental.set_virtual_device_configuration(
                gpu,
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6000)],
            )
        logical_gpus = tf.config.experimental.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

# Luego puedes proceder con tu estrategia
strategy = tf.distribute.MirroredStrategy()
send_telegram_message(
    "[CNN] Número de GPUs disponibles: {}".format(strategy.num_replicas_in_sync)
)
BATCH_SIZE = 32 * strategy.num_replicas_in_sync  # Tamaño de lote


# ======================== FIN - MIRRORED STRATEGY ========================= #


class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name="f1_score", **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.precision = Precision()
        self.recall = Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()

    def result(self):
        precision = self.precision.result()
        recall = self.recall.result()
        return 2 * ((precision * recall) / (precision + recall + 1e-5))


class EarlyStoppingNotification(tf.keras.callbacks.Callback):
    def __init__(self, early_stopping_callback, telegram_func):
        super().__init__()
        self.es_callback = early_stopping_callback
        self.telegram_func = telegram_func

    def on_epoch_end(self, epoch, logs=None):
        if self.es_callback.stopped_epoch > 0:
            self.telegram_func(
                f"[CNN] EarlyStopping activado en la época: {epoch + 1}."
            )


class CustomModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):
    def __init__(self, filepath, telegram_func, **kwargs):
        super().__init__(filepath, **kwargs)
        self.telegram_func = telegram_func

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        if self.best == logs.get(self.monitor):
            self.telegram_func(
                f"[CNN] Mejor modelo guardado en la época: {epoch + 1}. Pérdida de validación: {logs.get(self.monitor):.4f}"
            )


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
class_weights = (len(all_image_labels) / (len(unique_labels) * counts)).tolist()
class_weight_dict = dict(zip(unique_labels, class_weights))

train_dataset = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
validation_dataset = tf.data.Dataset.from_tensor_slices(
    (validation_paths, validation_labels)
)


def load_and_preprocess_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.cast(image, tf.float32)
    image /= 255.0
    return image, label


train_dataset = (
    train_dataset.map(
        load_and_preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    .batch(BATCH_SIZE)
    .shuffle(buffer_size=32)
    .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
)

validation_dataset = (
    validation_dataset.map(
        load_and_preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    .batch(BATCH_SIZE)
    .shuffle(buffer_size=32)
    .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
)


def get_model_summary(model):
    stream = io.StringIO()
    sys.stdout = stream
    model.summary()
    sys.stdout = sys.__stdout__
    summary_string = stream.getvalue()
    stream.close()
    return summary_string


def send_epoch_notification(epoch, logs):
    message = f"[CNN] Fin de la época {epoch+1}\n"
    message += f"Loss: {logs['loss']:.4f}\n"
    message += f"Accuracy: {logs['accuracy']:.4f}\n"
    message += f"AUC: {logs['auc']:.4f}\n"
    message += f"Precision: {logs['precision']:.4f}\n"
    message += f"Recall: {logs['recall']:.4f}\n"
    message += f"F1-Score: {logs['f1_score']:.4f}\n"
    if "val_loss" in logs:
        message += f"Loss de validación: {logs['val_loss']:.4f}\n"
        message += f"Accuracy de validación: {logs['val_accuracy']:.4f}\n"
        message += f"AUC de validación: {logs['val_auc']:.4f}\n"
        message += f"Precision de validación: {logs['val_precision']:.4f}\n"
        message += f"Recall de validación: {logs['val_recall']:.4f}\n"
        message += f"F1-Score de validación: {logs['val_f1_score']:.4f}\n"
    send_telegram_message(message)


def plot_training_history(history):
    file_path = os.path.join(
        "/home/srojas/tg2/Resultados/", f"Accuracy_Loss_{SCRIPT_NAMEse}.pdf"
    )
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title(f"CNN Model Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Validation"], loc="upper left")

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("CNN Model Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Validation"], loc="upper left")

    plt.tight_layout()

    plt.savefig(file_path)
    plt.close()

    send_telegram_message(f"[CNN] La gráfica Accuracy vs Loss se guardó en {file_path}")


model = Sequential()


def train_model():
    # Primera capa convolucional
    model.add(
        Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(360, 480, 3))
    )
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Segunda capa convolucional
    model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Tercera capa convolucional
    model.add(Conv2D(128, kernel_size=(3, 3), activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Cuarta capa convolucional
    model.add(Conv2D(256, kernel_size=(3, 3), activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Capas completamente conectadas
    model.add(Flatten())
    model.add(Dense(512, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(1, activation="sigmoid"))

    early_stopping = EarlyStopping(
        monitor="val_loss", patience=7, restore_best_weights=True
    )

    best_model_name = f"/home/srojas/tg2/Models/Best_Model_{SCRIPT_NAME}.h5"

    model_checkpoint = CustomModelCheckpoint(
        best_model_name,
        telegram_func=send_telegram_message,
        verbose=1,
        save_best_only=True,
        monitor="val_loss",
        mode="min",
    )

    epoch_notification_callback = LambdaCallback(
        on_epoch_end=lambda epoch, logs: send_epoch_notification(epoch, logs)
    )
    early_stopping_notification = EarlyStoppingNotification(
        early_stopping_callback=early_stopping, telegram_func=send_telegram_message
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, min_lr=0.0001
    )
    callbacks = [
        early_stopping,
        model_checkpoint,
        epoch_notification_callback,
        reduce_lr,
        early_stopping_notification,
    ]

    optimizer = Adam(learning_rate=0.001)
    model.compile(
        loss="binary_crossentropy",
        optimizer=optimizer,
        metrics=[
            "accuracy",
            AUC(name="auc"),
            Precision(name="precision"),
            Recall(name="recall"),
            F1Score(name="f1_score"),
        ],
    )

    model_summary = get_model_summary(model)
    send_telegram_message(f"[CNN] CNN Model Summary:\n{model_summary}")
    model.summary()
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=20,
        callbacks=callbacks,
        verbose=1,
        class_weight=class_weight_dict,
    )
    save_history_as_csv(history)
    plot_training_history(history)


def save_history_as_csv(history):
    file_path = os.path.join(
        "/home/srojas/tg2/Resultados/CSV_Files/", f"Training_History_{SCRIPT_NAME}.csv"
    )

    df = pd.DataFrame(history.history)
    df.to_csv(file_path, index=False)
    send_telegram_message(f"[CNN] History de entrenamiento guardado en {file_path}")


def save_roc_curve_data_as_csv(y_true, y_pred_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    data = {"FPR": fpr, "TPR": tpr, "Thresholds": thresholds}

    file_path = os.path.join(
        "/home/srojas/tg2/Resultados/CSV_Files/", f"ROC_Data_{SCRIPT_NAME}.csv"
    )

    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)
    send_telegram_message(
        f"[CNN] Los datos de la curva AUCROC se guardaron en {file_path}"
    )


def compute_mcc(confusion_matrix):
    TN, FP = confusion_matrix[0]
    FN, TP = confusion_matrix[1]

    max_val = max(TP, FP, TN, FN)

    if max_val == 0:
        return 0

    TP_scaled = TP / max_val
    FP_scaled = FP / max_val
    TN_scaled = TN / max_val
    FN_scaled = FN / max_val

    denominator_scaled = np.sqrt(
        (TP_scaled + FP_scaled)
        * (TP_scaled + FN_scaled)
        * (TN_scaled + FP_scaled)
        * (TN_scaled + FN_scaled)
    )

    if denominator_scaled == 0:
        return 0

    mcc = (TP_scaled * TN_scaled - FP_scaled * FN_scaled) / denominator_scaled
    return mcc


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


def calculate_far_frr(y_true, y_pred_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    far = fpr
    frr = 1 - tpr
    return far, frr, thresholds


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

    send_telegram_message(f"[CNN] La gráfica de FAR vs FRR se guardó en {file_path}")

    return eer, threshold_eer


def plot_roc_curve(y_true, y_pred_prob):
    save_roc_curve_data_as_csv(y_true, y_pred_prob)
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

    send_telegram_message(f"[CNN] La grafica de  AUCROC se guardó en {file_path}")


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

    send_telegram_message(f"[CNN] La matriz de confusión se guardo en {file_path}")


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
        f"[CNN] Los datos de la densidad del valor de clasificación se guardaron en {file_path}"
    )

    file_path = os.path.join(
        "/home/srojas/tg2/Resultados/", f"Densidad_{SCRIPT_NAME}.pdf"
    )
    plt.savefig(file_path)
    plt.close()

    send_telegram_message(
        f"[CNN] La densidad del valor de clasificación se guardó en {file_path}"
    )


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
        f"[CNN] Los datos de la distribucion del valor de clasificación se guardaron en {file_path}"
    )

    file_path = os.path.join(
        "/home/srojas/tg2/Resultados/", f"Distribucion_{SCRIPT_NAME}.pdf"
    )
    plt.savefig(file_path)
    plt.close()

    send_telegram_message(
        f"[CNN] La grafica de distribucion del valor de clasificación se guardo en {file_path}"
    )


def test_and_metrics():
    # Cargar el modelo
    loaded_model = tf.keras.models.load_model(
        MODEL_PATH, custom_objects={"F1Score": F1Score}
    )

    # Preparar el conjunto de datos de prueba
    test_dataset = tf.data.Dataset.from_tensor_slices((test_paths, test_labels))
    test_dataset = (
        test_dataset.map(
            load_and_preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        .batch(BATCH_SIZE)
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    )

    # Evaluar el modelo en el conjunto de test
    results = loaded_model.evaluate(test_dataset)

    # Crear el mensaje con las métricas de evaluación
    message = "[CNN] Metricas con el conjunto de Test:\n"
    for name, value in zip(loaded_model.metrics_names, results):
        message += f"{name}: {value:.4f}\n"
    send_telegram_message(message)

    # Obtener predicciones
    predictions = loaded_model.predict(test_dataset)
    predicted_labels = np.where(predictions > 0.5, 1, 0).flatten()

    # Calcular la matriz de confusión
    cm = confusion_matrix(test_labels, predicted_labels)

    # Crear gráficos
    plot_roc_curve(test_labels, predictions)
    plot_classification_distribution(test_labels, predictions)
    plot_classification_density(test_labels, predictions)
    plot_confusion_matrix(cm, classes=[0, 1])

    # Calcular métricas adicionales
    mcc_value = compute_mcc(cm)
    eer, threshold_eer = plot_far_frr(test_labels, predictions)

    # Crear el mensaje con las métricas adicionales
    message = "[CNN] Otras metricas con el conjunto de Test:\n"
    message += f"MCC Calculado: {mcc_value:.4f}\n"
    message += f"EER: {eer:.2f} at threshold: {threshold_eer:.2f}\n"

    send_telegram_message(message)


# =================== INICIO - ANALISIS DE POR METODO ===================== #


def save_frr_far_data_as_csv_by_method(y_true, y_pred_prob, method):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    far_values = fpr
    frr_values = 1 - tpr

    data = {"Thresholds": thresholds, "FAR": far_values, "FRR": frr_values}

    file_path = os.path.join(
        "/home/srojas/tg2/Resultados/CSV_Files/",
        f"FRR_FAR_Data_{method}_{SCRIPT_NAME}.csv",
    )

    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)
    send_telegram_message(
        f"[CNN] Los datos de FAR y FRR para {method} se guardaron en {file_path}"
    )


def plot_far_frr_by_method(y_true, y_pred_prob, method):
    save_frr_far_data_as_csv_by_method(y_true, y_pred_prob, method)
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
    plt.title(f"FAR vs. FRR ({method})")
    plt.legend(loc="lower right")

    file_path = os.path.join(
        "/home/srojas/tg2/Resultados/", f"FAR_FRR_{method}_{SCRIPT_NAME}.pdf"
    )
    plt.savefig(file_path)
    plt.close()

    send_telegram_message(
        f"[CNN] La gráfica de FAR vs FRR para {method} se guardó en {file_path}"
    )

    return eer, threshold_eer


def save_roc_curve_data_as_csv_by_method(y_true, y_pred_prob, method):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    data = {"FPR": fpr, "TPR": tpr, "Thresholds": thresholds}

    file_path = os.path.join(
        "/home/srojas/tg2/Resultados/CSV_Files/", f"ROC_Data_{method}_{SCRIPT_NAME}.csv"
    )

    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)
    send_telegram_message(
        f"[CNN] Los datos de la curva AUCROC para {method} se guardaron en {file_path}"
    )


def plot_roc_curve_by_method(y_true, y_pred_prob, method):
    save_roc_curve_data_as_csv_by_method(y_true, y_pred_prob, method)
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

    file_path = os.path.join(
        "/home/srojas/tg2/Resultados/", f"ROC_{method}_{SCRIPT_NAME}.pdf"
    )
    plt.savefig(file_path)
    plt.close()  # Cierre de la figura

    send_telegram_message(
        f"[CNN] La curva AUCROC para {method} se guardó en {file_path}"
    )


def plot_confusion_matrix_by_method(cm, method, classes):
    plt.figure(figsize=(10, 8))  # Incrementa el tamaño de la figura

    # Invierte el orden de la matriz y las clases para mejor visualizacion
    cm = cm[::-1, ::-1]
    classes = classes[::-1]

    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix ({method})")
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
        "/home/srojas/tg2/Resultados/", f"Confusion_Matrix_{method}_{SCRIPT_NAME}.pdf"
    )
    plt.savefig(file_path)
    plt.close()

    send_telegram_message(
        f"[CNN] La matriz de confusión de {method} se guardo en {file_path}"
    )


def plot_classification_density_by_method(y_true, y_pred_prob, method):
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
    ax.set_title(f"Densidad del Valor de Clasificación ({method})")

    if data["Classification_Value"].mean() > 0.5:
        legend_loc = "upper left"
    else:
        legend_loc = "upper right"

    ax.legend(loc=legend_loc)

    # Guardar CSV de los datos de la gráfica
    file_path = os.path.join(
        "/home/srojas/tg2/Resultados/CSV_Files/", f"Densidad_{method}_{SCRIPT_NAME}.csv"
    )
    data.to_csv(file_path, index=False)

    send_telegram_message(
        f"[CNN] Los datos de la densidad del valor de clasificación para {method} se guardaron en {file_path}"
    )

    file_path = os.path.join(
        "/home/srojas/tg2/Resultados/", f"Densidad_{method}_{SCRIPT_NAME}.pdf"
    )
    plt.savefig(file_path)
    plt.close()

    send_telegram_message(
        f"[CNN] La densidad del valor de clasificación para {method} se guardó en {file_path}"
    )


def plot_classification_distribution_by_method(y_true, y_pred_prob, method):
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
    plt.title(f"Distribución del Valor de Clasificación ({method})")
    plt.legend()

    file_path = os.path.join(
        "/home/srojas/tg2/Resultados/CSV_Files/",
        f"Distribucion_{method}_{SCRIPT_NAME}.csv",
    )
    data.to_csv(file_path, index=False)

    send_telegram_message(
        f"[CNN] Los datos de la distribucion del valor de clasificación para {method} se guardaron en {file_path}"
    )

    file_path = os.path.join(
        "/home/srojas/tg2/Resultados/", f"Distribucion_{method}_{SCRIPT_NAME}.pdf"
    )
    plt.savefig(file_path)
    plt.close()

    send_telegram_message(
        f"[CNN] La grafica de distribucion del valor de clasificación para {method} se guardo en {file_path}"
    )


def extract_num_chars_from_paths(true_paths, method):
    # Lista para almacenar los números de caracteres extraídos
    num_chars_list = []

    for path in true_paths:
        # Verifica si el nombre del archivo contiene el método
        if method in path:
            num_chars = path.split("_")[-1].split(".")[0]
            num_chars_list.append(int(num_chars))

    # Convertir la lista a una matriz NumPy 1D
    return np.array(num_chars_list)


def extract_method_from_path(path):
    # Lista de métodos reconocidos
    methods = ["LSB", "DCT", "DWT"]

    # Verifica si el nombre del archivo contiene alguno de los métodos reconocidos
    for method in methods:
        if method in path:
            return method  # Retorna el método encontrado
    # Si no se encuentra ninguno de los métodos, se asume que es una imagen original
    else:
        return "ORIGINAL"


"""
test_path = "IMAGE000001_LSB_12345.png"
send_telegram_message(extract_num_chars_from_paths(test_path))  # debería imprimir 12345
send_telegram_message(extract_method_from_path(test_path))  # debería imprimir LSB
"""


def calculate_errors(true_labels, predictions):
    true_labels_array = np.array(true_labels).ravel()
    predictions_array = np.array(predictions).ravel()
    # Error simple
    return np.abs(predictions_array - true_labels_array).tolist()


"""
def calculate_correlation_numpy(errors, num_chars_list):
    return np.corrcoef(errors, num_chars_list)[0, 1]
"""


def calculate_correlation_scipy(errors, num_chars_list):
    correlation = pearsonr(errors, num_chars_list)[0]
    return correlation


def divide_data_by_method(paths, labels):
    methods = [
        "LSB",
        "DCT",
        "DWT",
        "ORIGINAL",
    ]  # Agrega 'ORIGINAL' a la lista de métodos
    divided_data = {method: {"paths": [], "labels": []} for method in methods}

    for path, label in zip(paths, labels):
        method = extract_method_from_path(path)
        divided_data[method]["paths"].append(path)
        divided_data[method]["labels"].append(label)

    return divided_data


def test_and_metrics_by_method():
    loaded_model = tf.keras.models.load_model(
        MODEL_PATH, custom_objects={"F1Score": F1Score}
    )

    divided_data = divide_data_by_method(test_paths, test_labels)

    # Obtiene la lista de paths y labels para las imágenes ORIGINAL
    original_paths = divided_data["ORIGINAL"]["paths"]
    original_labels = divided_data["ORIGINAL"]["labels"]

    for method, data in divided_data.items():
        if method != "ORIGINAL":
            # Mezcla las listas de paths y labels para las imágenes ORIGINAL
            combined = list(zip(original_paths, original_labels))
            random.shuffle(combined)
            original_paths_shuffled, original_labels_shuffled = zip(*combined)

            # Obtiene un subconjunto de datos ORIGINAL que es del mismo tamaño que el conjunto de datos del método actual
            subset_size = len(data["paths"])
            original_subset_paths = original_paths_shuffled[:subset_size]
            original_subset_labels = original_labels_shuffled[:subset_size]

            # Combina los datos del método actual con los datos ORIGINAL
            combined_paths = data["paths"] + list(original_subset_paths)
            combined_labels = data["labels"] + list(original_subset_labels)

            # Crea el conjunto de datos de prueba
            test_dataset = tf.data.Dataset.from_tensor_slices(
                (combined_paths, combined_labels)
            )
            test_dataset = (
                test_dataset.map(
                    load_and_preprocess_image,
                    num_parallel_calls=tf.data.experimental.AUTOTUNE,
                )
                .batch(BATCH_SIZE)
                .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
            )

            results = loaded_model.evaluate(test_dataset)

            message = (
                f"[CNN] Metricas con el conjunto de Test para el método {method}:\n"
            )
            for name, value in zip(loaded_model.metrics_names, results):
                message += f"{name}: {value:.4f}\n"
            send_telegram_message(message)

            # Análisis detallado de las métricas
            predictions = loaded_model.predict(test_dataset)
            predicted_labels = np.where(predictions > 0.5, 1, 0).flatten()

            cm = confusion_matrix(combined_labels, predicted_labels)

            plot_roc_curve_by_method(combined_labels, predictions, method)
            plot_classification_distribution_by_method(
                combined_labels, predictions, method
            )
            plot_classification_density_by_method(combined_labels, predictions, method)
            plot_confusion_matrix_by_method(cm, method, classes=[0, 1])

            mcc_value = compute_mcc(cm)
            eer, threshold_eer = plot_far_frr_by_method(
                combined_labels, predictions, method
            )

            message = f"[CNN] Otras metricas con el conjunto de Test para el método {method}:\n"
            message += f"MCC Calculado: {mcc_value:.4f}\n"
            message += f"EER: {eer:.2f} at threshold: {threshold_eer:.2f}\n"
            send_telegram_message(message)

            # Error y correlacion con el numero de caracteres

            divided_test_dataset = tf.data.Dataset.from_tensor_slices(
                (data["paths"], data["labels"])
            )
            divided_test_dataset = (
                divided_test_dataset.map(
                    load_and_preprocess_image,
                    num_parallel_calls=tf.data.experimental.AUTOTUNE,
                )
                .batch(BATCH_SIZE)
                .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
            )

            divided_predictions = loaded_model.predict(divided_test_dataset)

            # Los primeros 5 valores de las listas y arrays al mensaje
            message = f"Primeros 5 valores de data['labels']: {data['labels'][:5]}\n"
            message += f"Primeros 5 valores de divided_predictions: {divided_predictions[:5]}\n"
            message += f"Primeros 5 valores de data['paths']: {data['paths'][:5]}\n"
            send_telegram_message(message)

            errors = calculate_errors(data["labels"], divided_predictions)
            num_chars_list = extract_num_chars_from_paths(data["paths"], method)
            correlation = calculate_correlation_scipy(errors, num_chars_list)

            message = f"Primeros 5 valores de errors: {errors[:5]}\n"
            message += f"Primeros 5 valores de num_chars_list: {num_chars_list[:5]}\n"
            message += f"Correlación para el método {method}: {correlation}\n"
            send_telegram_message(message)


# ===================== FIN - ANALISIS POR METODO ====================== #


def main():
    with strategy.scope():
        try:
            # train_model()
            test_and_metrics()
            test_and_metrics_by_method()
        except Exception as e:
            error_message = str(e) + "\n\n" + traceback.format_exc()
            send_telegram_message(
                f"[CNN] Error durante el entrenamiento:\n{error_message}"
            )

    send_telegram_message(
        "[CNN] El programa {} ha terminado de ejecutarse.".format(
            os.path.basename(__file__)
        )
    )


if __name__ == "__main__":
    main()
