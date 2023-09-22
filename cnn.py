import os
import io
import sys
import requests
import configparser
import traceback
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.metrics import AUC, Precision, Recall
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LambdaCallback,
    LearningRateScheduler,
)
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import matplotlib.pyplot as plt

# ======================== INICIO - TELEGRAM ========================= #

config = configparser.ConfigParser()
config.read("/home/srojas/tg2/Stuff/config.ini")
apiToken = config.get("Telegram", "apiToken")
chatID = config.get("Telegram", "chatID")


def send_telegram_message(message):
    apiURL = f"https://api.telegram.org/bot{apiToken}/sendMessage"
    try:
        response = requests.post(apiURL, json={"chat_id": chatID, "text": message})
    except Exception as e:
        print(e)


# ======================== FIN - TELEGRAM ========================= #

send_telegram_message(
    "[CRATOS] Tu programa {} ha empezado de ejecutarse.".format(
        os.path.basename(__file__)
    )
)

# ======================== INICIO - MIRRORED STRATEGY ========================= #

strategy = tf.distribute.MirroredStrategy()
print("Número de dispositivos: {}".format(strategy.num_replicas_in_sync))
send_telegram_message(
    "Número de dispositivos: {}".format(strategy.num_replicas_in_sync)
)
BATCH_SIZE = 32 * strategy.num_replicas_in_sync

# ======================== FIN - MIRRORED STRATEGY ========================= #

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


class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name="f1_score", **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.precision = Precision()
        self.recall = Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def reset_states(self):
        self.precision.reset_states()
        self.recall.reset_states()

    def result(self):
        precision = self.precision.result()
        recall = self.recall.result()
        return 2 * ((precision * recall) / (precision + recall + 1e-5))


def get_model_summary(model):
    stream = io.StringIO()
    sys.stdout = stream
    model.summary()
    sys.stdout = sys.__stdout__ 
    summary_string = stream.getvalue()
    stream.close()
    return summary_string


def load_and_preprocess_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.cast(image, tf.float32)
    image /= 255.0
    return image, label


def schedule(epoch, lr):
    if epoch % 10 == 0 and epoch != 0:
        return lr * 0.5
    return lr


train_dataset = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
validation_dataset = tf.data.Dataset.from_tensor_slices(
    (validation_paths, validation_labels)
)

train_dataset = (
    train_dataset.map(
        load_and_preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    .batch(BATCH_SIZE)
    .shuffle(buffer_size=64)
    .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
)

validation_dataset = (
    validation_dataset.map(
        load_and_preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    .batch(BATCH_SIZE)
    .shuffle(buffer_size=64)
    .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
)


class EarlyStoppingNotification(tf.keras.callbacks.Callback):
    def __init__(self, early_stopping_callback, telegram_func):
        super().__init__()
        self.es_callback = early_stopping_callback
        self.telegram_func = telegram_func

    def on_epoch_end(self, epoch, logs=None):
        if self.es_callback.stopped_epoch > 0:
            self.telegram_func(f"EarlyStopping activado en la época: {epoch + 1}.")


def send_epoch_notification(epoch, logs):
    message = f"[CRATOS] Fin de la época {epoch+1}\n"
    message += f"Pérdida: {logs['loss']:.4f}\n"
    message += f"Precisión: {logs['accuracy']:.4f}\n"
    if "val_loss" in logs and "val_accuracy" in logs:
        message += f"Pérdida de validación: {logs['val_loss']:.4f}\n"
        message += f"Precisión de validación: {logs['val_accuracy']:.4f}"
    send_telegram_message(message)


def plot_training_history(history):
    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title("Model accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Validation"], loc="upper left")

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("Model loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Validation"], loc="upper left")

    plt.tight_layout()

    script_name = os.path.basename(__file__).split(".")[
        0
    ]  # Obtén el nombre del script sin extensión
    file_path = os.path.join("/home/srojas/tg2/Resultados/", f"{script_name}.png")
    plt.savefig(file_path)

    send_telegram_message(f"[CRATOS] Tu gráfica se guardó en {file_path}")
    # plt.show()


def test_best_model():
    # 1. Cargar el modelo
    loaded_model = tf.keras.models.load_model(
        "/home/srojas/tg2/Stuff/model-best.h5", custom_objects={"F1Score": F1Score}
    )

    # 2. Preparar el conjunto de prueba (test_dataset)
    test_dataset = tf.data.Dataset.from_tensor_slices((test_paths, test_labels))
    test_dataset = (
        test_dataset.map(
            load_and_preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        .batch(BATCH_SIZE)
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    )

    # 3. Evaluar el modelo
    results = loaded_model.evaluate(test_dataset)

    # 4. Enviar los resultados a Telegram
    message = "[CRATOS] Resultados de la Evaluación en el Conjunto de Prueba:\n"
    for name, value in zip(loaded_model.metrics_names, results):
        message += f"{name}: {value:.4f}\n"
    send_telegram_message(message)


def main():
    with strategy.scope():
        try:
            model = Sequential()

            # Primera capa convolucional
            model.add(
                Conv2D(
                    32, kernel_size=(3, 3), activation="relu", input_shape=(360, 480, 3)
                )
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
                monitor="val_loss", patience=10, restore_best_weights=True
            )

            model_checkpoint = ModelCheckpoint(
                "/home/srojas/tg2/Stuff/model-best.h5",
                verbose=1,
                save_best_only=True,
                monitor="val_loss",  
                mode="min", 
            )
            epoch_notification_callback = LambdaCallback(
                on_epoch_end=lambda epoch, logs: send_epoch_notification(epoch, logs)
            )
            early_stopping_notification = EarlyStoppingNotification(
                early_stopping_callback=early_stopping,
                telegram_func=send_telegram_message,
            )
            lr_scheduler = LearningRateScheduler(schedule)

            callbacks = [
                early_stopping,
                model_checkpoint,
                epoch_notification_callback,
                lr_scheduler,
                early_stopping_notification,
            ]

            optimizer = RMSprop(learning_rate=0.001)

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
            send_telegram_message(f"Model Summary:\n{model_summary}")

            model.summary()

            history = model.fit(
                train_dataset,
                validation_data=validation_dataset,
                epochs=20,
                callbacks=callbacks,
                verbose=1,
                class_weight=class_weight_dict,
            )

            plot_training_history(history)
            test_best_model()
        except Exception as e:
            error_message = str(e) + "\n\n" + traceback.format_exc()
            send_telegram_message(f"Error durante el entrenamiento:\n{error_message}")
            print(f"Error durante el entrenamiento:\n{error_message}")

    send_telegram_message(
        "[CRATOS] Tu programa {} ha terminado de ejecutarse.".format(
            os.path.basename(__file__)
        )
    )


if __name__ == "__main__":
    main()
