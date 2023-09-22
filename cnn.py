import os
import requests
import configparser
import traceback
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LambdaCallback,
    LearningRateScheduler,
)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ======================== INICIO - TELEGRAM ========================= #

config = configparser.ConfigParser()
config.read("/home/srojas/Documentos/Stuff/config.ini")
apiToken = config.get("Telegram", "apiToken")
chatID = config.get("Telegram", "chatID")


def send_telegram_message(message):
    apiURL = f"https://api.telegram.org/bot{apiToken}/sendMessage"
    try:
        response = requests.post(apiURL, json={"chat_id": chatID, "text": message})
    except Exception as e:
        print(e)


# ======================== FIN - TELEGRAM ========================= #

# ======================== INICIO - MIRRORED STRATEGY ========================= #

strategy = tf.distribute.MirroredStrategy()
print("Número de dispositivos: {}".format(strategy.num_replicas_in_sync))
send_telegram_message("Número de dispositivos: {}".format(strategy.num_replicas_in_sync))
BATCH_SIZE = 32 * strategy.num_replicas_in_sync  # Ajustando el tamaño de lote

# ======================== FIN - MIRRORED STRATEGY ========================= #

folder_paths = [
    "/data/estudiantes/srojas/input-data-test",
    "/data/estudiantes/srojas/output-lsb-test",
    "/data/estudiantes/srojas/output-dct-test",
    "/data/estudiantes/srojas/output-dwt-test",
]

labels = [0, 1, 1, 1]
all_image_paths = []
all_image_labels = []

for folder_path, label in zip(folder_paths, labels):
    for image_file in os.listdir(folder_path):
        full_path = os.path.join(folder_path, image_file)
        all_image_paths.append(full_path)
        all_image_labels.append(label)

train_paths, validation_paths, train_labels, validation_labels = train_test_split(
    all_image_paths, all_image_labels, test_size=0.2, random_state=42
)


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
    plt.show()


def main():
    send_telegram_message(
        "[CRATOS] Tu programa {} ha empezado de ejecutarse.".format(
            os.path.basename(__file__)
        )
    )

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
                "model-best.h5", verbose=1, save_best_only=True, save_weights_only=True
            )
            epoch_notification_callback = LambdaCallback(
                on_epoch_end=lambda epoch, logs: send_epoch_notification(epoch, logs)
            )
            lr_scheduler = LearningRateScheduler(schedule)

            callbacks = [
                early_stopping,
                model_checkpoint,
                epoch_notification_callback,
                lr_scheduler,
            ]

            optimizer = RMSprop(learning_rate=0.001)

            model.compile(
                loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"]
            )
            model.summary()

            history = model.fit(
                train_dataset,
                validation_data=validation_dataset,
                epochs=20,
                callbacks=callbacks,
                verbose=1,
            )

            plot_training_history(history)
        except Exception as e:
            error_message = str(e) + "\n\n" + traceback.format_exc()
            send_telegram_message(f"Error during model training:\n{error_message}")
            print(f"Error during model training:\n{error_message}")

    send_telegram_message(
        "[CRATOS] Tu programa {} ha terminado de ejecutarse.".format(
            os.path.basename(__file__)
        )
    )


if __name__ == "__main__":
    main()
