from tensorflow.keras.callbacks import Callback


class CustomLogger(Callback):

    def on_epoch_end(self, epoch: int, logs=None) -> None:
        print(f"custom log {epoch}")
