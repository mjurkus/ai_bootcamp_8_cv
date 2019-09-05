import pandas as pd
from keras_preprocessing.image import DataFrameIterator
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def generate_images(data_frame: pd.DataFrame) -> DataFrameIterator:
    datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )

    return datagen.flow_from_dataframe(
        dataframe=data_frame,
        directory='data/cactus/train',
        x_col='id',
        y_col='has_cactus',
        target_size=(32, 32),
        batch_size=32
    )
