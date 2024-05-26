from test import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def createModel():
    m=tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(224,224, 3)),
                tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
                tf.keras.layers.MaxPooling2D((2, 2), strides=2),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(2, activation='softmax')
            ])

    m.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    history = m.fit(
                    x=tf.cast(np.array(data_x), tf.float64), 
                    y=tf.cast(list(map(int,y_data)),tf.int32), 
                    epochs=10)

    m.save("model.h5")

data_x, y_data = createDataset()
createModel()