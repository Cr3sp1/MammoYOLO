from keras.layers import Input, Layer, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, LeakyReLU
import keras.backend as K
from keras.models import Model

C = 2
S = 7
B = 2
IMG_SIZE = 448

# Final layer of yolo, reshapes previous layer
class Yolo_Reshape(Layer):
    def __init__(self, **kwargs):
        super(Yolo_Reshape, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], S, S, C  + B * 5)
    def call(self, inputs, **kwargs):
        idx1 = S * S * C
        idx2 = idx1 + S * S * B
        # class prediction
        class_probs = K.reshape(
            inputs[:, :idx1], (K.shape(inputs)[0],) + tuple([S, S, C]))
        # confidence
        confs = K.reshape(
            inputs[:, idx1:idx2], (K.shape(inputs)[0],) + tuple([S, S, B]))
        # boxes
        boxes = K.reshape(
            inputs[:, idx2:], (K.shape(inputs)[0],) + tuple([S, S, B * 4]))
        outputs = K.concatenate([class_probs, confs, boxes])
        return outputs
    

def tiny_yolov1():
    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 1))
    x = Conv2D(filters=16, kernel_size=3, padding='same', name='convolutional_0',
               activation=LeakyReLU(alpha=0.1))(inputs)
    x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x)

    x = Conv2D(filters=32, kernel_size=3, padding='same', name='convolutional_1',
               activation=LeakyReLU(alpha=0.1))(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x)

    x = Conv2D(filters=64, kernel_size=3, padding='same', name='convolutional_2',
               activation=LeakyReLU(alpha=0.1))(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x)

    x = Conv2D(filters=128, kernel_size=3, padding='same', name='convolutional_3',
               activation=LeakyReLU(alpha=0.1))(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x)

    x = Conv2D(filters=256, kernel_size=3, padding='same', name='convolutional_4',
               activation=LeakyReLU(alpha=0.1))(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x)

    x = Conv2D(filters=512, kernel_size=3, padding='same', name='convolutional_5',
               activation=LeakyReLU(alpha=0.1))(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x)

    x = Conv2D(filters=1024, kernel_size=3, padding='same', name='convolutional_6',
               activation=LeakyReLU(alpha=0.1))(x)
    x = Conv2D(filters=256, kernel_size=3, padding='same', name='convolutional_7',
               activation=LeakyReLU(alpha=0.1))(x)

    x = Flatten()(x)

    x = Dense(1470, activation=LeakyReLU(alpha=0.1), name='connected_0')(x)
    
    x = Dropout(0.5)(x)

    x = Dense(S*S*(B*5 + C), activation='linear', name='connected_1')(x)
    outputs = Yolo_Reshape()(x)

    return Model(inputs, outputs, name="tiny_yolov1")
    

def yolov1():
    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 1))
    x = Conv2D(filters=64, kernel_size=7, strides=2, padding='same', name='convolutional_0',
               activation=LeakyReLU(alpha=0.1))(inputs)
    x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x)

    x = Conv2D(filters=192, kernel_size=7, padding='same', name='convolutional_1',
               activation=LeakyReLU(alpha=0.1))(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x)

    x = Conv2D(filters=128, kernel_size=1, padding='same', name='convolutional_2',
               activation=LeakyReLU(alpha=0.1))(x)
    x = Conv2D(filters=256, kernel_size=3, padding='same', name='convolutional_3',
               activation=LeakyReLU(alpha=0.1))(x)
    x = Conv2D(filters=256, kernel_size=1, padding='same', name='convolutional_4',
               activation=LeakyReLU(alpha=0.1))(x)
    x = Conv2D(filters=512, kernel_size=3, padding='same', name='convolutional_5',
               activation=LeakyReLU(alpha=0.1))(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x)

    x = Conv2D(filters=256, kernel_size=1, padding='same', name='convolutional_6',
               activation=LeakyReLU(alpha=0.1))(x)
    x = Conv2D(filters=512, kernel_size=3, padding='same', name='convolutional_7',
               activation=LeakyReLU(alpha=0.1))(x)
    x = Conv2D(filters=256, kernel_size=1, padding='same', name='convolutional_8',
               activation=LeakyReLU(alpha=0.1))(x)
    x = Conv2D(filters=512, kernel_size=3, padding='same', name='convolutional_9',
               activation=LeakyReLU(alpha=0.1))(x)
    x = Conv2D(filters=256, kernel_size=1, padding='same', name='convolutional_10',
               activation=LeakyReLU(alpha=0.1))(x)
    x = Conv2D(filters=512, kernel_size=3, padding='same', name='convolutional_11',
               activation=LeakyReLU(alpha=0.1))(x)
    x = Conv2D(filters=256, kernel_size=1, padding='same', name='convolutional_12',
               activation=LeakyReLU(alpha=0.1))(x)
    x = Conv2D(filters=512, kernel_size=3, padding='same', name='convolutional_13',
               activation=LeakyReLU(alpha=0.1))(x)
    x = Conv2D(filters=512, kernel_size=1, padding='same', name='convolutional_14',
               activation=LeakyReLU(alpha=0.1))(x)
    x = Conv2D(filters=1024, kernel_size=3, padding='same', name='convolutional_15',
               activation=LeakyReLU(alpha=0.1))(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x)

    x = Conv2D(filters=512, kernel_size=1, padding='same', name='convolutional_16',
               activation=LeakyReLU(alpha=0.1))(x)
    x = Conv2D(filters=1024, kernel_size=3, padding='same', name='convolutional_17',
               activation=LeakyReLU(alpha=0.1))(x)
    x = Conv2D(filters=512, kernel_size=1, padding='same', name='convolutional_18',
               activation=LeakyReLU(alpha=0.1))(x)
    x = Conv2D(filters=1024, kernel_size=3, padding='same', name='convolutional_19',
               activation=LeakyReLU(alpha=0.1))(x)
    x = Conv2D(filters=1024, kernel_size=3, padding='same', name='convolutional_20',
               activation=LeakyReLU(alpha=0.1))(x)
    x = Conv2D(filters=1024, kernel_size=3, strides=2, padding='same', name='convolutional_21',
               activation=LeakyReLU(alpha=0.1))(x)
    
    x = Conv2D(filters=1024, kernel_size=3, padding='same', name='convolutional_22',
               activation=LeakyReLU(alpha=0.1))(x)
    x = Conv2D(filters=256, kernel_size=3, padding='same', name='convolutional_23',
               activation=LeakyReLU(alpha=0.1))(x)

    x = Flatten()(x)
    
    x = Dense(1470, activation=LeakyReLU(alpha=0.1), name='connected_0')(x)

    x = Dropout(0.5)(x)

    x = Dense(S*S*(B*5 + C), activation='linear', name='connected_1')(x)
    outputs = Yolo_Reshape()(x)

    return Model(inputs, outputs, name="yolov1")