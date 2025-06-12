from keras.layers import Input, Layer, Conv2D, MaxPooling2D, Flatten, Dense, LeakyReLU, BatchNormalization
from keras.regularizers import l2
import keras.backend as K
from keras.models import Model

C = 2
S = 7
B = 2
IMG_SIZE = 448

# Final layer of yolo
class Yolo_Reshape(Layer):
    def __init__(self, target_shape, **kwargs):
        super(Yolo_Reshape, self).__init__(**kwargs)
        self.target_shape = tuple(target_shape)

    def compute_output_shape(self, input_shape):
        return (input_shape[0],) + self.target_shape

    def call(self, inputs, **kwargs):
        S = [self.target_shape[0], self.target_shape[1]]
        C = 2
        B = 2
        idx1 = S[0] * S[1] * C
        idx2 = idx1 + S[0] * S[1] * B
        # class prediction
        class_probs = K.reshape(
            inputs[:, :idx1], (K.shape(inputs)[0],) + tuple([S[0], S[1], C]))
        class_probs = K.softmax(class_probs)
        # confidence
        confs = K.reshape(
            inputs[:, idx1:idx2], (K.shape(inputs)[0],) + tuple([S[0], S[1], B]))
        confs = K.sigmoid(confs)
        # boxes
        boxes = K.reshape(
            inputs[:, idx2:], (K.shape(inputs)[0],) + tuple([S[0], S[1], B * 4]))
        boxes = K.sigmoid(boxes)
        # return np.array([class_probs, confs, boxes])
        outputs = K.concatenate([class_probs, confs, boxes])
        return outputs


def tiny_yolov1():
    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 1))
    x = Conv2D(16, (3, 3), padding='same', name='convolutional_0', use_bias=False,
               kernel_regularizer=l2(5e-4))(inputs)
    x = BatchNormalization(name='bnconvolutional_0')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x)

    x = Conv2D(32, (3, 3), padding='same', name='convolutional_1', use_bias=False,
               kernel_regularizer=l2(5e-4))(x)
    x = BatchNormalization(name='bnconvolutional_1')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x)

    x = Conv2D(64, (3, 3), padding='same', name='convolutional_2', use_bias=False,
               kernel_regularizer=l2(5e-4))(x)
    x = BatchNormalization(name='bnconvolutional_2')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x)

    x = Conv2D(128, (3, 3), padding='same', name='convolutional_3', use_bias=False,
               kernel_regularizer=l2(5e-4))(x)
    x = BatchNormalization(name='bnconvolutional_3')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x)

    x = Conv2D(256, (3, 3), padding='same', name='convolutional_4', use_bias=False,
               kernel_regularizer=l2(5e-4))(x)
    x = BatchNormalization(name='bnconvolutional_4')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x)

    x = Conv2D(512, (3, 3), padding='same', name='convolutional_5', use_bias=False,
               kernel_regularizer=l2(5e-4))(x)
    x = BatchNormalization(name='bnconvolutional_5')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x)

    x = Conv2D(1024, (3, 3), padding='same', name='convolutional_6', use_bias=False,
               kernel_regularizer=l2(5e-4))(x)
    x = BatchNormalization(name='bnconvolutional_6')(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = Conv2D(256, (3, 3), padding='same', name='convolutional_7', use_bias=False,
               kernel_regularizer=l2(5e-4))(x)
    x = BatchNormalization(name='bnconvolutional_7')(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = Flatten()(x)
    x = Dense(S*S*(B*5 + C), activation='linear', name='connected_0')(x)
    outputs = Yolo_Reshape((S, S, B*5 + C))(x)

    return Model(inputs, outputs)