from keras.layers import Input, Layer, Conv2D, MaxPooling2D, LocallyConnected2D, Flatten, Dense, Dropout, LeakyReLU
from keras.regularizers import l2
import keras.backend as K
from keras.models import Model

C = 2
S = 7
B = 2
IMG_SIZE = 448

# Final layer of yolo, reshapes previous layer
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
    

def yolov1():
    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 1))
    x = Conv2D(filters=64, kernel_size=7, strides=2, padding='same', name='convolutional_0',
               kernel_regularizer=l2(5e-4), activation=LeakyReLU(alpha=0.1))(inputs)
    x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x)

    x = Conv2D(filters=192, kernel_size=7, padding='same', name='convolutional_1',
               kernel_regularizer=l2(5e-4), activation=LeakyReLU(alpha=0.1))(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x)

    x = Conv2D(filters=128, kernel_size=1, padding='same', name='convolutional_2',
               kernel_regularizer=l2(5e-4), activation=LeakyReLU(alpha=0.1))(x)
    x = Conv2D(filters=256, kernel_size=3, padding='same', name='convolutional_3',
               kernel_regularizer=l2(5e-4), activation=LeakyReLU(alpha=0.1))(x)
    x = Conv2D(filters=256, kernel_size=1, padding='same', name='convolutional_4',
               kernel_regularizer=l2(5e-4), activation=LeakyReLU(alpha=0.1))(x)
    x = Conv2D(filters=512, kernel_size=3, padding='same', name='convolutional_5',
               kernel_regularizer=l2(5e-4), activation=LeakyReLU(alpha=0.1))(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x)

    x = Conv2D(filters=256, kernel_size=1, padding='same', name='convolutional_6',
               kernel_regularizer=l2(5e-4), activation=LeakyReLU(alpha=0.1))(x)
    x = Conv2D(filters=512, kernel_size=3, padding='same', name='convolutional_7',
               kernel_regularizer=l2(5e-4), activation=LeakyReLU(alpha=0.1))(x)
    x = Conv2D(filters=256, kernel_size=1, padding='same', name='convolutional_8',
               kernel_regularizer=l2(5e-4), activation=LeakyReLU(alpha=0.1))(x)
    x = Conv2D(filters=512, kernel_size=3, padding='same', name='convolutional_9',
               kernel_regularizer=l2(5e-4), activation=LeakyReLU(alpha=0.1))(x)
    x = Conv2D(filters=256, kernel_size=1, padding='same', name='convolutional_10',
               kernel_regularizer=l2(5e-4), activation=LeakyReLU(alpha=0.1))(x)
    x = Conv2D(filters=512, kernel_size=3, padding='same', name='convolutional_11',
               kernel_regularizer=l2(5e-4), activation=LeakyReLU(alpha=0.1))(x)
    x = Conv2D(filters=256, kernel_size=1, padding='same', name='convolutional_12',
               kernel_regularizer=l2(5e-4), activation=LeakyReLU(alpha=0.1))(x)
    x = Conv2D(filters=512, kernel_size=3, padding='same', name='convolutional_13',
               kernel_regularizer=l2(5e-4), activation=LeakyReLU(alpha=0.1))(x)
    x = Conv2D(filters=512, kernel_size=1, padding='same', name='convolutional_14',
               kernel_regularizer=l2(5e-4), activation=LeakyReLU(alpha=0.1))(x)
    x = Conv2D(filters=1024, kernel_size=3, padding='same', name='convolutional_15',
               kernel_regularizer=l2(5e-4), activation=LeakyReLU(alpha=0.1))(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x)

    x = Conv2D(filters=512, kernel_size=1, padding='same', name='convolutional_16',
               kernel_regularizer=l2(5e-4), activation=LeakyReLU(alpha=0.1))(x)
    x = Conv2D(filters=1024, kernel_size=3, padding='same', name='convolutional_17',
               kernel_regularizer=l2(5e-4), activation=LeakyReLU(alpha=0.1))(x)
    x = Conv2D(filters=512, kernel_size=1, padding='same', name='convolutional_18',
               kernel_regularizer=l2(5e-4), activation=LeakyReLU(alpha=0.1))(x)
    x = Conv2D(filters=1024, kernel_size=3, padding='same', name='convolutional_19',
               kernel_regularizer=l2(5e-4), activation=LeakyReLU(alpha=0.1))(x)
    x = Conv2D(filters=1024, kernel_size=3, padding='same', name='convolutional_20',
               kernel_regularizer=l2(5e-4), activation=LeakyReLU(alpha=0.1))(x)
    x = Conv2D(filters=1024, kernel_size=3, strides=2, padding='same', name='convolutional_21',
               kernel_regularizer=l2(5e-4), activation=LeakyReLU(alpha=0.1))(x)
    
    x = Conv2D(filters=1024, kernel_size=3, padding='same', name='convolutional_22',
               kernel_regularizer=l2(5e-4), activation=LeakyReLU(alpha=0.1))(x)
    x = Conv2D(filters=1024, kernel_size=3, padding='same', name='convolutional_23',
               kernel_regularizer=l2(5e-4), activation=LeakyReLU(alpha=0.1))(x)
    
    x = LocallyConnected2D(filters=256, kernel_size=3, strides=1, activation=LeakyReLU(alpha=0.1))(x)
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(1715, activation=LeakyReLU(alpha=0.1), name='connected_0')(x)

    x = Dense(S*S*(B*5 + C), activation='linear', name='connected_1')(x)
    outputs = Yolo_Reshape((S, S, B*5 + C))(x)

    return Model(inputs, outputs)


def tiny_yolov1():
    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 1))
    x = Conv2D(filters=16, kernel_size=3, padding='same', name='convolutional_0',
               kernel_regularizer=l2(5e-4), activation=LeakyReLU(alpha=0.1))(inputs)
    x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x)

    x = Conv2D(filters=32, kernel_size=3, padding='same', name='convolutional_1',
               kernel_regularizer=l2(5e-4), activation=LeakyReLU(alpha=0.1))(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x)

    x = Conv2D(filters=64, kernel_size=3, padding='same', name='convolutional_2',
               kernel_regularizer=l2(5e-4), activation=LeakyReLU(alpha=0.1))(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x)

    x = Conv2D(filters=128, kernel_size=3, padding='same', name='convolutional_3',
               kernel_regularizer=l2(5e-4), activation=LeakyReLU(alpha=0.1))(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x)

    x = Conv2D(filters=256, kernel_size=3, padding='same', name='convolutional_4',
               kernel_regularizer=l2(5e-4), activation=LeakyReLU(alpha=0.1))(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x)

    x = Conv2D(filters=512, kernel_size=3, padding='same', name='convolutional_5',
               kernel_regularizer=l2(5e-4), activation=LeakyReLU(alpha=0.1))(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x)

    x = Conv2D(filters=1024, kernel_size=3, padding='same', name='convolutional_6',
               kernel_regularizer=l2(5e-4), activation=LeakyReLU(alpha=0.1))(x)
    x = Conv2D(filters=256, kernel_size=3, padding='same', name='convolutional_7',
               kernel_regularizer=l2(5e-4), activation=LeakyReLU(alpha=0.1))(x)

    x = Flatten()(x)

    x = Dense(units = 1470, activation=LeakyReLU(alpha=0.1), name='connected_0')(x)
    x = Dropout(0.5)(x)

    x = Dense(units = S*S*(B*5 + C), activation='linear', name='connected_1')(x)
    outputs = Yolo_Reshape((S, S, B*5 + C))(x)

    return Model(inputs, outputs)