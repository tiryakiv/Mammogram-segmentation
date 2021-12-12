import numpy as np 
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.losses import *

def tversky(y_true, y_pred):
    smooth = 1e-15
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1-y_pred_pos))
    false_pos = K.sum((1-y_true_pos)*y_pred_pos)
    alpha = 0.7
    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)

def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true,y_pred)

def focal_tversky(y_true,y_pred):
    pt_1 = tversky(y_true, y_pred)
    gamma = 0.75
    return K.pow((1-pt_1), gamma)

def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score

def dice_loss(y_true, y_pred):
    loss = 1 - dice_coef(y_true, y_pred)
    return loss

def log_cosh_dice_loss(y_true, y_pred):
    x = dice_loss(y_true, y_pred)
    return tf.math.log((tf.exp(x) + tf.exp(-x)) / 2.0)

#https://github.com/nikhilroxtomar/Polyp-Segmentation-using-UNET-in-TensorFlow-2.0/blob/master/train.py
def iou(y_true, y_pred):
    def f(y_true, y_pred):
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        x = (intersection + 1e-15) / (union + 1e-15)
        x = x.astype(np.float32)
        return x
    return tf.numpy_function(f, [y_true, y_pred], tf.float32)

def conv_block(input, num_filters,dropout_rate):
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(dropout_rate)(x)
    
    return x

def decoder_block(input, skip_features, num_filters, dropout_rate):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters,dropout_rate)
    return x

def build_resnet50_unet(dropout_rate=0.5):
    """ Input """
    inputs = Input(shape=(960, 480, 3))

    """ Pre-trained ResNet50 Model """
    resnet50 = ResNet50(include_top=False, weights="imagenet", input_tensor=inputs)

    """ Encoder """
    s1 = resnet50.get_layer("input_1").output           ## (512 x 512)
    s1 = Dropout(dropout_rate)(s1)
    s2 = resnet50.get_layer("conv1_relu").output        ## (256 x 256)
    s2 = Dropout(dropout_rate)(s2)
    s3 = resnet50.get_layer("conv2_block3_out").output  ## (128 x 128)
    s3 = Dropout(dropout_rate)(s3)
    s4 = resnet50.get_layer("conv3_block4_out").output  ## (64 x 64)
    s4 = Dropout(dropout_rate)(s4)

    """ Bridge """
    b1 = resnet50.get_layer("conv4_block6_out").output  ## (32 x 32)

    """ Decoder """
    d1 = decoder_block(b1, s4, 512, dropout_rate)       ## (64 x 64)
    d2 = decoder_block(d1, s3, 256, dropout_rate)       ## (128 x 128)
    d3 = decoder_block(d2, s2, 128, dropout_rate)       ## (256 x 256)
    d4 = decoder_block(d3, s1, 64, dropout_rate)        ## (512 x 512)

    """ Output """
    outputs = Conv2D(4, 1, padding="same", activation="softmax")(d4)

    model = Model(inputs, outputs, name="Resnet50_U-Net")
    return model