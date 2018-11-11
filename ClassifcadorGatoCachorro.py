import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.python.client import device_lib
from keras.models import Sequential
from keras.layers import Dense, Flatten,Conv2D,MaxPooling2D,Dropout
from keras.utils import np_utils
from keras_sequential_ascii import sequential_model_to_ascii_printout
from keras.layers.normalization import BatchNormalization
from tensorflow.python.client import device_lib

def montaRede ( larguraImagem, alturaImagem ) :
    """
    Função que monta a cNN e
    a retorna como parâmetro final

    """

    cNN = Sequential (  )

    # primeira camada
    cNN.add ( Conv2D(
        filters = 32,
        kernel_size = (3, 3),
        input_shape = ( larguraImagem, alturaImagem, 3 ),
        activation = "relu",
    ) )

    cNN.add ( Conv2D(
        filters = 64,
        kernel_size = (3, 3),
        activation = "relu",
    ) )

    cNN.add ( BatchNormalization (  ) )
    cNN.add ( MaxPooling2D (
        pool_size = ( 2, 2 )
    ) )

    # segunda camada
    cNN.add ( Conv2D (
        filters = 32,
        kernel_size = ( 3, 3 ),
        input_shape = ( larguraImagem, alturaImagem, 3 ),
        activation = "relu"
    ) )

    cNN.add ( BatchNormalization (  ) )
    cNN.add ( MaxPooling2D (
        pool_size = ( 2, 2 )
    ) )

    cNN.add ( Flatten ( ) )

    # camada densa
    cNN.add ( Dense (
        units = 128,
        activation = "relu"
    ) )
    cNN.add ( Dropout ( 0.2 ) )
    cNN.add ( Dense (
        units = 128,
        activation = "relu"
    ) )
    cNN.add ( Dropout ( 0.2 ) )
    cNN.add ( Dense (
        units = 2,
        activation = "softmax"
    ) )

    cNN.compile (
         loss = "categorical_crossentropy",
         optimizer = "SGD",
         metrics = ["accuracy"]
     )

    return cNN

def main (  ) :



    print(device_lib.list_local_devices())

    cNN = montaRede ( 64, 64 )

if __name__ == '__main__':
    main()
