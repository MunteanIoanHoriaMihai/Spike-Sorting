import keras
from keras.layers import *
from keras import Model
from keras.regularizers import l1
import tensorflow as tf
import numpy as np

from feature_extraction.autoencoder.model_auxiliaries import get_codes
from feature_extraction.autoencoder.softplusplus_activation import softplusplus


class AutoencoderModel(Model):

    def __init__(self, input_size, encoder_layer_sizes, decoder_layer_sizes, code_size, output_activation, loss_function):
        super(AutoencoderModel, self).__init__()
        self.input_size = input_size
        self.loss_function = loss_function
        # self.code_size = 2  # size of the desired compression 2D or 3D to be able to visualize

        self.input_spike = Input(shape=(self.input_size,))

        current_layer = self.input_spike
        for hidden_layer_size in encoder_layer_sizes:
            hidden_layer = Dense(hidden_layer_size, activation='relu')(current_layer)
            current_layer = hidden_layer

        # self.code_result = Dense(code_size, activation='tanh', activity_regularizer=l1(10e-7))(current_layer)
        self.code_result = Dense(code_size, activation=output_activation, activity_regularizer=l1(10e-7))(current_layer)

        # self.input_img = Input(shape=(self.input_size,))
        # input_hidden_0 = Dense(hidden_size0, activation='relu')(self.input_img)
        # input_hidden_1 = Dense(hidden_size1, activation='relu')(input_hidden_0)
        # # input_hidden_1 = Dropout(.2)(input_hidden_1)
        # input_hidden_2 = Dense(hidden_size2, activation='relu')(input_hidden_1)
        # input_hidden_3 = Dense(hidden_size3, activation='relu')(input_hidden_2)
        # # input_hidden_3 = Dropout(.2)(input_hidden_3)
        # input_hidden_4 = Dense(hidden_size4, activation='relu')(input_hidden_3)
        # input_hidden_5 = Dense(hidden_size5, activation='relu')(input_hidden_4)
        # # input_hidden_5 = Dropout(.2)(input_hidden_5)
        # input_hidden_6 = Dense(hidden_size6, activation='relu')(input_hidden_5)
        # input_hidden_7 = Dense(hidden_size7, activation='relu')(input_hidden_6)
        # # activity regularizer for sparsity constraints
        # self.code_result = Dense(code_size, activation='tanh', activity_regularizer=l1(10e-7))(input_hidden_7)

        decoder_layer_sizes = np.flip(decoder_layer_sizes)

        # self.code_input = Input(shape=(code_size,))
        current_layer = self.code_result
        for hidden_layer_size in decoder_layer_sizes:
            hidden_layer = Dense(hidden_layer_size, activation='relu')(current_layer)
            current_layer = hidden_layer

        # output_hidden_7 = Dense(hidden_size7, activation='relu')(self.code_input)
        # output_hidden_6 = Dense(hidden_size6, activation='relu')(output_hidden_7)
        # output_hidden_5 = Dense(hidden_size5, activation='relu')(output_hidden_6)
        # output_hidden_4 = Dense(hidden_size4, activation='relu')(output_hidden_5)
        # output_hidden_3 = Dense(hidden_size3, activation='relu')(output_hidden_4)
        # output_hidden_2 = Dense(hidden_size2, activation='relu')(output_hidden_3)
        # output_hidden_1 = Dense(hidden_size1, activation='relu')(output_hidden_2)
        # output_hidden_0 = Dense(hidden_size0, activation='relu')(output_hidden_1)
        # self.output_img = Dense(self.input_size, activation='tanh')(output_hidden_0)
        # self.output_spike = Dense(self.input_size, activation='tanh')(current_layer)
        self.output_spike = Dense(self.input_size, activation=output_activation)(current_layer)

        self.autoencoder = Model(self.input_spike, self.output_spike)

        self.encoder = Model(self.input_spike, self.code_result)

    def pre_train(self, training_data, autoencoder_layer_sizes, epochs):
        encoder_layer_weights = []
        decoder_layer_weights = []

        current_training_data = training_data
        current_input_size = self.input_size

        for code_size in autoencoder_layer_sizes:
            input = Input(shape=(current_input_size,))
            code_out = Dense(code_size, activation='tanh', activity_regularizer=l1(10e-8))(input)
            code_in = Input(shape=(code_size,))
            output = Dense(current_input_size, activation='tanh')(code_in)

            encoder = Model(input, code_out)
            decoder = Model(code_in, output)

            input = Input(shape=(current_input_size,))
            code = encoder(input)
            decoded = decoder(code)

            pretrained = Model(input, decoded)


            # autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
            pretrained.compile(optimizer='adam', loss=self.loss_function)
            pretrained.fit(current_training_data, current_training_data, epochs=epochs)

            encoder_layer_weights.append(pretrained.get_weights()[0])
            encoder_layer_weights.append(pretrained.get_weights()[1])
            decoder_layer_weights.append(pretrained.get_weights()[3])
            decoder_layer_weights.append(pretrained.get_weights()[2])

            current_input_size = code_size
            current_training_data = get_codes(current_training_data, encoder)

        decoder_layer_weights = decoder_layer_weights[::-1]
        encoder_layer_weights.extend(decoder_layer_weights)
        return encoder_layer_weights

    def train(self, training_data, epochs=50, verbose="auto", learning_rate=0.001):
        # autoencoder = Model(input_img, output_img)
        # autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
        # autoencoder.fit(training_data, training_data, epochs=20)

        # autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
        opt = keras.optimizers.Adam(learning_rate=learning_rate)
        self.autoencoder.compile(optimizer=opt, loss='mse')
        self.autoencoder.fit(training_data, training_data, epochs=epochs, verbose=verbose)

        # plot_model(encoder, "./figures/autoencoder/model_encoder.png", show_shapes=True)
        # plot_model(decoder, "./figures/autoencoder/model_decoder.png", show_shapes=True)
        # plot_model(autoencoder, "./figures/autoencoder/model_autoencoder.png", show_shapes=True)

    #     return self.encoder, self.decoder
    #
    def return_encoder(self):
        return self.encoder, self.autoencoder

    def return_autoencoder(self):
        return self.autoencoder
