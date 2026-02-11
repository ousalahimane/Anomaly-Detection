from base import BaseModel
import os
import numpy as np
import matplotlib.pylab as plt
from matplotlib.pyplot import savefig
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


class VAEmodel(BaseModel):
    def __init__(self, config):
        super(VAEmodel, self).__init__(config)
        self.input_dims = self.config['l_win'] * self.config['n_channel']

        self.define_iterator()
        self.build_model()
        self.define_loss()
        self.training_variables()
        self.compute_gradients()
        self.init_saver()

    def define_iterator(self):
        self.original_signal = tf.compat.v1.placeholder(
            tf.float32, [None, self.config['l_win'], self.config['n_channel']]
        )
        self.seed = tf.compat.v1.placeholder(tf.int64, shape=())
        self.dataset = tf.data.Dataset.from_tensor_slices(self.original_signal)
        self.dataset = self.dataset.shuffle(buffer_size=60000, seed=self.seed)
        self.dataset = self.dataset.repeat(8000)
        self.dataset = self.dataset.batch(self.config['batch_size'], drop_remainder=False)
        self.iterator = tf.compat.v1.data.make_initializable_iterator(self.dataset)
        self.input_image = self.iterator.get_next()
        self.code_input = tf.compat.v1.placeholder(tf.float32, [None, self.config['code_size']])
        self.is_code_input = tf.compat.v1.placeholder(tf.bool)
        self.sigma2_offset = tf.constant(self.config['sigma2_offset'])

    def build_model(self):
        init = tf.keras.initializers.GlorotUniform()
        input_tensor = tf.expand_dims(self.original_signal, -1)

        # ---------- Encoder ----------
        if self.config['l_win'] == 24:
          with tf.compat.v1.variable_scope('encoder'):
            conv_1 = tf.keras.layers.Conv2D(
                filters=int(self.config['num_hidden_units'] / 16),
                kernel_size=(3, self.config['n_channel']),
                strides=(2, 1),
                padding='same',
                activation=tf.nn.leaky_relu,
                kernel_initializer=init
            )(tf.pad(input_tensor, [[0,0],[4,4],[0,0],[0,0]], "SYMMETRIC"))

            conv_2 = tf.keras.layers.Conv2D(
                filters=int(self.config['num_hidden_units'] / 8),
                kernel_size=(3, self.config['n_channel']),
                strides=(2, 1),
                padding='same',
                activation=tf.nn.leaky_relu,
                kernel_initializer=init
            )(conv_1)

            conv_3 = tf.keras.layers.Conv2D(
                filters=int(self.config['num_hidden_units'] / 4),
                kernel_size=(3, self.config['n_channel']),
                strides=(2, 1),
                padding='same',
                activation=tf.nn.leaky_relu,
                kernel_initializer=init
            )(conv_2)

            conv_4 = tf.keras.layers.Conv2D(
                filters=int(self.config['num_hidden_units']),
                kernel_size=(4, self.config['n_channel']),
                strides=1,
                padding='valid',
                activation=tf.nn.leaky_relu,
                kernel_initializer=init
            )(conv_3)

        elif self.config['l_win'] == 48:
            conv_1 = tf.keras.layers.Conv2D(
                filters=int(self.config['num_hidden_units'] / 16),
                kernel_size=(3, self.config['n_channel']),
                strides=(2, 1),
                padding='same',
                activation=tf.nn.leaky_relu,
                kernel_initializer=init
            )(input_tensor)

            conv_2 = tf.keras.layers.Conv2D(
                filters=int(self.config['num_hidden_units'] / 8),
                kernel_size=(3, self.config['n_channel']),
                strides=(2, 1),
                padding='same',
                activation=tf.nn.leaky_relu,
                kernel_initializer=init
            )(conv_1)

            conv_3 = tf.keras.layers.Conv2D(
                filters=int(self.config['num_hidden_units'] / 4),
                kernel_size=(3, self.config['n_channel']),
                strides=(2, 1),
                padding='same',
                activation=tf.nn.leaky_relu,
                kernel_initializer=init
            )(conv_2)

            conv_4 = tf.keras.layers.Conv2D(
                filters=int(self.config['num_hidden_units']),
                kernel_size=(6, self.config['n_channel']),
                strides=1,
                padding='valid',
                activation=tf.nn.leaky_relu,
                kernel_initializer=init
            )(conv_3)

        elif self.config['l_win'] == 144:
            conv_1 = tf.keras.layers.Conv2D(
                filters=int(self.config['num_hidden_units'] / 16),
                kernel_size=(3, self.config['n_channel']),
                strides=(4, 1),
                padding='same',
                activation=tf.nn.leaky_relu,
                kernel_initializer=init
            )(input_tensor)

            conv_2 = tf.keras.layers.Conv2D(
                filters=int(self.config['num_hidden_units'] / 8),
                kernel_size=(3, self.config['n_channel']),
                strides=(4, 1),
                padding='same',
                activation=tf.nn.leaky_relu,
                kernel_initializer=init
            )(conv_1)

            conv_3 = tf.keras.layers.Conv2D(
                filters=int(self.config['num_hidden_units'] / 4),
                kernel_size=(3, self.config['n_channel']),
                strides=(3, 1),
                padding='same',
                activation=tf.nn.leaky_relu,
                kernel_initializer=init
            )(conv_2)

            conv_4 = tf.keras.layers.Conv2D(
                filters=int(self.config['num_hidden_units']),
                kernel_size=(3, self.config['n_channel']),
                strides=1,
                padding='valid',
                activation=tf.nn.leaky_relu,
                kernel_initializer=init
            )(conv_3)

        encoded_signal = tf.keras.layers.Flatten()(conv_4)
        encoded_signal = tf.keras.layers.Dense(
            units=self.config['code_size'] * 4,
            activation=tf.nn.leaky_relu,
            kernel_initializer=init
        )(encoded_signal)

        self.code_mean = tf.keras.layers.Dense(
            units=self.config['code_size'],
            activation=None,
            kernel_initializer=init,
            name='code_mean'
        )(encoded_signal)

        self.code_std_dev = tf.keras.layers.Dense(
            units=self.config['code_size'],
            activation=tf.nn.relu,
            kernel_initializer=init,
            name='code_std_dev'
        )(encoded_signal)
        self.code_std_dev = self.code_std_dev + 1e-2

        mvn = tfd.MultivariateNormalDiag(loc=self.code_mean, scale_diag=self.code_std_dev)
        self.code_sample = mvn.sample()
        print("finish encoder: \n{}".format(self.code_sample))

        # ---------- Decoder ----------
        with tf.compat.v1.variable_scope('decoder'):
          encoded = tf.cond(self.is_code_input,
                            lambda: self.code_input,
                            lambda: self.code_sample)

          decoded_1 = tf.keras.layers.Dense(
              units=self.config['num_hidden_units'],
              activation=tf.nn.leaky_relu,
              kernel_initializer=init
          )(encoded)
          decoded_1 = tf.reshape(decoded_1, [-1, 1, 1, self.config['num_hidden_units']])

          # Pour le decoder, tu remplaces tf.layers.conv2d par tf.keras.layers.Conv2D
          # et tf.nn.depth_to_space reste le même
          # Exemple pour l_win = 48
          if self.config['l_win'] == 48:
              # On projette directement vers un tenseur (48, n_channel, 1)
              decoded_2 = tf.keras.layers.Conv2D(
                  filters=self.config['num_hidden_units'] // 2,  # ex: 256
                  kernel_size=(3, 1),
                  padding='same',
                  activation=tf.nn.leaky_relu,
                  kernel_initializer=init
              )(decoded_1)

              decoded_3 = tf.keras.layers.UpSampling2D(size=(2, 1))(decoded_2)  # 1 -> 2
              decoded_3 = tf.keras.layers.Conv2D(
                  filters=self.config['num_hidden_units'] // 4,  # ex: 128
                  kernel_size=(3, 1),
                  padding='same',
                  activation=tf.nn.leaky_relu,
                  kernel_initializer=init
              )(decoded_3)

              decoded_4 = tf.keras.layers.UpSampling2D(size=(2, 1))(decoded_3)  # 2 -> 4
              decoded_4 = tf.keras.layers.Conv2D(
                  filters=self.config['num_hidden_units'] // 8,  # ex: 64
                  kernel_size=(3, 1),
                  padding='same',
                  activation=tf.nn.leaky_relu,
                  kernel_initializer=init
              )(decoded_4)

              decoded_5 = tf.keras.layers.UpSampling2D(size=(2, 1))(decoded_4)  # 4 -> 8
              decoded_5 = tf.keras.layers.Conv2D(
                  filters=self.config['num_hidden_units'] // 16,  # ex: 32
                  kernel_size=(3, 1),
                  padding='same',
                  activation=tf.nn.leaky_relu,
                  kernel_initializer=init
              )(decoded_5)

              # Maintenant on "étire" la hauteur jusqu'à 48 avec UpSampling (8 -> 48)
              decoded_6 = tf.keras.layers.UpSampling2D(size=(6, 1))(decoded_5)  # 8*6 = 48

              # Dernière couche: sortie (batch, 48, 1, n_channel)
              decoded = tf.keras.layers.Conv2D(
                  filters=self.config['n_channel'],   # 3
                  kernel_size=(5, 1),
                  padding='same',
                  activation=None,
                  kernel_initializer=init
              )(decoded_6)

              # decoded: (batch, 48, 1, 3)  -> (batch, 48, 3)
              self.decoded = tf.squeeze(decoded, axis=2)


        # ---------- sigma2 ----------
        with tf.compat.v1.variable_scope('sigma2_dataset'):
          sigma = tf.Variable(tf.cast(self.config['sigma'], tf.float32),
                              dtype=tf.float32,
                              trainable=self.config['TRAIN_sigma']==1)
          self.sigma2 = tf.square(sigma) + self.sigma2_offset
          print("sigma2: \n{}\n")

class lstmKerasModel:
  def __init__(self, data):
    self.data = data 

  def create_lstm_model(self, config):
      lstm_input = tf.keras.layers.Input(shape=(config['l_seq']-1, config['code_size']))
      LSTM1 = tf.keras.layers.LSTM(config['num_hidden_units_lstm'], return_sequences=True)(lstm_input)
      LSTM2 = tf.keras.layers.LSTM(config['num_hidden_units_lstm'], return_sequences=True)(LSTM1)
      lstm_output = tf.keras.layers.LSTM(config['code_size'], return_sequences=True, activation=None)(LSTM2)
      lstm_model = tf.keras.Model(lstm_input, lstm_output)
      lstm_model.compile(
          optimizer=tf.keras.optimizers.Adam(learning_rate=config['learning_rate_lstm']),
          loss='mse',
          metrics=['mse'],
          run_eagerly=True
      )

      return lstm_model

  def produce_embeddings(self, config, model_vae, data, sess):

      # ===== TRAIN =====
      self.embedding_lstm_train = np.zeros(
          (data.n_train_lstm, config['l_seq'], config['code_size'])
      )

      for i in range(data.n_train_lstm):
          feed_dict = {
              model_vae.original_signal: data.train_set_lstm['data'][i],
              model_vae.is_code_input: False,
              model_vae.code_input: np.zeros((1, config['code_size']))
          }
          self.embedding_lstm_train[i] = sess.run(
              model_vae.code_mean, feed_dict=feed_dict
          )

      print("Finish processing the embeddings of the entire dataset.")
      print("The first a few embeddings are\n{}".format(
          self.embedding_lstm_train[0, 0:5])
      )

      self.x_train = self.embedding_lstm_train[:, :config['l_seq'] - 1]
      self.y_train = self.embedding_lstm_train[:, 1:]

      #conversion NumPy CORRECTE
      self.x_train_np = np.array(self.x_train)
      self.y_train_np = np.array(self.y_train)

      # ===== TEST =====
      self.embedding_lstm_test = np.zeros(
          (data.n_val_lstm, config['l_seq'], config['code_size'])
      )

      for i in range(data.n_val_lstm):
          feed_dict = {
              model_vae.original_signal: data.val_set_lstm['data'][i],
              model_vae.is_code_input: False,
              model_vae.code_input: np.zeros((1, config['code_size']))
          }
          self.embedding_lstm_test[i] = sess.run(
              model_vae.code_mean, feed_dict=feed_dict
          )

      self.x_test = self.embedding_lstm_test[:, :config['l_seq'] - 1]
      self.y_test = self.embedding_lstm_test[:, 1:]

      self.x_test_np = np.array(self.x_test)
      self.y_test_np = np.array(self.y_test)


  def load_model(self, lstm_model, config, checkpoint_path):
      if os.path.isfile(checkpoint_path):
          lstm_model.load_weights(checkpoint_path)
          print("LSTM weights loaded from:", checkpoint_path)
      else:
          print("No LSTM weights file found at:", checkpoint_path)


  def train(self, config, lstm_model, cp_callback, sess=None):

    # sécurité : vérifier qu'on est bien sur du NumPy
    import numpy as np
    assert isinstance(self.x_train_np, np.ndarray)
    assert isinstance(self.y_train_np, np.ndarray)
    print("x_train_np:", type(self.x_train_np), self.x_train_np.shape)
    print("y_train_np:", type(self.y_train_np), self.y_train_np.shape)
    print("x_test_np :", type(self.x_test_np),  self.x_test_np.shape)
    print("y_test_np :", type(self.y_test_np),  self.y_test_np.shape)

    history = lstm_model.fit(
        self.x_train, self.y_train,
        batch_size=config['batch_size_lstm'],
        epochs=config['num_epochs_lstm'],
        validation_data=(self.x_test, self.y_test),
        callbacks=[cp_callback]
    )

    return history

  def plot_reconstructed_lt_seq(self, idx_test, config, model_vae, sess, data, lstm_embedding_test):
    feed_dict_vae = {model_vae.original_signal: np.zeros((config['l_seq'], config['l_win'], config['n_channel'])),
                     model_vae.is_code_input: True,
                     model_vae.code_input: self.embedding_lstm_test[idx_test]}
    decoded_seq_vae = np.squeeze(sess.run(model_vae.decoded, feed_dict=feed_dict_vae))
    print("Decoded seq from VAE: {}".format(decoded_seq_vae.shape))

    feed_dict_lstm = {model_vae.original_signal: np.zeros((config['l_seq'] - 1, config['l_win'], config['n_channel'])),
                      model_vae.is_code_input: True,
                      model_vae.code_input: lstm_embedding_test[idx_test]}
    decoded_seq_lstm = np.squeeze(sess.run(model_vae.decoded, feed_dict=feed_dict_lstm))
    print("Decoded seq from lstm: {}".format(decoded_seq_lstm.shape))

    fig, axs = plt.subplots(config['n_channel'], 2, figsize=(15, 4.5 * config['n_channel']), edgecolor='k')
    fig.subplots_adjust(hspace=.4, wspace=.4)
    axs = axs.ravel()
    for j in range(config['n_channel']):
      for i in range(2):
        axs[i + j * 2].plot(np.arange(0, config['l_seq'] * config['l_win']),
                            np.reshape(data.val_set_lstm['data'][idx_test, :, :, j],
                                       (config['l_seq'] * config['l_win'])))
        axs[i + j * 2].grid(True)
        axs[i + j * 2].set_xlim(0, config['l_seq'] * config['l_win'])
        axs[i + j * 2].set_xlabel('samples')
      if config['n_channel'] == 1:
        axs[0 + j * 2].plot(np.arange(0, config['l_seq'] * config['l_win']),
                            np.reshape(decoded_seq_vae, (config['l_seq'] * config['l_win'])), 'r--')
        axs[1 + j * 2].plot(np.arange(config['l_win'], config['l_seq'] * config['l_win']),
                            np.reshape(decoded_seq_lstm, ((config['l_seq'] - 1) * config['l_win'])), 'g--')
      else:
        axs[0 + j * 2].plot(np.arange(0, config['l_seq'] * config['l_win']),
                            np.reshape(decoded_seq_vae[:, :, j], (config['l_seq'] * config['l_win'])), 'r--')
        axs[1 + j * 2].plot(np.arange(config['l_win'], config['l_seq'] * config['l_win']),
                            np.reshape(decoded_seq_lstm[:, :, j], ((config['l_seq'] - 1) * config['l_win'])), 'g--')
      axs[0 + j * 2].set_title('VAE reconstruction - channel {}'.format(j))
      axs[1 + j * 2].set_title('LSTM reconstruction - channel {}'.format(j))
      for i in range(2):
        axs[i + j * 2].legend(('ground truth', 'reconstruction'))
      savefig(config['result_dir'] + "lstm_long_seq_recons_{}.pdf".format(idx_test))
      fig.clf()
      plt.close()

  def plot_lstm_embedding_prediction(self, idx_test, config, model_vae, sess, data, lstm_embedding_test):
    self.plot_reconstructed_lt_seq(idx_test, config, model_vae, sess, data, lstm_embedding_test)

    fig, axs = plt.subplots(2, config['code_size'] // 2, figsize=(15, 5.5), edgecolor='k')
    fig.subplots_adjust(hspace=.4, wspace=.4)
    axs = axs.ravel()
    for i in range(config['code_size']):
      axs[i].plot(np.arange(1, config['l_seq']), np.squeeze(self.embedding_lstm_test[idx_test, 1:, i]))
      axs[i].plot(np.arange(1, config['l_seq']), np.squeeze(lstm_embedding_test[idx_test, :, i]))
      axs[i].set_xlim(1, config['l_seq'] - 1)
      axs[i].set_ylim(-2.5, 2.5)
      axs[i].grid(True)
      axs[i].set_title('Embedding dim {}'.format(i))
      axs[i].set_xlabel('windows')
      if i == config['code_size'] - 1:
        axs[i].legend(('VAE\nembedding', 'LSTM\nembedding'))
    savefig(config['result_dir'] + "lstm_seq_embedding_{}.pdf".format(idx_test))
    fig.clf()
    plt.close()
