import tensorflow_probability as tfp
import random
import numpy as np
import time
import matplotlib.pylab as plt
from matplotlib.pyplot import plot, savefig, figure
from utils import count_trainable_variables
tfd = tfp.distributions
import tensorflow as tf
from tensorflow.train import list_variables


class BaseDataGenerator:
  def __init__(self, config):
    self.config = config

  # separate training and val sets
  def separate_train_and_val_set(self, n_win):
    n_train = int(np.floor((n_win * 0.9)))
    n_val = n_win - n_train
    idx_train = random.sample(range(n_win), n_train)
    idx_val = list(set(idx_train) ^ set(range(n_win)))
    return idx_train, idx_val, n_train, n_val


class BaseModel:
  def __init__(self, config):
    self.config = config
    # init the global step
    self.init_global_step()
    # init the epoch counter
    self.init_cur_epoch()
    self.two_pi = tf.constant(2 * np.pi)

  # save function that saves the checkpoint in the path defined in the config file
  def save(self, sess):
    print("Saving model...")
    self.saver.save(sess, self.config['checkpoint_dir'],
                    self.global_step_tensor)
    print("Model saved.")

  # load latest checkpoint from the experiment path defined in the config file
  def load(self, sess):
      ckpt_dir = self.config["checkpoint_dir"]
      latest_checkpoint = tf.train.latest_checkpoint(ckpt_dir)
      print(f"checkpoint_dir at loading: {ckpt_dir}")
      if not latest_checkpoint:
          print("[WARN] No checkpoint found → nothing loaded.")
          return
      # Si on a construit un "partial saver" mais qu'il n'a aucune var à restore, on skip
      if getattr(self, "_has_restore_vars", False) is False:
          print("[WARN] Saver has no variables to restore → skip restore.")
          print("       (Tu es probablement dans un cas de mismatch total de noms.)")
          return
      print(f"Loading model checkpoint {latest_checkpoint} ...\n")
      self.saver.restore(sess, latest_checkpoint)
      print("Model loaded.")


  # initialize a tensorflow variable to use it as epoch counter
  def init_cur_epoch(self):
      with tf.compat.v1.variable_scope('cur_epoch'):
          self.cur_epoch_tensor = tf.compat.v1.Variable(0, trainable=False, name='cur_epoch')
          self.increment_cur_epoch_tensor = tf.compat.v1.assign(
              self.cur_epoch_tensor, self.cur_epoch_tensor + 1
          )

  # just initialize a tensorflow variable to use it as global step counter
  def init_global_step(self):
      with tf.compat.v1.variable_scope('global_step'):
          self.global_step_tensor = tf.compat.v1.Variable(0, trainable=False, name='global_step')
          self.increment_global_step_tensor = tf.compat.v1.assign(
              self.global_step_tensor, self.global_step_tensor + 1
          )

  def define_loss(self):
    with tf.name_scope("loss"):
      # KL divergence loss - analytical result
      KL_loss = 0.5 * (tf.reduce_sum(tf.square(self.code_mean), axis=1)
                      + tf.reduce_sum(tf.square(self.code_std_dev), axis=1)
                      - tf.reduce_sum(tf.math.log(tf.square(self.code_std_dev)), axis=1)
                      - self.config['code_size'])
      self.KL_loss = tf.reduce_mean(KL_loss)

      # norm 1 of standard deviation of the sample-wise encoder prediction
      self.std_dev_norm = tf.reduce_mean(self.code_std_dev, axis=0)

      weighted_reconstruction_error_dataset = tf.reduce_sum(
        tf.square(self.original_signal - self.decoded), [1, 2])
      weighted_reconstruction_error_dataset = tf.reduce_mean(weighted_reconstruction_error_dataset)
      self.weighted_reconstruction_error_dataset = weighted_reconstruction_error_dataset / (2 * self.sigma2)

      # least squared reconstruction error
      ls_reconstruction_error = tf.reduce_sum(
        tf.square(self.original_signal - self.decoded), [1, 2])
      self.ls_reconstruction_error = tf.reduce_mean(ls_reconstruction_error)

      # sigma regularisor - input elbo
      self.sigma_regularisor_dataset = self.input_dims / 2 * tf.math.log(self.sigma2)
      two_pi = self.input_dims / 2 * tf.constant(2 * np.pi)

      self.elbo_loss = two_pi + self.sigma_regularisor_dataset + \
                       0.5 * self.weighted_reconstruction_error_dataset + self.KL_loss

  def training_variables(self):
      # Récupère toutes les variables entraînables Keras/TF1
      self.train_vars_VAE = [v for v in tf.compat.v1.trainable_variables()
                            if 'encoder' in v.name or 'decoder' in v.name or 'sigma2_dataset' in v.name]

      # Vérification
      print("Trainable variables found:")
      for v in self.train_vars_VAE:
          print(v.name)

      # Calcul du nombre total de paramètres
      self.num_vars_total = sum(np.prod(v.shape.as_list()) for v in self.train_vars_VAE)
      print("Total number of trainable parameters in the VAE network is: {}".format(self.num_vars_total))

  def compute_gradients(self):
    self.lr = tf.compat.v1.placeholder(tf.float32, [])
    opt = tf.compat.v1.train.AdamOptimizer(learning_rate=self.lr, beta1=0.9, beta2=0.95)
    update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
    gvs_dataset = opt.compute_gradients(self.elbo_loss, var_list=self.train_vars_VAE)
    print('gvs for dataset: {}'.format(gvs_dataset))
    capped_gvs = [(self.ClipIfNotNone(grad), var) for grad, var in gvs_dataset]

    with tf.control_dependencies(update_ops):
      self.train_step_gradient = opt.apply_gradients(capped_gvs)
    print("Reach the definition of loss for VAE")

  def ClipIfNotNone(self, grad):
    if grad is None:
      return grad
    return tf.clip_by_value(grad, -1, 1)

  def init_saver(self):
        """
        Crée un Saver robuste:
        - Si checkpoint existe → restaure seulement l'intersection (partial restore)
        - Si intersection vide → fallback Saver standard + on skip restore dans load()
        """
        ckpt_dir = self.config["checkpoint_dir"]
        ckpt = tf.train.latest_checkpoint(ckpt_dir)

        # cas: aucun checkpoint → saver standard
        if ckpt is None:
            print("[WARN] Aucun checkpoint trouvé → Saver standard")
            self.saver = tf.compat.v1.train.Saver(max_to_keep=1)
            self._has_restore_vars = False
            return

        print("[INFO] Chargement partiel du checkpoint :", ckpt)

        # Variables présentes dans le checkpoint
        ckpt_vars = {name for name, _ in list_variables(ckpt)}

        # Variables du graphe (toutes)
        graph_vars = self.train_vars_VAE

        vars_to_restore = {}
        skipped = []

        for v in graph_vars:
            name = v.name.split(":")[0]
            if name in ckpt_vars:
                vars_to_restore[name] = v
            else:
                skipped.append(name)

        print(f"[INFO] Variables restaurées : {len(vars_to_restore)}")
        print(f"[INFO] Variables ignorées  : {len(skipped)}")
        if skipped:
            print("[INFO] Exemples ignorés :", skipped[:5])

        # IMPORTANT: si 0 variables communes → éviter l'erreur "No variables to save"
        if len(vars_to_restore) == 0:
            print("[WARN] 0 variable commune avec le checkpoint → Saver standard, restore SKIPPED.")
            self.saver = tf.compat.v1.train.Saver(max_to_keep=1)
            self._has_restore_vars = False
            return

        # Sinon, saver partiel OK
        self.saver = tf.compat.v1.train.Saver(var_list=vars_to_restore, max_to_keep=1)
        self._has_restore_vars = True

class BaseTrain:
  def __init__(self, sess, model, data, config):
    self.model = model
    self.config = config
    self.sess = sess
    self.data = data
    self.init = tf.compat.v1.group(
        tf.compat.v1.global_variables_initializer(),
        tf.compat.v1.local_variables_initializer()
    )
    self.sess.run(self.init)

    # keep a record of the training result
    self.train_loss = []
    self.val_loss = []
    self.train_loss_ave_epoch = []
    self.val_loss_ave_epoch = []
    self.recons_loss_train = []
    self.recons_loss_val = []
    self.KL_loss_train = []
    self.KL_loss_val = []
    self.sample_std_dev_train = []
    self.sample_std_dev_val = []
    self.iter_epochs_list = []
    self.test_sigma2 = []

  def train(self):
    self.start_time = time.time()
    for cur_epoch in range(0, self.config['num_epochs_vae'], 1):
      self.train_epoch()

      # compute current execution time
      self.current_time = time.time()
      elapsed_time = (self.current_time - self.start_time) / 60
      est_remaining_time = (
                                   self.current_time - self.start_time) / (cur_epoch + 1) * (
                                     self.config['num_epochs_vae'] - cur_epoch - 1)
      est_remaining_time = est_remaining_time / 60
      print("Already trained for {} min; Remaining {} min.".format(elapsed_time, est_remaining_time))
      self.sess.run(self.model.increment_cur_epoch_tensor)

  def save_variables_VAE(self):
    # save some variables for later inspection
    file_name = "{}{}-batch-{}-epoch-{}-code-{}-lr-{}.npz".format(self.config['result_dir'],
                                                                  self.config['exp_name'],
                                                                  self.config['batch_size'],
                                                                  self.config['num_epochs_vae'],
                                                                  self.config['code_size'],
                                                                  self.config['learning_rate_vae'])
    np.savez(file_name,
             iter_list_val=self.iter_epochs_list,
             train_loss=self.train_loss,
             val_loss=self.val_loss,
             n_train_iter=self.n_train_iter,
             n_val_iter=self.n_val_iter,
             recons_loss_train=self.recons_loss_train,
             recons_loss_val=self.recons_loss_val,
             KL_loss_train=self.KL_loss_train,
             KL_loss_val=self.KL_loss_val,
             num_para_all=self.model.num_vars_total,
             sigma2=self.test_sigma2)

  def plot_train_and_val_loss(self):
    # plot the training and validation loss over epochs
    plt.clf()
    figure(num=1, figsize=(8, 6))
    plot(self.train_loss, 'b-')
    plot(self.iter_epochs_list, self.val_loss_ave_epoch, 'r-')
    plt.legend(('training loss (total)', 'validation loss'))
    plt.title('training loss over iterations (val @ epochs)')
    plt.ylabel('total loss')
    plt.xlabel('iterations')
    plt.grid(True)
    savefig(self.config['result_dir'] + '/loss.png')

    # plot individual components of validation loss over epochs
    plt.clf()
    figure(num=1, figsize=(8, 6))
    plot(self.recons_loss_val, 'b-')
    plot(self.KL_loss_val, 'r-')
    plt.legend(('Reconstruction loss', 'KL loss'))
    plt.title('validation loss breakdown')
    plt.ylabel('loss')
    plt.xlabel('num of batch')
    plt.grid(True)
    savefig(self.config['result_dir'] + '/val-loss.png')

    # plot individual components of validation loss over epochs
    plt.clf()
    plt.close(1)
    plt.figure(num=1, figsize=(8, 6))

    plot(self.test_sigma2, 'b-')
    plt.title('sigma2 over training')
    plt.ylabel('sigma2')
    plt.xlabel('iter')
    plt.grid(True)
    savefig(self.config['result_dir'] + '/sigma2.png')
