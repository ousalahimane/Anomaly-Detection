from base import BaseDataGenerator
import numpy as np
import matplotlib.pylab as plt
from matplotlib.pyplot import savefig
from glob import glob
from numpy.lib.stride_tricks import sliding_window_view

class DataGenerator(BaseDataGenerator):
  def __init__(self, config):
    super(DataGenerator, self).__init__(config)

    if self.config.get("multi_npz", False):
      self.load_multi_npz(self.config.get("data_glob"), self.config['y_scale'])
    else:
      self.load_NAB_dataset(self.config['dataset'], self.config['y_scale'])

  def load_multi_npz(self, data_glob, y_scale=6):
    do_lstm = int(self.config.get("TRAIN_LSTM", 0)) == 1
    if data_glob is None:
      raise ValueError("multi_npz=True mais data_glob est manquant dans la config.")

    paths = sorted(glob(data_glob))
    if not paths:
      raise FileNotFoundError(f"Aucun .npz trouvé avec le pattern: {data_glob}")

    max_st = int(self.config.get("max_stations", 0))
    if max_st > 0:
      paths = paths[:max_st]

    l_win = int(self.config['l_win'])
    l_seq = int(self.config['l_seq'])
    batch_size = int(self.config['batch_size'])
    n_channel = int(self.config.get('n_channel', 3))

    all_vae_windows = []
    all_lstm_seq = []

    used = 0
    skipped = 0

    for p in paths:
      data = np.load(p, allow_pickle=True)

      # --- charger X_norm multivarié ---
      if 'X_norm' in data.files:
        Xn = data['X_norm']
      else:
        # fallback (rare si tu as bien sauvegardé)
        train_m = data['train_m']
        train_std = data['train_std']
        Xn = (data['X'] - train_m) / (train_std + 1e-8)

      # force (T,C)
      if Xn.ndim == 1:
        Xn = Xn.reshape(-1, 1)

      Xn = Xn[:, :n_channel]
      T = Xn.shape[0]

      # --- split train/test depuis idx_split ---
      idx_split = np.asarray(data['idx_split']).reshape(-1)
      split_start_test = int(idx_split[1]) if len(idx_split) > 1 else int(0.7*T)

      training = Xn[:split_start_test]
      if training.shape[0] < l_win:
        skipped += 1
        continue

      n_train_sample = training.shape[0]
      n_train_vae = n_train_sample - l_win + 1
      if n_train_vae <= 0:
          skipped += 1
          continue

      # --- rolling windows VAE (VERSION RAPIDE) ---
      rolling_windows = sliding_window_view(
          training,
          (l_win, n_channel)
      )[:, 0, :, :].astype(np.float32)

      all_vae_windows.append(rolling_windows)

      # --- sequences LSTM (SEULEMENT si demandé) ---
      if do_lstm and n_train_vae >= l_seq:
          n_train_lstm = n_train_vae - l_seq + 1
          seq = np.zeros((n_train_lstm, l_seq, l_win, n_channel), dtype=np.float32)
          for i in range(n_train_lstm):
              seq[i] = rolling_windows[i:i+l_seq]
          all_lstm_seq.append(seq)

      used += 1

    if used == 0:
      raise ValueError("Aucune station utilisable (trop courte après split / l_win).")

    # concaténer sur l’axe des exemples
    vae_windows = np.concatenate(all_vae_windows, axis=0)

    if do_lstm:
        if len(all_lstm_seq) == 0:
            raise ValueError("TRAIN_LSTM=1 mais aucune station n'a assez de données (n_train_vae < l_seq).")
        lstm_seq = np.concatenate(all_lstm_seq, axis=0)
    else:
        lstm_seq = None
    # split train/val global (mélange d’exemples de toutes stations)
    idx_train, idx_val, self.n_train_vae, self.n_val_vae = self.separate_train_and_val_set(len(vae_windows))
    self.train_set_vae = dict(data=vae_windows[idx_train])
    self.val_set_vae   = dict(data=vae_windows[idx_val])
    self.test_set_vae  = dict(data=vae_windows[idx_val[:batch_size]])

    if do_lstm:
        idx_train, idx_val, self.n_train_lstm, self.n_val_lstm = self.separate_train_and_val_set(len(lstm_seq))
        self.train_set_lstm = dict(data=lstm_seq[idx_train])
        self.val_set_lstm   = dict(data=lstm_seq[idx_val])


    print(f"[MULTI] Stations utilisées: {used}, ignorées: {skipped}")
    print("[MULTI] VAE windows:", vae_windows.shape)
    if do_lstm:
        print("[MULTI] LSTM seq    :", lstm_seq.shape)
    else:
        print("[MULTI] LSTM seq    : skipped (TRAIN_LSTM=0)")

  def load_NAB_dataset(self, dataset, y_scale=6):
    default_path = '../datasets/NAB-known-anomaly/' + dataset + '.npz'
    path = self.config.get('data_path', default_path)
    data = np.load(path)

    train_m = data['train_m']
    train_std = data['train_std']

    # rendre tm/ts scalaires (évite TypeError avec {:.4f})
    tm = float(np.asarray(train_m).reshape(-1)[0])
    ts = float(np.asarray(train_std).reshape(-1)[0])

    # 1) Lire la série normalisée
    if 'X_norm' in data.files:
      readings_norm = data['X_norm']
    else:
      readings_norm = (data['X'] - tm) / ts

    feature_idx = int(self.config.get('feature_idx', 0))

    # transformer en 1D si besoin
    if readings_norm.ndim == 2:
      readings_normalised_1d = readings_norm[:, feature_idx]
    else:
      readings_normalised_1d = readings_norm

    # 2) Plot normalisé
    fig, axs = plt.subplots(1, 1, figsize=(18, 4), edgecolor='k')
    fig.subplots_adjust(hspace=.4, wspace=.4)
    axs.plot(data['t'], readings_normalised_1d)

    if data['idx_split'][0] == 0:
      axs.plot(data['idx_split'][1] * np.ones(20), np.linspace(-y_scale, y_scale, 20), 'b-')
    else:
      for i in range(2):
        axs.plot(data['idx_split'][i] * np.ones(20), np.linspace(-y_scale, y_scale, 20), 'b-')

    # correction (au lieu de axs.plot(*np.ones(20), ...))
    axs.plot(0 * np.ones(20), np.linspace(-y_scale, y_scale, 20), 'b--')

    for j in range(len(data['idx_anomaly'])):
      axs.plot(data['idx_anomaly'][j] * np.ones(20), np.linspace(-y_scale, 0.75 * y_scale, 20), 'r--')

    axs.grid(True)
    axs.set_xlim(0, len(data['t']))
    axs.set_ylim(-y_scale, y_scale)
    axs.set_xlabel("timestamp (every {})".format(data['t_unit']))
    axs.set_ylabel("readings")
    axs.set_title("{} dataset\n(normalised by train mean {:.4f} and std {:.4f})".format(dataset, tm, ts))
    axs.legend(('data', 'train test set split', 'anomalies'))
    savefig(self.config['result_dir'] + '/raw_data_normalised.pdf')

    # 3) Construire "training" depuis X + idx_split
    split_start_test = int(np.asarray(data['idx_split']).reshape(-1)[1])

    # --- training multivarié (T, C) ---
    if 'X_norm' in data.files:
        Xn = data['X_norm']
    else:
        Xn = (data['X'] - tm) / ts

    training = Xn[:split_start_test]              # (T, F) ou (T,)
    if training.ndim == 1:
        training = training.reshape(-1, 1)        # (T, 1)

    C = int(self.config.get('n_channel', training.shape[1]))
    training = training[:, :C]                    # (T, C)

    # 4) Rolling windows VAE (multivarié)
    n_train_sample = training.shape[0]
    C = training.shape[1]
    n_train_vae = n_train_sample - self.config['l_win'] + 1
    if n_train_vae <= 0:
        raise ValueError(f"Train trop court: {n_train_sample} < l_win={self.config['l_win']}.")

    rolling_windows = np.zeros((n_train_vae, self.config['l_win'], C))
    for i in range(n_train_vae):
        rolling_windows[i] = training[i:i + self.config['l_win'], :]

    idx_train, idx_val, self.n_train_vae, self.n_val_vae = self.separate_train_and_val_set(n_train_vae)
    self.train_set_vae = dict(data=rolling_windows[idx_train])  # (batch, 48, 3)
    self.val_set_vae   = dict(data=rolling_windows[idx_val])
    self.test_set_vae  = dict(data=rolling_windows[idx_val[:self.config['batch_size']]])

    # 5) LSTM sequences (OVERLAP à partir des rolling windows)
    if n_train_vae < self.config['l_seq']:
        raise ValueError(f"n_train_vae={n_train_vae} < l_seq={self.config['l_seq']}")

    n_train_lstm = n_train_vae - self.config['l_seq'] + 1
    lstm_seq = np.zeros((n_train_lstm, self.config['l_seq'], self.config['l_win'], C))
    for i in range(n_train_lstm):
        lstm_seq[i] = rolling_windows[i:i + self.config['l_seq']]

    idx_train, idx_val, self.n_train_lstm, self.n_val_lstm = self.separate_train_and_val_set(n_train_lstm)
    self.train_set_lstm = dict(data=lstm_seq[idx_train])
    self.val_set_lstm   = dict(data=lstm_seq[idx_val])


  def plot_time_series(self, data, time, data_list):
    fig, axs = plt.subplots(1, 4, figsize=(18, 2.5), edgecolor='k')
    fig.subplots_adjust(hspace=.8, wspace=.4)
    axs = axs.ravel()
    for i in range(4):
      axs[i].plot(time / 60., data[:, i])
      axs[i].set_title(data_list[i])
      axs[i].set_xlabel('time (h)')
      axs[i].set_xlim((np.amin(time) / 60., np.amax(time) / 60.))
    savefig(self.config['result_dir'] + '/raw_training_set_normalised.pdf')
