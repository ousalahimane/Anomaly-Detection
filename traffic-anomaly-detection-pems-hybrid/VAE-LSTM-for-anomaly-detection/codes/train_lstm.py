import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf

from data_loader import DataGenerator
from models import VAEmodel, lstmKerasModel
from utils import process_config, create_dirs, get_args


def main():
    args = get_args()
    config = process_config(args.config)

    # dossiers
    if "checkpoint_dir_lstm" not in config:
        raise KeyError("Ajoute 'checkpoint_dir_lstm' dans ton config JSON")
    create_dirs([config["checkpoint_dir_lstm"], config["result_dir"]])
    os.makedirs(config["checkpoint_dir_lstm"], exist_ok=True)

    # Data (doit déjà charger ton NPZ via data_path)
    data = DataGenerator(config)

    # =========================================================
    # 1) VAE en TF1 (graph + session séparés)
    # =========================================================
    vae_graph = tf.Graph()
    with vae_graph.as_default():
        sess = tf.compat.v1.Session(
            graph=vae_graph,
            config=tf.compat.v1.ConfigProto(log_device_placement=False)
        )

        model_vae = VAEmodel(config)
        sess.run(tf.compat.v1.global_variables_initializer())

        # charger checkpoint VAE entraîné
        model_vae.load(sess)

        # produire embeddings à partir des fenêtres LSTM
        lstm_model = lstmKerasModel(data)
        lstm_model.produce_embeddings(config, model_vae, data, sess)

        sess.close()

    print("Embeddings générés")

    # =========================================================
    # 2) LSTM en Keras (TF2) - session TF1 déjà fermée
    # =========================================================
    tf.keras.backend.clear_session()

    lstm_nn_model = lstm_model.create_lstm_model(config)
    lstm_nn_model.summary()

    checkpoint_path = os.path.join(config["checkpoint_dir_lstm"], "cp.weights.h5")
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        verbose=1
    )

    if int(config.get("num_epochs_lstm", 0)) > 0:
        lstm_model.train(config, lstm_nn_model, cp_callback, sess=None)

    print("LSTM entraîné et sauvegardé:", checkpoint_path)


if __name__ == "__main__":
    main()
