import os
import tensorflow as tf

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

tf.compat.v1.disable_eager_execution()

from data_loader import DataGenerator
from models import VAEmodel
from trainers import vaeTrainer
from utils import process_config, create_dirs, get_args, save_config


def main():
    args = get_args()
    config = process_config(args.config)

    # IMPORTANT: checkpoint_dir doit exister dans config
    create_dirs([config["result_dir"], config["checkpoint_dir"]])
    save_config(config)

    sess = tf.compat.v1.Session(
        config=tf.compat.v1.ConfigProto(log_device_placement=False)
    )

    data = DataGenerator(config)
    model_vae = VAEmodel(config)
    trainer = vaeTrainer(sess, model_vae, data, config)

    # init variables
    sess.run(tf.compat.v1.global_variables_initializer())

    # load checkpoint if exists
    latest_ckpt = tf.train.latest_checkpoint(config["checkpoint_dir"])
    if latest_ckpt:
        print("Checkpoint trouvé -> chargement VAE:", latest_ckpt)
        model_vae.load(sess)
    else:
        print("Aucun checkpoint -> entraînement depuis zéro")

    if config["TRAIN_VAE"] and config["num_epochs_vae"] > 0:
        trainer.train()
        model_vae.save(sess)

    print("VAE prêt")


if __name__ == "__main__":
    main()
