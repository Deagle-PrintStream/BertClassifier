r"""
    file: main.py
    author: Swayee
    email:  suiyi_liu@mail.ustc.edu.cn
    time:   2023/12/29
    encoding:   utf-8
"""

import os, sys, warnings
import argparse, logging, datetime
import yaml

import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold

from dataset import TextDataset
from train import train
from predict import predict


def parse_argument() -> argparse.Namespace:
    """read in arguments from command line, all configuration are put within one yaml file"""
    parser = argparse.ArgumentParser(description="cat vs dog classifier")
    parser.add_argument("--config", type=str, default="./config/test.yaml")

    args = parser.parse_args()
    return args


def init_logging() -> None:
    """initialize logging setting"""
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.basicConfig(
        level=logging.DEBUG,
        filename="./log/log" + current_time + ".log",
        format="%(asctime)s %(message)s",
        datefmt="%I:%M:%S ",
    )


def save_model(model: torch.nn.Module, model_path: str) -> None:
    print("model saved.")
    logging.info("model saved as " + model_path)
    torch.save(model, model_path)


def main() -> None:
    """shell function for all operations needed for a training-validation-testing process."""
    os.chdir(sys.path[0])
    warnings.filterwarnings("ignore")

    init_logging()
    args = parse_argument()
    try:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
    except IOError as e:
        raise IOError("config file not found")

    random_seed = config["seed"]
    dataset_dir = config["dir"]
    num_folds = config["nfold"]
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    learning_rate = config["learning_rate"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.info(config)
    print(config)

    torch.manual_seed(random_seed)

    # Create the dataset instance
    chn_news_dataset = TextDataset(file_path=dataset_dir)
    logging.info("dataset size:%d" % (len(chn_news_dataset)))

    print("cross validation with {num_folds} folds")
    kf = KFold(n_splits=num_folds, shuffle=True)

    for fold_idx, (train_indices, val_indices) in enumerate(kf.split(chn_news_dataset)):
        print(f"Fold {fold_idx + 1}:")
        logging.info(f"Fold {fold_idx + 1}:")
        train_dataset = Subset(chn_news_dataset, train_indices)
        val_dataset = Subset(chn_news_dataset, val_indices)
        # train_loader = DataLoader(Subset(chn_news_dataset, train_indices), batch_size=batch_size, shuffle=True)
        # val_loader = DataLoader(Subset(chn_news_dataset, val_indices), batch_size=batch_size, shuffle=False)

        # Training loop
        logging.info("training loop")
        my_model = train(
            train_dataset,
            device=device,
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate,
        )
        # Validation loop
        logging.info("validation loop")
        new_f1_score = predict(
            my_model,
            val_dataset,
            device=device,
            batch_size=batch_size,
            epochs=epochs,
        )
    # save the model with best f1 score
    torch.save(my_model, "./models/" + ".pth")


if __name__ == "__main__":
    main()
