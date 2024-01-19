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
from torch.utils.data import  Subset
from sklearn.model_selection import KFold

from dataset import TextDataset
from train import train
from predict import predict


def parse_argument() -> argparse.Namespace:
    """read in arguments from command line, all configuration are put within one yaml file"""
    parser = argparse.ArgumentParser(description="cat vs dog classifier")
    parser.add_argument("--config", type=str, default="./config/default.yaml")

    args = parser.parse_args()
    return args


def init_logging() -> str:
    """initialize logging setting"""
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.basicConfig(
        level=logging.DEBUG,
        filename="./log/log" + current_time + ".log",
        format="%(asctime)s %(message)s",
        datefmt="%I:%M:%S ",
    )
    return current_time


def save_model(model: torch.nn.Module, model_path: str="./models/test.pkl") -> None:
    print("model saved.")
    logging.info("model saved as " + model_path)
    torch.save(model.state_dict(), model_path)

def load_model(model:torch.nn.Module, model_path: str="./models/test.pkl")->torch.nn.Module:
    # 加载训练好的模型
    model.load_state_dict(
        torch.load(model_path, map_location=torch.device("cpu"))
    )
    return model

def cross_validation(
    dataset_dir: str,
    num_folds: int = 4,
    batch_size: int = 16,
    device="cuda",
    epochs: int = 20,
    learning_rate: float = 0.001,
    flag_debug:bool=False,
) -> torch.nn.Module | None:
    chn_news_dataset = TextDataset(file_path=dataset_dir)
    logging.info("dataset size:%d" % (len(chn_news_dataset)))

    print(f"cross validation with {num_folds} folds")
    kf = KFold(n_splits=num_folds, shuffle=True)

    best_f1: float = 0.0
    best_model: torch.nn.Module | None = None

    for fold_idx, (train_indices, test_indices) in enumerate(kf.split(chn_news_dataset)):
        print(f"Fold {fold_idx + 1}:")
        logging.info(f"Fold {fold_idx + 1}:")
        
        train_dataset = Subset(chn_news_dataset, train_indices)
        test_dataset = Subset(chn_news_dataset, test_indices)
        logging.info(f"dataset size: train {len(train_dataset)} test {len(test_dataset)}")
        
        # Training loop
        logging.info("training loop")
        my_model = train(
            train_dataset,
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate,
            device=device,
        )
        
        # Validation loop
        logging.info("validation loop")
        new_f1_score = predict(
            my_model,
            test_dataset,
            batch_size=batch_size,
            epochs=epochs,
            device=device,
        )
        print("f1 score macro: %0.5f" % (new_f1_score))
        if new_f1_score > best_f1:
            best_model = my_model
        if flag_debug==True:
            break
    return best_model

def main() -> None:
    """shell function for all operations needed for a training-testing process."""
    os.chdir(sys.path[0])
    warnings.filterwarnings("ignore")

    start_time=init_logging()
    args = parse_argument()
    try:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
    except IOError as e:
        raise IOError("config file not found")

    random_seed = int(config["seed"])
    train_path = config["train_path"]
    test_path = config["test_path"]
    num_folds = int(config["nfold"])
    batch_size = int(config["batch_size"])
    epochs = int(config["epochs"])
    learning_rate = float(config["learning_rate"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.info(config)
    print(config)

    torch.manual_seed(random_seed)
    '''
    # cross validation is omitted since a test dataset is given already
    best_model = cross_validation(
        train_path, num_folds, batch_size, device, epochs, learning_rate
    )
    '''
    # Create the dataset instance
    train_dataset = TextDataset(file_path=train_path)
    logging.info("training dataset size:%d" % (len(train_dataset)))
    test_dataset = TextDataset(file_path=test_path)
    logging.info("test dataset size:%d" % (len(test_dataset)))
        
    # Training loop
    logging.info("training part")
    my_model = train(
        train_dataset,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        device=device,
    )
    save_model(my_model,"./models/"+start_time+".pkl")
    
    # Validation loop
    logging.info("validation part")
    new_f1_score = predict(
        my_model,
        test_dataset,
        batch_size=batch_size,
        epochs=epochs,
        device=device,
    )
    print("f1 score macro on test datset: %0.5f" % (new_f1_score))
    logging.info("process completed")


if __name__ == "__main__":
    main()
