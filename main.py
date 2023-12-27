# coding: utf-8
# @File: main.py
# @Author: Swayee
# @Email: suiyiliu@mail.ustc.edu.cn
# @Time: 2023/12/27

import os,sys, warnings
import argparse, logging, datetime
import yaml

import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold

from dataset import TextDataset
from train import train
from predict import predict

def parse_argument() -> argparse.Namespace:
    """read in arguments from command line, kernel arguments:
    configuration file
    """
    parser = argparse.ArgumentParser(description="chinese news title classifiler")
    parser.add_argument("--config", type=str, default="./config/test.yaml")

    args = parser.parse_args()
    return args

def init_logging()->None:
    #loggin setting
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.basicConfig(
        level=logging.DEBUG,
        filename="./save/log" + current_time + ".log",
        format="%(asctime)s %(message)s",
        datefmt="%I:%M:%S ",
    )

def main() -> None:

    os.chdir(sys.path[0])
    warnings.filterwarnings("ignore")

    init_logging()
    args = parse_argument()
    try:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
    except IOError as e:
        raise IOError("config file not found")
    
    random_seed=config["seed"]
    dataset_dir = config["dir"]
    num_folds = config["nfold"]
    batch_size=config["batch_size"]
    epochs=config["epochs"]
    learning_rate=config["learning_rate"]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.info(config)
    print(config)

    # Create the dataset instance
    chn_news_dataset = TextDataset(file_path=dataset_dir)

    logging.info("dataset size:"+len(chn_news_dataset))
    print("dataset size:"+len(chn_news_dataset))

    torch.manual_seed(random_seed)

    kf = KFold(n_splits=num_folds, shuffle=True)

    for fold_idx, (train_indices, val_indices) in enumerate(kf.split(chn_news_dataset)):
        print(f"Fold {fold_idx + 1}:")
        logging.info(f"Fold {fold_idx + 1}:")

        train_loader = DataLoader(Subset(chn_news_dataset, train_indices), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(Subset(chn_news_dataset, val_indices), batch_size=batch_size, shuffle=False)
        network=train(train_loader,
                      device=device,
                      batch_size=batch_size,
                      epochs=epochs,
                      learning_rate=learning_rate,
                      architecture="resnet18")
        # Validation loop
        new_f1_score=predict(network,
                             val_loader,
                             device=device)
    #save the model with best f1 score
    torch.save(network, './models/'+'.pth')

        
'''        for batch_images, batch_labels in train_loader:
            # Your training code here
            pass
        
        # Validation loop
        for batch_images, batch_labels in val_loader:
            # Your validation code here
            pass'''


if __name__ == '__main__':
    main()

