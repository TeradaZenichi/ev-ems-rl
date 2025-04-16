from opt.DQN.train import train as train_dqn
from opt.DDQN.train import train as train_ddqn
import os
import json

if __name__ == "__main__":
    # Load configuration file
    with open('data/parameters.json', 'r') as f:
        config = json.load(f)

    if config["ENV"]["MODE"] == "train":
        print("Training mode")
    else:
        print("ERROR: Testing mode")
        # stop running
        exit(0)

    if config["MODEL"]["MODEL_TYPE"] in ["DQN", "LSTMDQN", "SelfAttentionDQN"]:
        print(f"Training DQN model: {config["MODEL"]["MODEL_TYPE"]}")
        train_dqn()
    elif config["MODEL"]["MODEL_TYPE"] in ["DDQN", "CDDQN", "MHADDQN"]:
        print(f"Training DDQN model {config["MODEL"]["MODEL_TYPE"]}")
        train_ddqn()
