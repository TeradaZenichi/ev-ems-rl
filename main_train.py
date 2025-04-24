from opt.D3QN.train import train_and_evaluate
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

    if config["MODEL"]["MODEL_TYPE"] in ["DDQN", "CDDQN", "MHADDQN", "HMHADDQN", "NHMHADDQN"]:
        print(f"Training D3QN model {config["MODEL"]["MODEL_TYPE"]}")
        train_and_evaluate()
