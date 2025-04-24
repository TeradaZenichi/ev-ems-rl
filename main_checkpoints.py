from opt.D3QN.test_checkpoints import *

if __name__ == "__main__":
    # Load configuration file
    with open('data/parameters.json', 'r') as f:
        config = json.load(f)

    if config["ENV"]["MODE"] == "train":
        print("ERROR: Training mode")
        # stop running
        exit(0)
    else:
        print("Testing mode")
        evaluate_checkpoints()