from opt.DDQN.test import test_model as test_ddqn_model
from opt.DQN.test import test_model as test_dqn_model
import json

if __name__ == "__main__":
    # Define test indices as desired
    START_IDX = 288 * 0  # for example, starting at day 30
    END_IDX = START_IDX +1*288  # one day (288 steps)

    # Load configuration file
    with open('data/parameters.json', 'r') as f:
        config = json.load(f)

    if config["ENV"]["MODE"] == "train":
        print("ERROR: Training mode")
        # stop running
        exit(0)
    else:
        print("Testing mode")

    if config["MODEL"]["MODEL_TYPE"] in ["DQN", "LSTMDQN", "SelfAttentionDQN"]:
        print(f"Training DQN model: {config["MODEL"]["MODEL_TYPE"]}")
        test_dqn_model(START_IDX, END_IDX)
    elif config["MODEL"]["MODEL_TYPE"] in ["DDQN", "CDDQN", "MHADDQN"]:
        print(f"Training DDQN model {config["MODEL"]["MODEL_TYPE"]}")
        test_ddqn_model(START_IDX, END_IDX)   


