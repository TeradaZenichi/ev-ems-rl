from opt.D3QN.test import test_model as test_ddqn_model
import json

if __name__ == "__main__":
    # Define test indices as desired
    START_IDX = 288 * 0  # for example, starting at day 30
    END_IDX = START_IDX +2*288  # one day (288 steps)

   
    # save model_config to file


    # Load configuration file
    with open('data/parameters.json', 'r') as f:
        config = json.load(f)

    # if config["ENV"]["MODE"] == "train":
    #     print("ERROR: Training mode")
    #     # stop running
    #     exit(0)
    # else:
    #     print("Testing mode")

    if config["MODEL"]["MODEL_TYPE"] in ["DDQN", "CDDQN", "HMHADDQN", "NHMHADDQN"]:
        print(f"test DDQN model {config["MODEL"]["MODEL_TYPE"]}")
        test_ddqn_model(START_IDX, END_IDX,"train")   


