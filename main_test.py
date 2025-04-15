from opt.DDQN.test import test_model

if __name__ == "__main__":
    # Define test indices as desired
    START_IDX = 288 * 0  # for example, starting at day 30
    END_IDX = START_IDX +1*288  # one day (288 steps)
    test_model(START_IDX, END_IDX)
