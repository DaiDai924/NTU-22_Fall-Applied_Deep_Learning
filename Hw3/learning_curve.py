import json
import matplotlib.pyplot as plt
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--trainer_state_file", type=str, default=None, help="A json file containing the trainer states."
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    with open(args.trainer_state_file, newline='', encoding='utf-8') as jsonfile:
        data = json.load(jsonfile)
    
    step_list, rouge_1_list, rouge_2_list, rouge_l_list = [], [], [], []
    for state in data["log_history"]:
        if "eval_rouge-1" in state:
            step_list.append(state["step"])
            rouge_1_list.append(state["eval_rouge-1"])
            rouge_2_list.append(state["eval_rouge-2"])
            rouge_l_list.append(state["eval_rouge-l"])
    
    plt.figure()
    plt.title("ROUGE Learning Curve")
    plt.xlabel("Step")
    plt.ylabel("ROUGE")
    plt.plot(step_list, rouge_1_list, 'r-o', label="rouge-1")
    plt.plot(step_list, rouge_2_list, 'g-o', label="rouge-2")
    plt.plot(step_list, rouge_l_list, 'b-o', label="rouge-l")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()