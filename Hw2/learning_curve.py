import json
import matplotlib.pyplot as plt
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--trainer_state_file", type=str, default=None, help="A json file containing the trainer states."
    )
    parser.add_argument(
        "--figure_title", type=str, default=None
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    with open(args.trainer_state_file, newline='', encoding='utf-8') as jsonfile:
        data = json.load(jsonfile)
    
    step_list, loss_list, EM_list = [], [], []
    for state in data["log_history"]:
        if "loss" in state:
            step_list.append(state["step"])
            loss_list.append(state["loss"])
        if "eval_exact_match" in state:
            EM_list.append(state["eval_exact_match"])
    
    figure, axis = plt.subplots(2, 1)

    if args.figure_title is not None:
        figure.suptitle(args.figure_title)
    axis[0].set_title("Loss Learning Curve")
    axis[0].set_xlabel("Step")
    axis[0].set_ylabel("Loss")
    axis[0].plot(step_list, loss_list, '-o')
    
    axis[1].set_title("EM Learning Curve")
    axis[1].set_xlabel("Step")
    axis[1].set_ylabel("Exact Match")
    axis[1].plot(step_list, EM_list, '-o')

    figure.tight_layout(pad=0.005)
    plt.show()

if __name__ == "__main__":
    main()