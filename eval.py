import torch
from processor import special_tokens
import matplotlib.pyplot as plt


def eval_model(llm_model, inference_ids, vocabulary=special_tokens, answer_position=0):
    correct = 0
    ood = 0
    false_positive = 0
    false_negative = 0
    positive = 0
    negative = 0
    correct_by_depth = {}
    incorrect_by_depth = {}
    for blob in inference_ids:
        depth = blob["depth"]
        expected = blob["output"][answer_position]
        model_input = torch.tensor([blob["input"]]).cuda()

        with torch.no_grad():
            logits = llm_model(model_input, **blob)
            last_token_logits = logits[:, -1, :]
            next_token_id = torch.argmax(last_token_logits, dim=-1)

        if next_token_id == vocabulary[1]:
            positive += 1
        if next_token_id == vocabulary[0]:
            negative += 1

        if next_token_id not in vocabulary.values():
            ood += 1
        elif expected == next_token_id:
            correct += 1
            if depth not in correct_by_depth:
                correct_by_depth[depth] = 0
            correct_by_depth[depth] += 1
        else:
            if depth not in incorrect_by_depth:
                incorrect_by_depth[depth] = 0
            incorrect_by_depth[depth] += 1
            if next_token_id == vocabulary[1]:
                false_positive += 1
            else:
                false_negative += 1

    print(f"positive predictions: {positive}, negative predictions: {negative}")
    print(f"correct: {correct}, false positive: {false_positive}, false negative: {false_negative}, ood: {ood}")
    print(f"correct by depth: {correct_by_depth}")
    print(f"incorrect by depth: {incorrect_by_depth}")


def visualize_routes(routes):
    for route_idx, route in enumerate(routes):
        plt.plot(list(range(len(route))), route)

    plt.xlabel('Time/Steps')
    plt.ylabel('Layer')
    plt.title('Layer Visitation over Time')
    plt.show()
