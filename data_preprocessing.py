from torch.utils.data import default_collate
import torch


def next_word_prediction_labels(input_ids, answer_ids):
    labels = [-100 for _ in input_ids] + [x for x in answer_ids]
    labels.pop(0)
    input = input_ids + answer_ids
    input.pop(-1)
    return labels, input


def pad_collate(batch, padding, length=None):
    max_size = max([len(x["input_ids"]) for x in batch]) if not length else length
    collated = []

    for elem in batch:
        copy = {}
        copy["input_ids"] = torch.tensor(elem["input_ids"] + [padding] * (max_size - len(elem["input_ids"])))

        if elem["for_classification"]:
            copy["labels"] = torch.tensor(elem["labels"]).float()
        else:
            copy["labels"] = torch.tensor(elem["labels"] + [-100] * (max_size - len(elem["labels"])))

        if elem["segments"]:
            copy["segments"] = torch.tensor(elem["segments"] + [elem["segments"][-1] + 1] * (max_size - len(elem["segments"])))

        if elem["global_attention_mask"]:
            copy["global_attention_mask"] = torch.tensor(elem["global_attention_mask"] + [False] * (max_size - len(elem["global_attention_mask"])))

        copy["depth"] = elem["depth"]
        collated.append(copy)

    return default_collate(collated)
