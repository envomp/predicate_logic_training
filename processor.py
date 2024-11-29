import json
import random

from llama_models.llama3.api import Tokenizer

from data_preprocessing import next_word_prediction_labels

adjectives = ["wrong", "glamorous", "stormy", "weary", "witty", "tense", "pessimistic", "frightened", "cruel",
              "helpful", "hurt", "comfortable", "worried", "aggressive", "blushing", "proud", "rude", "lucky",
              "fearless", "diplomatic", "hypocritical", "quaint", "perfect", "jittery", "friendly", "horrible",
              "attentive", "victorious", "naughty", "condemned", "fancy", "zealous", "crowded", "sincere", "busy",
              "curious", "talented", "modern", "bright", "mean", "foolish", "tender", "mysterious", "alert", "fine",
              "amused", "bad-tempered", "lonely", "good", "muddy", "frantic", "shy", "versatile", "loving", "elated",
              "powerful", "troubled", "easy", "innocent", "hilarious", "strange", "disobedient", "elegant", "joyous",
              "careless", "shiny", "adorable", "inquisitive", "precious", "wide-eyed", "beautiful", "confused",
              "embarrassed", "famous", "nervous", "plain", "distinct", "courageous", "gleaming", "bored",
              "broad-minded", "fragile", "outstanding", "cooperative", "inexpensive", "charming", "confident", "grumpy",
              "messy", "excited", "long", "talkative", "reserved", "tame", "wandering", "different", "agreeable",
              "cute", "scared", "popular", "exuberant", "impartial", "old-fashioned", "clean", "helpless",
              "thoughtless", "selfish", "serious", "homely", "worrisome", "polite", "tired", "ugly", "light", "ugliest",
              "tidy", "sleepy", "unpleasant", "enchanting", "intellectual", "combative", "rational", "uptight",
              "sensible", "supportive", "spotless", "disgusted", "ambitious", "pleasant", "silly", "clumsy", "average",
              "vivacious", "gifted", "straightforward", "smart", "impatient", "outrageous", "calm", "gorgeous", "frail",
              "dull", "thoughtful", "dishonest", "difficult", "bossy", "stubborn", "anxious", "stupid", "attractive"]
rule_block = 200
deduction_separator = 201
rule_separator = 202
fact_block = 203
query_block = 204
preds_block = 205
end_of_turn = 206
end_of_text = 207
special_tokens = {1: 210, 0: 211}
pad = 208


def process_atoms(facts):
    # return [x for x in facts]
    return [adjectives[int(x)] for x in facts]


def process_rules(rules):
    answer = []
    for facts, deduction in rules:
        facts = process_atoms(facts)
        answer.append(f"{' and '.join(facts)} is {process_atoms([deduction])[0]}")
    return ". ".join(answer)


def llama_tokenize_input(blob, tokenizer: Tokenizer):
    from llama_models.llama3.api import ChatFormat

    message = f"""
facts: {", ".join(process_atoms(blob["facts"]))}
rules: {process_rules(blob["rules"])}
query: {process_atoms(blob["query"][0])[0]}
result: """
    format = ChatFormat(tokenizer)
    input_ids = format.encode_content(message).tokens
    return input_ids


def llama_tokenize_output(blob, tokenizer: Tokenizer, for_classification: bool):
    from llama_models.llama3.api import ChatFormat

    if for_classification:
        return [0, 1] if blob["label"][0] else [1, 0]
    else:
        answer = "True" if blob["label"][0] else "False"
        format = ChatFormat(tokenizer)
        answer_ids = format._encode_content(answer)[0] + [tokenizer.special_tokens["<|eom_id|>"]]
        return answer_ids


def tokenize_input(blob, global_attention_mask=False):
    query = [query_block] + [int(x) for x in blob["query"]]
    facts = [fact_block] + [int(x) for i, x in enumerate(blob["facts"])]
    preds = [preds_block] + [int(x) for i, x in enumerate(blob["preds"])]

    rules = [[int(y) for y in x] + [deduction_separator, int(r)] for i, (x, r) in enumerate(blob["rules"])]
    rules = [rule_block] + [x for i, x in enumerate([y for x in rules for y in x])]
    input_ids = preds + rules + facts + query + [end_of_turn]

    if not global_attention_mask:
        return input_ids
    else:
        return input_ids, [False for _ in input_ids]


def tokenize_output(blob, for_classification: bool):
    if for_classification:
        return [0, 1] if blob["label"][0] else [1, 0]
    else:
        label = list(map(lambda x: special_tokens[int(x)], blob["label"]))
        answer_ids = label + [end_of_text]
        return answer_ids


segment_by = [deduction_separator, rule_separator, rule_block, fact_block, query_block, end_of_turn, end_of_text, pad]


def generate_segments(input_ids):
    segments = []
    for elem in input_ids:
        segments.append(1 if elem in segment_by else 0)
    return segments


def is_reachable(facts, rules, query):
    reachable = set(facts)
    changed = True
    while changed:
        changed = False
        for rule in rules:
            premises, conclusion = rule
            if all(premise in reachable for premise in premises) and conclusion not in reachable:
                reachable.add(conclusion)
                changed = True
    return 1 if query in reachable else 0


def process_labels(blob, expand):
    blob["query"] = blob["preds"] if expand else [blob["query"]]
    blob["label"] = [is_reachable(blob["facts"], blob["rules"], query) for query in blob["preds"]] if expand else [blob["label"]]
    return blob


def load(file, expand=False, tokenizer=None, for_classification=False):
    with open(file, "r") as f:
        data = json.load(f)

    processed = [process_labels(x, expand) for x in data]

    ds = []
    for elem in processed:
        global_attention_mask = None
        if tokenizer is not None:
            inp, out = llama_tokenize_input(elem, tokenizer), llama_tokenize_output(elem, tokenizer, for_classification)
        else:
            (inp, global_attention_mask), out = tokenize_input(elem, global_attention_mask=True), tokenize_output(elem, for_classification)

        # input/output for inference, input_ids/labels for training
        if for_classification:
            data = {"input": inp, "output": out, "input_ids": inp, "labels": out}
        else:
            labels, input_ids = next_word_prediction_labels(inp, out)
            data = {"input": inp, "output": out, "input_ids": input_ids, "labels": labels}

        if global_attention_mask:
            data["global_attention_mask"] = global_attention_mask
        else:
            data["global_attention_mask"] = None

        if not tokenizer:
            data["segments"] = generate_segments(data["input_ids"])
        else:
            data["segments"] = None

        data["depth"] = elem["depth"]
        data["for_classification"] = for_classification

        if data["depth"] <= 1:
            ds.append(data)
    return ds


def train_curriculum(ds: list, epoch, select_layer_items=500, non_select_layer_items=50):
    samples_by_depth = {i: [] for i in range(7)}
    for elem in ds:
        samples_by_depth[elem["depth"]].append(elem)
    ds.clear()

    if epoch >= 7:
        dump = []
        for i in range(7):
            dump.extend(samples_by_depth[i])
        random.shuffle(dump)
        return dump

    train_ds = []
    for _ in range(select_layer_items):
        train_ds.append(samples_by_depth[epoch].pop(0))
    for j in range(epoch):
        for _ in range(non_select_layer_items):
            train_ds.append(samples_by_depth[j].pop(0))

    for i in range(7):
        ds.extend(samples_by_depth[i])
    return train_ds


def select(ds: list, select_items=500):
    samples_by_depth = {i: [] for i in range(7)}
    for elem in ds:
        samples_by_depth[elem["depth"]].append(elem)
    ds.clear()

    train_ds = []
    for key in samples_by_depth.keys():
        for _ in range(select_items):
            train_ds.append(samples_by_depth[key].pop(0))
    for i in range(7):
        ds.extend(samples_by_depth[i])
    return train_ds


if __name__ == '__main__':
    result = load("validation/prop_examples_lp.txt", expand=False, for_classification=False)
    print(result[2])
