def doc_to_text(eg):
    doc = eg["meta"]["doc"]
    answer_to_num = {"1": 0, "2": 1}
    return answer_to_num[doc["answer"]]


def doc_to_target(eg):
    doc = eg["meta"]["doc"]
    idx = doc["sentence"].index("_") + 1
    return doc["sentence"][idx:].strip()


def doc_to_choice(eg):
    doc = eg["meta"]["doc"]
    idx = doc["sentence"].index("_")
    options = [doc["option1"], doc["option2"]]
    return [doc["sentence"][:idx] + opt for opt in options]
