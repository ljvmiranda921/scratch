def doc_to_text(eg) -> str:
    """
    Passage: <passage>
    Question: <question>
    Choices:
    A. <choice1>
    B. <choice2>
    C. <choice3>
    D. <choice4>
    Answer:
    """
    doc = eg["meta"]["doc"]
    choices = ["a", "b", "c", "d"]
    prompt = "Passage: " + doc["context"] + "\n"
    prompt += "Question: " + doc["query"] + "\nChoices:\n"
    for choice, option in zip(choices, doc["options"]):
        prompt += f"{choice.upper()}. {option}\n"
    prompt += "Answer:"
    return prompt


def doc_to_target(eg) -> int:
    doc = eg["meta"]["doc"]
    choices = ["a", "b", "c", "d"]
    return choices.index(doc["correct_options"].strip())
