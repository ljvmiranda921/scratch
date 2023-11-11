import prodigy
from prodigy.components.loaders import JSONL
from wasabi import msg


@prodigy.recipe(
    "humaneval",
    dataset=("Dataset to save answers to", "positional", None, str),
    source=("The source data as a JSONL file", "positional", None, str),
    view_id=("Annotation interface (choice/textbox)", "option", "v", str),
)
def humaneval_recipe(dataset, source, view_id="choice"):
    # Load the stream from a JSONL file and return a generator that yields a
    # dictionary for each example in the data.
    stream = JSONL(source)

    if view_id == "choice":
        config = {"choice_style": "single", "choice_auto_accept": True}
    elif view_id == "textbox":
        view_id = "blocks"
        config = {"blocks": [{"view_id": "text_input"}]}
    else:
        msg.fail("Unknown view_id, choose from choice or textbox.", exits=True)

    return {
        "view_id": view_id,
        "dataset": dataset,
        "stream": stream,
        "config": config,
    }
