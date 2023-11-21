import prodigy
from prodigy.components.loaders import JSONL
from wasabi import msg


@prodigy.recipe(
    "humaneval",
    dataset=("Dataset to save answers to", "positional", None, str),
    source=("The source data as a JSONL file", "positional", None, str),
    view_id=("Annotation interface (choice/textbox)", "option", "v", str),
    show_meta=("Hide meta containing the label", "option", None, bool),
)
def humaneval_recipe(dataset, source, view_id="choice", show_meta=False):
    # Load the stream from a JSONL file and return a generator that yields a
    # dictionary for each example in the data.
    stream = JSONL(source)

    if view_id == "choice":
        config = {"choice_style": "single", "choice_auto_accept": True}
    elif view_id == "textbox":
        view_id = "blocks"
        config = {
            "blocks": [
                {"view_id": "text"},
                {
                    "view_id": "text_input",
                    "field_id": "user_input",
                    "field_label": "",
                    "field_rows": 5,
                    "field_placeholder": "Type here...",
                    "field_autofocus": False,
                },
            ]
        }

    else:
        msg.fail("Unknown view_id, choose from choice or textbox.", exits=True)

    # Hide meta
    config["hide_meta"] = not show_meta

    return {
        "view_id": view_id,
        "dataset": dataset,
        "stream": stream,
        "config": config,
    }
