"""Module that contains all prompts for each type of chain

Reference: https://langchain.readthedocs.io/en/latest/modules/indexes/combine_docs.html
"""

from langchain.prompts import PromptTemplate


class Stuff:
    """Stuff all the related data into the prompt as context to pass to the
    language model"""

    _prompt_template = ""
    prompt = PromptTemplate(
        template=_prompt_template, input_variables=["context", "question"]
    )


class MapReduce:
    """Create an initial prompt on each chunk of data. Then a different prompt
    is run to combine all the initial outputs."""

    _prompt_template = ""
    question_prompt = PromptTemplate(
        template=_prompt_template, input_variables=["context", "question"]
    )

    combine_prompt = PromptTemplate(
        template=_prompt_template, input_variables=["context", "question"]
    )


class MapRerank:
    """Run an initial prompt on each chunk of data, it also gives a score for
    how certain it is in its answer. The responses are ranked according to this
    score, and the highest score is returned."""

    _prompt_template = ""
    prompt = PromptTemplate(
        template=_prompt_template, input_variables=["context", "question"]
    )


class Refine:
    """Run an initial prompt on the first chunk of data, then generate some output.
    For the remaining documents, that output is passed in, and refine the output
    based on that new document.
    """

    _prompt_template = ""
    question_prompt = PromptTemplate(
        template=_prompt_template, input_variables=["context", "question"]
    )

    refine_prompt = PromptTemplate(
        template=_prompt_template, input_variables=["context", "question"]
    )
