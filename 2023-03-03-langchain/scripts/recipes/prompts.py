"""Module that contains all prompts for each type of chain

Reference: https://langchain.readthedocs.io/en/latest/modules/indexes/combine_docs.html
"""

from langchain.prompts import PromptTemplate
from langchain.prompts.base import RegexParser

output_parser_scored = RegexParser(
    regex=r"answer:(.*)\nscore:(.*)",
    output_keys=["answer", "score"],
    default_output_key="answer",
)


class Stuff:
    """Stuff all the related data into the prompt as context to pass to the
    language model"""

    PROMPT_TMPL = """
    Use the following guidelines as context to classify the text at the end.   
    This is a binary classification task, so only choose a single label. Then,
    answer in the following format:

    answer: <string>

    Here is the context:
    {context}

    Here is the text:

    Text: {question}
    """
    prompt = PromptTemplate(
        template=PROMPT_TMPL, input_variables=["context", "question"]
    )


class MapReduce:
    """Create an initial prompt on each chunk of data. Then a different prompt
    is run to combine all the initial outputs."""

    QUESTION_PROMPT_TMPL = """
    Use the following portion of a long document to see if any of the text
    is relevant to classify the text. Return any relevant text verbatim.
    {context}

    Text: {question}
    Relevant text, if any:
    """
    question_prompt = PromptTemplate(
        template=QUESTION_PROMPT_TMPL, input_variables=["context", "question"]
    )

    COMBINE_PROMPT_TMPL = """
    Given the following extracted parts of a long document and a text, classify
    the text into its label. This is a binary classification task, so only choose
    a single label. Then, answer in the following format:

    answer: <string>

    Text: {question} 
    ================
    {summaries}
    """
    combine_prompt = PromptTemplate(
        template=COMBINE_PROMPT_TMPL, input_variables=["summaries", "question"]
    )


class MapRerank:
    """Run an initial prompt on each chunk of data, it also gives a score for
    how certain it is in its answer. The responses are ranked according to this
    score, and the highest score is returned."""

    PROMPT_TMPL = """
    Use the following pieces of context to classify the text at the end. 
    In addition to giving an answer, also return a score of how fully it classifies
    the text. This should be in the following format:

    answer: <string>
    score: <score between 0 and 100>

    How to determine the score:
    - Higher is a better answer
    - Better classifies fully the given text, with sufficient level of detail
    - If you do not know the answer based on the context, that should be a score of 0.
    - Don't be overconfident!

    Context:
    ---------
    {context}
    ---------
    Text: {question}
    """
    prompt = PromptTemplate(
        template=PROMPT_TMPL,
        input_variables=["context", "question"],
        output_parser=output_parser_scored,
    )


class Refine:
    """Run an initial prompt on the first chunk of data, then generate some output.
    For the remaining documents, that output is passed in, and refine the output
    based on that new document.
    """

    QUESTION_PROMPT_TPL = """
    Context information is below.
    -----------------------------------------
    {context}
    -----------------------------------------
    Given the context information and not prior knowledge, classify
    the following text:
    {question}
    """
    question_prompt = PromptTemplate(
        template=QUESTION_PROMPT_TPL, input_variables=["context", "question"]
    )

    REFINE_PROMPT_TPL = """
    The original text to classify is as follows: {question}
    We have provided an existing answer: {existing_answer}
    We have the opportunity to refine the existing answer (only if needed)
    with some more context below.
    ----------------------------------------------
    {context}
    ----------------------------------------------
    Given the new context, refine the original answer to better
    classify the question. If the context isn't useful, return
    the original answer.
    """

    refine_prompt = PromptTemplate(
        template=REFINE_PROMPT_TPL,
        input_variables=["question", "existing_answer", "context"],
    )
