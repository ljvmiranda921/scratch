"""Module that contains all chains needed for a Prodigy Chain

Implementation was based from the question & answering use-case:
https://github.com/hwchase17/langchain/blob/master/langchain/chains/question_answering/__init__.py
"""

from typing import Any, Mapping, Literal, Optional, Protocol

from langchain.callbacks.base import BaseCallbackManager
from langchain.chains.combine_documents import map_reduce, map_rerank, refine
from langchain.chains.combine_documents import stuff
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.llms.base import BaseLLM
from langchain.prompts.base import BasePromptTemplate

from .prompts import MapReduce, MapRerank, Refine, Stuff


def load_prodigy_chain(
    llm: BaseLLM,
    chain_type: Literal["stuff", "map_reduce", "refine", "map_rerank"] = "stuff",
    *,
    verbose: Optional[bool] = None,
    callback_manager: Optional[BaseCallbackManager] = None,
    **kwargs: Any,
) -> BaseCombineDocumentsChain:
    """Load Prodigy chain

    This is the main function for loading a Prodigy chain. You can set the
    `chain_type` parameter to define how each chunk of a document is treated in
    the prompt. For more information, check:
    https://langchain.readthedocs.io/en/latest/modules/indexes/combine_docs.html
    """
    loader_mapping: Mapping[str, LoadingCallable] = {
        "stuff": _load_stuff_chain,
        "map_reduce": _load_map_reduce_chain,
        "refine": _load_refine_chain,
        "map_rerank": _load_map_rerank_chain,
    }
    if chain_type not in loader_mapping:
        raise ValueError(
            f"Got unsupported chain type: {chain_type}. "
            f"Should be one of {loader_mapping.keys()}"
        )

    return loader_mapping[chain_type](
        llm, verbose=verbose, callback_manager=callback_manager, **kwargs
    )


class LoadingCallable(Protocol):
    """Interface for loading the combine documents chain."""

    def __call__(self, llm: BaseLLM, **kwargs: Any) -> BaseCombineDocumentsChain:
        """Callable to load the combine documents chain."""


def _load_stuff_chain(
    llm: BaseLLM,
    prompt: BasePromptTemplate = Stuff.prompt,
    document_variable_name: str = "context",
    verbose: Optional[bool] = None,
    callback_manager: Optional[BaseCallbackManager] = None,
    **kwargs: Any,
) -> stuff.StuffDocumentsChain:
    llm_chain = LLMChain(
        llm=llm, prompt=prompt, verbose=verbose, callback_manager=callback_manager
    )
    return stuff.StuffDocumentsChain(
        llm_chain=llm_chain,
        input_key=document_variable_name,
        document_variable_name=document_variable_name,
        verbose=verbose,
        callback_manager=callback_manager,
        **kwargs,
    )


def _load_map_reduce_chain(
    llm: BaseLLM,
    question_prompt: BasePromptTemplate = MapReduce.question_prompt,
    combine_prompt: BasePromptTemplate = MapReduce.combine_prompt,
    combine_document_variable_name: str = "summaries",
    map_reduce_document_variable_name: str = "context",
    collapse_prompt: Optional[BasePromptTemplate] = None,
    reduce_llm: Optional[BaseLLM] = None,
    collapse_llm: Optional[BaseLLM] = None,
    verbose: Optional[bool] = None,
    callback_manager: Optional[BaseCallbackManager] = None,
    **kwargs: Any,
) -> map_reduce.MapReduceDocumentsChain:
    map_chain = LLMChain(
        llm=llm,
        prompt=question_prompt,
        verbose=verbose,
        callback_manager=callback_manager,
    )
    _reduce_llm = reduce_llm or llm
    reduce_chain = LLMChain(
        llm=_reduce_llm,
        prompt=combine_prompt,
        verbose=verbose,
        callback_manager=callback_manager,
    )
    combine_document_chain = stuff.StuffDocumentsChain(
        llm_chain=reduce_chain,
        document_variable_name=combine_document_variable_name,
        verbose=verbose,
        callback_manager=callback_manager,
    )
    if collapse_prompt is None:
        collapse_chain = None
        if collapse_llm is not None:
            raise ValueError(
                "collapse_llm provided, but collapse_prompt was not: please "
                "provide one or stop providing collapse_llm."
            )
    else:
        _collapse_llm = collapse_llm or llm
        collapse_chain = stuff.StuffDocumentsChain(
            llm_chain=LLMChain(
                llm=_collapse_llm,
                prompt=collapse_prompt,
                verbose=verbose,
                callback_manager=callback_manager,
            ),
            document_variable_name=combine_document_variable_name,
            verbose=verbose,
            callback_manager=callback_manager,
        )
    return map_reduce.MapReduceDocumentsChain(
        llm_chain=map_chain,
        input_key=map_reduce_document_variable_name,
        combine_document_chain=combine_document_chain,
        document_variable_name=map_reduce_document_variable_name,
        collapse_document_chain=collapse_chain,
        verbose=verbose,
        callback_manager=callback_manager,
        **kwargs,
    )


def _load_refine_chain(
    llm: BaseLLM,
    question_prompt: BasePromptTemplate = Refine.question_prompt,
    refine_prompt: BasePromptTemplate = Refine.refine_prompt,
    document_variable_name: str = "context",
    initial_response_name: str = "existing_answer",
    refine_llm: Optional[BaseLLM] = None,
    verbose: Optional[bool] = None,
    callback_manager: Optional[BaseCallbackManager] = None,
    **kwargs: Any,
) -> refine.RefineDocumentsChain:
    initial_chain = LLMChain(
        llm=llm,
        prompt=question_prompt,
        verbose=verbose,
        callback_manager=callback_manager,
    )
    _refine_llm = refine_llm or llm
    refine_chain = LLMChain(
        llm=_refine_llm,
        prompt=refine_prompt,
        verbose=verbose,
        callback_manager=callback_manager,
    )
    return refine.RefineDocumentsChain(
        initial_llm_chain=initial_chain,
        input_key=document_variable_name,
        refine_llm_chain=refine_chain,
        document_variable_name=document_variable_name,
        initial_response_name=initial_response_name,
        verbose=verbose,
        callback_manager=callback_manager,
        **kwargs,
    )


def _load_map_rerank_chain(
    llm: BaseLLM,
    prompt: BasePromptTemplate = MapRerank.prompt,
    verbose: bool = False,
    document_variable_name: str = "context",
    rank_key: str = "score",
    answer_key: str = "answer",
    callback_manager: Optional[BaseCallbackManager] = None,
    **kwargs: Any,
) -> map_rerank.MapRerankDocumentsChain:
    llm_chain = LLMChain(
        llm=llm, prompt=prompt, verbose=verbose, callback_manager=callback_manager
    )
    return map_rerank.MapRerankDocumentsChain(
        llm_chain=llm_chain,
        rank_key=rank_key,
        input_key=document_variable_name,
        answer_key=answer_key,
        document_variable_name=document_variable_name,
        verbose=verbose,
        callback_manager=callback_manager,
        **kwargs,
    )
