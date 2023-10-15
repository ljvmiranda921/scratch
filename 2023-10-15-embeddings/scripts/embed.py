from pathlib import Path

import numpy as np
import spacy
import typer
from spacy.language import Language
from spacy.tokens import Doc, DocBin, Span, Token
from tqdm import tqdm
from wasabi import msg

Arg = typer.Argument
Opt = typer.Option


# c.f. https://applied-language-technology.mooc.fi/html/notebooks/part_iii/05_embeddings_continued.html
@Language.factory("tensor2attr")
class Tensor2Attr:
    """Get contextual vectors for each token, realign them,
    and store them in the .vector attribute"""

    def __init__(self, name: str, nlp: Language):
        Doc.set_extension("ctx_vector", default=[])
        Span.set_extension("ctx_vector", default=[])
        Token.set_extension("ctx_vector", default=[])

    def __call__(self, doc: Doc) -> Doc:
        self.add_attributes(doc)
        return doc

    def add_attributes(self, doc: Doc):
        """Override the .vector and .similarity attributes
        with our own implementation."""

        doc._.set("ctx_doc_vector", doc._.trf_data.tensors[-1].mean(axis=0))
        for token in doc:
            token._.set("ctx_token_vector", self.get_token_tensor(token))
        for span in doc.ents:
            span._.set("ctx_span_vector", self.get_span_tensor(span))

    def get_doc_tensor(self, doc: Doc):
        """Take a Doc object as input and returns the embedding for the entire Doc."""
        return doc._.trf_data.tensors[-1].mean(axis=0)

    def get_span_tensor(self, span: Span):
        """Take a Span as input and returns its transformer embedding."""
        tensor_ix = span.doc._.trf_data.align[span.start : span.end].data.flatten()
        out_dim = span.doc._.trf_data.tensors[0].shape[-1]
        tensor = span.doc._.trf_data.tensors[0].reshape(-1, out_dim)[tensor_ix]
        return tensor.mean(axis=0)

    def get_token_tensor(self, token: Token):
        """Take a Token as input and return its transformer embedding."""
        tensor_ix = token.doc._.trf_data.align[token.i].data.flatten()
        out_dim = token.doc._.trf_data.tensors[0].shape[-1]
        tensor = token.doc._.trf_data.tensors[0].reshape(-1, out_dim)[tensor_ix]
        return tensor.mean(axis=0)

    def get_similarity(self, doc1, doc2):
        """Get similarity score between two contextual vectors"""
        return np.dot(doc1.vector, doc2.vector) / (doc1.vector_norm * doc2.vector_norm)


def embed(
    # fmt: off
    corpus: Path = Arg(..., help="Path to the corpus directory containing spaCy files."),
    outfile: Path = Arg(..., help="Path to save the embeddings in spaCy format."),
    model: str = Opt("tl_calamancy_trf", "-m", "--model", help="Model to use for embedding."),
    verbose: bool = Opt(False, "-v", "--verbose", help="Print more information.")
    # fmt: on
):
    """Get embeddings for each span label in each document"""
    nlp = spacy.load(model, disable=["ner"])
    nlp.add_pipe("tensor2attr")
    msg.text(f"Pipeline components: {nlp.pipeline}", show=verbose)

    # Combine corpus
    files = corpus.glob("*.spacy")
    docs = []
    for file in files:
        doc_bin = DocBin().from_disk(file)
        docs.extend(list(doc_bin.get_docs(nlp.vocab)))
    msg.info(f"Found {len(docs)} documents in '{corpus}'")

    docs = docs[:10]

    # Get embeddings
    output_docs = [nlp(doc) for doc in tqdm(docs)]
    for doc in output_docs:
        for ent in doc.ents:
            print(ent, ent._.ctx_vector[:3])
    doc_bin_out = DocBin(docs=output_docs, store_user_data=True)
    doc_bin_out.to_disk(outfile)
    msg.good(f"Saved embeddings to disk: {outfile}")


if __name__ == "__main__":
    typer.run(embed)
