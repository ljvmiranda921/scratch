<!-- SPACY PROJECT: AUTO-GENERATED DOCS START (do not remove) -->

# 🪐 spaCy Project: GPT-3 annotation for argument mining

Accompanying spaCy project for my blog post, [*GPT-3 for
argument mining
annotation*](https://ljvmiranda921.github.io/notebook/2023/03/28/chain-of-thought-annotation/), where
I explored how LLM-assisted annotation can help in complex annotation tasks.

I am using an argument mining dataset from the [UKP Sentential Argument Mining
Corpus](https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/2345) ([Stab et
al., 2018](https://aclanthology.org/D18-1402/)). Here, they have sentences on
a variety of issues like cloning, minimum wage, abortion, with labels such as
`NoArgument`, `Argument_For`, and `Argument_Against`. Note that you need to
[send a request](https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/2345/restricted-resource?bitstreamId=90a1de18-7a2e-4706-89e6-cf8108cfd3e9)
to the TU Datalib in order to access the corpus. Once you have the data, copy
the `cloning.tsv` and `minimum_wage.tsv` into the `assets/` directory.

The experiments here simply tests the "reliability" of zero-shot and
chain-of-thought annotations from GPT-3. Ideally, there would be an
HCI-element in my experiments but for the sake of the blog post, I'm limiting
my work to empirical tests. However, if this work piqued your interest and
you're working in HCI, let me know and we can collaborate!


## 📋 project.yml

The [`project.yml`](project.yml) defines the data assets required by the
project, as well as the available commands and workflows. For details, see the
[spaCy projects documentation](https://spacy.io/usage/projects).

<!-- SPACY PROJECT: AUTO-GENERATED DOCS END (do not remove) -->