<!-- SPACY PROJECT: AUTO-GENERATED DOCS START (do not remove) -->

# 🪐 spaCy Project: Chain of thought annotation for argument mining

Accompanying spaCy project for my blog post, [*Chain of thought prompting for
argument mining
annotation*](https://ljvmiranda921.github.io/notebook/2023/03/28/chain-of-thought-annotation/), where
I explored how LLM-assisted annotation can help in complex annotation tasks.

I am using a re-annotated dataset of the [UKP Sentential Argument Mining
Corpus](https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/2345) ([Stab et
al., 2018](https://aclanthology.org/D18-1402/)) based from the work of
[Jakobsen, Barrett, et al., (2022)](https://aclanthology.org/2022.law-1.6/).
Here, they reframed the corpus into a text categorization task. Note that you
need to [send a
request](https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/2345/restricted-resource?bitstreamId=90a1de18-7a2e-4706-89e6-cf8108cfd3e9)
to the TU Datalib in order to access the corpus.

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