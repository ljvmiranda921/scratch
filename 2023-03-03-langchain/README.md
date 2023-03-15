<!-- SPACY PROJECT: AUTO-GENERATED DOCS START (do not remove) -->

# 🪐 spaCy Project: Including annotation guidelines for argument mining annotation

Accompanying spaCy project for my blog post, [*GPT-3 for argument mining
annotation*](https://ljvmiranda921.github.io/notebook/2023/05/03/annotation-guidlines-llm/),
where I explored how we can include annotation guidelines for LLM-assisted
annotation for complex tasks like argument mining. 

I am using an argument mining dataset from the [UKP Sentential Argument Mining
Corpus](https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/2345) ([Stab et
al., 2018](https://aclanthology.org/D18-1402/)). Here, they have sentences on
a variety of issues like cloning, minimum wage, abortion, with labels such as
`NoArgument`, `Argument_For`, and `Argument_Against`. Note that you need to
[send a request](https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/2345/restricted-resource?bitstreamId=90a1de18-7a2e-4706-89e6-cf8108cfd3e9)
to the TU Datalib in order to access the corpus. Once you have the data, copy
the `cloning.tsv` and `minimum_wage.tsv` into the `assets/` directory.

### Quick setup

Make sure to [install Prodigy](https://prodi.gy/docs/install) as well as a few additional Python dependencies:

```bash
python -m pip install prodigy -f https://XXXX-XXXX-XXXX-XXXX@download.prodi.gy
python -m pip install -r requirements.txt
```

With `XXXX-XXXX-XXXX-XXXX` being your personal Prodigy license key.

Then, create a new API key from [openai.com](https://beta.openai.com/account/api-keys) or fetch an existing
one. Record the secret key as well as the [organization key](https://beta.openai.com/account/org-settings)
and make sure these are available as environmental variables. For instance, set them in a `.env` file in the
root directory:

```
OPENAI_ORG = "org-..."
OPENAI_KEY = "sk-..."
```


## 📋 project.yml

The [`project.yml`](project.yml) defines the data assets required by the
project, as well as the available commands and workflows. For details, see the
[spaCy projects documentation](https://spacy.io/usage/projects).

### ⏯ Commands

The following commands are defined by the project. They
can be executed using [`spacy project run [name]`](https://spacy.io/api/cli#project-run).
Commands are only re-run if their inputs have changed.

| Command | Description |
| --- | --- |
| `convert` | Convert UKP's TSV file into both JSONL and spaCy formats |
| `fetch` | Run an LLM-assisted textcat.fetch recipe to label text in bulk |
| `evaluate` | Evaluate predictions on a gold-annotated dataset |

### ⏭ Workflows

The following workflows are defined by the project. They
can be executed using [`spacy project run [name]`](https://spacy.io/api/cli#project-run)
and will run the specified commands in order. Commands are only re-run if their
inputs have changed.

| Workflow | Steps |
| --- | --- |
| `all` | `convert` &rarr; `fetch` &rarr; `evaluate` |

### 🗂 Assets

The following assets are defined by the project. They can
be fetched by running [`spacy project assets`](https://spacy.io/api/cli#project-assets)
in the project directory.

| File | Source | Description |
| --- | --- | --- |
| `assets/guidelines/morante2020.pdf` | URL | Annotation guidelines from [Morante et al. (2020)](https://aclanthology.org/2020.lrec-1.611) for context-independent claim-like sentence detection. The original guidelines are from https://git.io/J1OKR and the IAA is measured by its token-level annotation F-score, which in this case is 42.4. |
| `assets/guidelines/levy2018.pdf` | URL | Annotation guidelines from [Levy at al. (2018)](https://aclanthology.org/C18-1176/) for context-dependent claim-detection.  Here, the term claim refers to the "assertion the argument aims to prove" or simply, the conclusion. The IAA, using Cohen's kappa metric, is 0.58. |
| `assets/guidelines/stab2018.pdf` | URL | Annotation guidelines from [Stab et al. (2018)](https://aclanthology.org/D18-1402/) for context-dependent claim and premise detection. The UKP dataset came from this work.  Sometimes, it contains statements of general topics that do not reflect a conclusion in itself. In the original paper, they also require the annotators to distinguish between supporting and Argument_againsts. The IAA, using Cohen's kappa metric, is 0.721 for two expert annotators and 0.40 for non-experts. |
| `assets/guidelines/shnarch2018.pdf` | URL | Annotation guidelines from [Shnarch et al. (2018)](https://aclanthology.org/2020.findings-emnlp.243/) for context-dependent claim and premise detection. They use the term claim as meaning the conclusion and premise as a type of evidence. The IAA, using Fleiss' kappa metric, is 0.45. |

<!-- SPACY PROJECT: AUTO-GENERATED DOCS END (do not remove) -->