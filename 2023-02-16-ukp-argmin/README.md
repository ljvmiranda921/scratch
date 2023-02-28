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

### Getting the OpenAI annotations

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

From there, you can run the following command to get batch annotations from OpenAI:

```
python -m spacy project run openai
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
| `preprocess` | Convert the data to spaCy's binary format |
| `train` | Train a text classification model |
| `evaluate` | Evaluate the model and export metrics |
| `openai-preprocess` | Convert the corpus into JSONL files to load into Prodigy |
| `openai-fetch-textcat` | Run batch annotations using the `textcat.openai.fetch` recipe |
| `openai-evaluate` | Evaluate OpenAI annotations to the test data |
| `openai-fetch-spans` | Run batch annotations using the `spans.openai.fetch` recipe |
| `openai-fetch-cot` | Run batch annotations using the `textcat.openai.fetch` recipe using chain-of-thought |

### ⏭ Workflows

The following workflows are defined by the project. They
can be executed using [`spacy project run [name]`](https://spacy.io/api/cli#project-run)
and will run the specified commands in order. Commands are only re-run if their
inputs have changed.

| Workflow | Steps |
| --- | --- |
| `textcat` | `preprocess` &rarr; `train` &rarr; `evaluate` |
| `openai` | `openai-preprocess` &rarr; `openai-fetch-textcat` &rarr; `openai-evaluate` |

<!-- SPACY PROJECT: AUTO-GENERATED DOCS END (do not remove) -->