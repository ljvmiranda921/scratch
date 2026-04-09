# MADLAD-400 Classification

## Setup

Download the GGUF and start llama-server:

```bash
huggingface-cli download mradermacher/translategemma-4b-it-GGUF --include "translategemma-4b-it.Q4_K_M.gguf" --local-dir models/
llama-server -m models/translategemma-4b-it.Q4_K_M.gguf --port 8080 --parallel 4 --ctx-size 16384 --n-gpu-layers 99
```

## Language codes

| Language | MADLAD | translategemma |
|---|---|---|
| Lao | `lo` | `lo` |
| Faroese | `fo` | `fo` |
| Bashkir | `ba` | `ba` |
| Turkmen | — | `tk` |
| Shona | `sn` | `sn` |
| Sundanese | `su` | `su` |
| Papiamento | `pap` | — |
| Igbo | — | `ig` |
| Zulu | `zu` | `zu` |
| Xhosa | `xh` | `xh` |
| Nyanja | — | `ny` |
| Yoruba | `yo` | `yo` |
| Southern Sotho | — | `st` |
| Mizo | `lus` | — |
| Occitan | `oc` | `oc` |
| Assamese | `as` | `as` |

## Usage

### With WebOrganizer models

```bash
python scripts/classify_madlad_data.py --language tl --truncate 8192 --limit 100
```

### With LM (OpenAI / Azure OpenAI)

Create a `.env` file:

```bash
# For OpenAI
OPENAI_API_KEY=sk-...

# For Azure OpenAI
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=...
```

Run with OpenAI:

```bash
python scripts/classify_madlad_data_lm.py --language tl --truncate 8000 --limit 100
```

Run with Azure OpenAI:

```bash
python scripts/classify_madlad_data_lm.py --language lo --truncate 8192 --limit 100 --use_azure --model gpt-4.1-mini
```

Resume a previous run:

```bash
python scripts/classify_madlad_data_lm.py --language lo --resume data/classified/tl_20260409_classified.csv
```

Outputs are saved to `data/classified/` with columns: `topic`, `topic_reasoning`, `format`, `format_reasoning`, `sib200`, `sib200_reasoning`.
