# MADLAD-400 Classification

## Setup

Download the GGUF and start llama-server:

```bash
huggingface-cli download mradermacher/translategemma-4b-it-GGUF --include "translategemma-4b-it.Q8_0.gguf" --local-dir models/
llama-server -m models/translategemma-4b-it.Q8_0.gguf --port 8080 --parallel 4 --ctx-size 16384 --n-gpu-layers 99
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

```bash
python scripts/classify_madlad_data.py --language tl
```
