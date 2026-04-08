# MADLAD-400 Classification

## Setup

Download the GGUF and start llama-server:

```bash
huggingface-cli download mradermacher/translategemma-4b-it-GGUF --include "translategemma-4b-it.Q8_0.gguf" --local-dir models/
llama-server -m models/translategemma-4b-it.Q8_0.gguf --port 8080
```

## Language codes

| Language | MADLAD | translategemma |
|---|---|---|
| Lao | `lao` | `lo` |
| Faroese | `fao` | `fo` |
| Bashkir | `bak` | `ba` |
| Turkmen | `tuk` | `tk` |
| Shona | `sna` | `sn` |
| Sundanese | `sun` | `su` |
| Papiamento | `pap` | — |
| Igbo | `ibo` | `ig` |
| Zulu | `zul` | `zu` |
| Xhosa | `xho` | `xh` |
| Nyanja | `nya` | `ny` |
| Yoruba | `yor` | `yo` |
| Southern Sotho | `sot` | `st` |
| Mizo | `lus` | — |
| Occitan | `oci` | `oc` |
| Assamese | `asm` | `as` |

## Usage

```bash
python scripts/classify_madlad_data.py --language tl
```
