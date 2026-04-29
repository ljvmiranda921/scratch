import argparse
import re
import unicodedata
from collections import OrderedDict

import pandas as pd


CANONICAL_TO_ISO3 = {
    "Agikuyu": "kik",
    "Amharic": "amh",
    "Arabic": "ara",
    "Armenian": "hye",
    "Assamese": "asm",
    "Azerbaijani": "aze",
    "Balochi": "bal",
    "Bambara": "bam",
    "Basque": "eus",
    "Batak": "btk",
    "Belarusian": "bel",
    "Bengali": "ben",
    "Bikol": "bik",
    "Bini": "bin",
    "Bosnian": "bos",
    "Braj": "bra",
    "Brahui": "brh",
    "Bulgarian": "bul",
    "Burushaski": "bsk",
    "Buryat": "bua",
    "Cantonese": "yue",
    "Catalan": "cat",
    "Cebuano": "ceb",
    "Chinese": "zho",
    "Chuvash": "chv",
    "Croatian": "hrv",
    "Czech": "ces",
    "Danish": "dan",
    "Dutch": "nld",
    "Efik": "efi",
    "Ekpeye": "ekp",
    "English": "eng",
    "Estonian": "est",
    "Ewe": "ewe",
    "Faroese": "fao",
    "Filipino": "fil",
    "Finnish": "fin",
    "Fon": "fon",
    "French": "fra",
    "Fulfulde": "ful",
    "Galician": "glg",
    "Georgian": "kat",
    "German": "deu",
    "Greek": "ell",
    "Gujarati": "guj",
    "Hausa": "hau",
    "Hawaiian": "haw",
    "Hebrew": "heb",
    "Hindi": "hin",
    "Hokkien": "nan",
    "Hungarian": "hun",
    "Icelandic": "isl",
    "Igbo": "ibo",
    "Indonesian": "ind",
    "Isoko": "iso",
    "Italian": "ita",
    "Javanese": "jav",
    "Japanese": "jpn",
    "Kannada": "kan",
    "Kashmiri": "kas",
    "Kazakh": "kaz",
    "Khakas": "kjh",
    "Kikongo": "kon",
    "Kinyarwanda": "kin",
    "Korean": "kor",
    "Kurdish": "kur",
    "Lambani": "lmn",
    "Lingala": "lin",
    "Lithuanian": "lit",
    "Luganda": "lug",
    "Luo": "luo",
    "Macedonian": "mkd",
    "Magahi": "mag",
    "Maithili": "mai",
    "Malagasy": "mlg",
    "Malay": "msa",
    "Malayalam": "mal",
    "Mandarin": "cmn",
    "Marathi": "mar",
    "Marwari": "mwr",
    "Nepali": "nep",
    "Nigerian Pidgin": "pcm",
    "Norwegian Bokmal": "nob",
    "Norwegian Nynorsk": "nno",
    "Obolo": "ann",
    "Odia": "ori",
    "Pashto": "pus",
    "Persian": "fas",
    "Polish": "pol",
    "Portuguese": "por",
    "Punjabi": "pan",
    "Romanian": "ron",
    "Russian": "rus",
    "Sanskrit": "san",
    "Saraiki": "skr",
    "Serbian": "srp",
    "Sepedi": "nso",
    "Shekhawati": "",
    "Sinhala": "sin",
    "Sindhi": "snd",
    "Slovak": "slk",
    "Slovenian": "slv",
    "Spanish": "spa",
    "Sundanese": "sun",
    "Swahili": "swa",
    "Swedish": "swe",
    "Sylheti": "syl",
    "Tagalog": "tgl",
    "Tajik": "tgk",
    "Tamil": "tam",
    "Telugu": "tel",
    "Thai": "tha",
    "Tshivenda": "ven",
    "Tunisian Arabic": "aeb",
    "Turkish": "tur",
    "Ukrainian": "ukr",
    "Urdu": "urd",
    "Uzbek": "uzb",
    "Uyghur": "uig",
    "Vietnamese": "vie",
    "Xhosa": "xho",
    "Yoruba": "yor",
    "Zarma": "dje",
    "Zulu": "zul",
}


ALIAS_TO_CANONICAL = {
    "agikuyu": "Agikuyu",
    "agikuyu language family": "Agikuyu",
    "akikuyu": "Agikuyu",
    "amharic": "Amharic",
    "arabic": "Arabic",
    "arabic algerian dialect": "Arabic",
    "arabic najdi saudi": "Arabic",
    "arabic msa": "Arabic",
    "armenian": "Armenian",
    "assamese": "Assamese",
    "azerbaijani": "Azerbaijani",
    "balochi": "Balochi",
    "bangla": "Bengali",
    "bangla india": "Bengali",
    "bambara": "Bambara",
    "batak": "Batak",
    "bataknese": "Batak",
    "basque": "Basque",
    "belarusian": "Belarusian",
    "belorusian": "Belarusian",
    "bengali": "Bengali",
    "bengali all dialects": "Bengali",
    "bikol": "Bikol",
    "bini": "Bini",
    "bosnian": "Bosnian",
    "braj": "Braj",
    "brahui": "Brahui",
    "bulgarian": "Bulgarian",
    "burushaski": "Burushaski",
    "buryat language": "Buryat",
    "cantonese": "Cantonese",
    "catalan": "Catalan",
    "cebuano": "Cebuano",
    "chinese": "Chinese",
    "chuvash": "Chuvash",
    "croatian": "Croatian",
    "czech": "Czech",
    "danish": "Danish",
    "dholuo": "Luo",
    "dutch": "Dutch",
    "efik": "Efik",
    "ekpeye": "Ekpeye",
    "english": "English",
    "estonian": "Estonian",
    "ewe": "Ewe",
    "faroese": "Faroese",
    "farsi": "Persian",
    "farsi persian": "Persian",
    "filipino": "Filipino",
    "finnish": "Finnish",
    "fon": "Fon",
    "fongbe": "Fon",
    "french": "French",
    "french from cameroon": "French",
    "fulfulde": "Fulfulde",
    "galician": "Galician",
    "georgian": "Georgian",
    "german": "German",
    "greek": "Greek",
    "gujarati": "Gujarati",
    "hausa": "Hausa",
    "hawaiian": "Hawaiian",
    "hawaiian olelo hawaii": "Hawaiian",
    "hebrew": "Hebrew",
    "hindi": "Hindi",
    "hindia": "Hindi",
    "hokkien": "Hokkien",
    "hungarian": "Hungarian",
    "icelandic": "Icelandic",
    "igbo": "Igbo",
    "indonesia": "Indonesian",
    "indonesian": "Indonesian",
    "iranian persian": "Persian",
    "isixhosa": "Xhosa",
    "isoko": "Isoko",
    "italian": "Italian",
    "japanese": "Japanese",
    "java": "Javanese",
    "javanese": "Javanese",
    "kannada": "Kannada",
    "kashmiri": "Kashmiri",
    "kazakh": "Kazakh",
    "khakas": "Khakas",
    "kikongo": "Kikongo",
    "kikongo language": "Kikongo",
    "kikuyu": "Agikuyu",
    "kinyarwanda": "Kinyarwanda",
    "korean": "Korean",
    "kurdish": "Kurdish",
    "lambani": "Lambani",
    "lingala": "Lingala",
    "lithuania": "Lithuanian",
    "lithuanian": "Lithuanian",
    "luganda": "Luganda",
    "luo": "Luo",
    "macedonian": "Macedonian",
    "magahi": "Magahi",
    "maithili": "Maithili",
    "malagasy": "Malagasy",
    "malagasy language spoken in madagascar": "Malagasy",
    "malay": "Malay",
    "malayalam": "Malayalam",
    "mandarin": "Mandarin",
    "mandarin chinese": "Mandarin",
    "marathi": "Marathi",
    "marwadi": "Marwari",
    "marwari": "Marwari",
    "middleeast": "",
    "nepali": "Nepali",
    "nigerian pidgin": "Nigerian Pidgin",
    "norwegian bokmal": "Norwegian Bokmal",
    "norwegian nynorsk": "Norwegian Nynorsk",
    "obolo": "Obolo",
    "odia": "Odia",
    "odia indo aryan language from eastern india": "Odia",
    "odia indo aryan language not in current list": "Odia",
    "odia indo aryan language not in global piqa": "Odia",
    "odiya": "Odia",
    "oriya": "Odia",
    "pashto": "Pashto",
    "persian": "Persian",
    "pidgin": "Nigerian Pidgin",
    "polish": "Polish",
    "portuguese": "Portuguese",
    "portuguese brazilian": "Portuguese",
    "punjabi": "Punjabi",
    "pushto": "Pashto",
    "romanian": "Romanian",
    "russian": "Russian",
    "sanskrit": "Sanskrit",
    "saraiki": "Saraiki",
    "serbian": "Serbian",
    "sepedi": "Sepedi",
    "shekhawati": "Shekhawati",
    "sinhala": "Sinhala",
    "sindhi": "Sindhi",
    "slovak": "Slovak",
    "slovenian": "Slovenian",
    "spanish": "Spanish",
    "sundanese": "Sundanese",
    "swahili": "Swahili",
    "swahilli": "Swahili",
    "swedish": "Swedish",
    "sylheti": "Sylheti",
    "tagalog": "Tagalog",
    "tajik": "Tajik",
    "tamil": "Tamil",
    "telugu": "Telugu",
    "thai": "Thai",
    "thailand": "Thai",
    "tshivenda": "Tshivenda",
    "tunisian dialect": "Tunisian Arabic",
    "turkish": "Turkish",
    "uhami": "",
    "ukrainian": "Ukrainian",
    "urdu": "Urdu",
    "uzbek": "Uzbek",
    "uyghur": "Uyghur",
    "vietnamese": "Vietnamese",
    "will discuss": "",
    "xhosa": "Xhosa",
    "yoruba": "Yoruba",
    "yoruba nigerian pidigin english": "Nigerian Pidgin",
    "yoruba nigerian pidigin": "Nigerian Pidgin",
    "zarma": "Zarma",
    "zulu": "Zulu",
}


NOISE_KEYS = {
    "will discuss",
    "python",
}


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", default="contact_leads.csv")
    parser.add_argument("-o", "--output_path", default="contact_leads_dedup.csv")
    # fmt: on
    return parser.parse_args()


def _ascii_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    ascii_only = ascii_only.lower()
    ascii_only = re.sub(r"[^a-z0-9]+", " ", ascii_only).strip()
    return re.sub(r"\s+", " ", ascii_only)


def _split_candidates(value: str) -> list[str]:
    if not isinstance(value, str) or not value.strip():
        return []

    text = value.strip()
    text = re.sub(r"\b(and|or)\s*/\s*\b(and|or)\b", ",", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(and|or)\b", ",", text, flags=re.IGNORECASE)
    text = text.replace("&", ",")
    text = text.replace("/", ",")
    text = text.replace(";", ",")
    text = re.sub(r"[()]", ",", text)
    text = re.sub(r"\s*,\s*", ",", text)
    text = re.sub(r"\s+", " ", text)
    return [chunk.strip(" .,-") for chunk in text.split(",") if chunk.strip(" .,-")]


def _canonicalize(candidate: str) -> str | None:
    key = _ascii_key(candidate)
    if not key:
        return None
    if key in NOISE_KEYS:
        return None

    if key in ALIAS_TO_CANONICAL:
        canonical = ALIAS_TO_CANONICAL[key]
        return canonical or None

    for alias, canonical in sorted(ALIAS_TO_CANONICAL.items(), key=lambda item: len(item[0]), reverse=True):
        if alias and re.search(rf"\b{re.escape(alias)}\b", key):
            return canonical or None

    clean = re.sub(r"\b(i can|i would like to|preferable|native|possibly|if needed|if needed)\b", " ", key)
    clean = re.sub(r"\s+", " ", clean).strip()
    if clean in ALIAS_TO_CANONICAL:
        canonical = ALIAS_TO_CANONICAL[clean]
        return canonical or None

    if 1 <= len(clean.split()) <= 3 and re.fullmatch(r"[a-z0-9 ]+", clean):
        return " ".join(part.capitalize() for part in clean.split())
    return None


def _extract_languages(raw_value: str) -> list[str]:
    extracted = []
    for chunk in _split_candidates(raw_value):
        canonical = _canonicalize(chunk)
        if canonical:
            extracted.append(canonical)
    return list(OrderedDict.fromkeys(extracted))


def main():
    args = get_args()
    df = pd.read_csv(args.input_path)

    output_rows: list[dict] = []
    for row in df.to_dict(orient="records"):
        languages = _extract_languages(row.get("Language", ""))
        if not languages:
            output_row = dict(row)
            output_row["Language"] = ""
            output_row["Language_ISO3"] = ""
            output_rows.append(output_row)
            continue

        for language in languages:
            output_row = dict(row)
            output_row["Language"] = language
            output_row["Language_ISO3"] = CANONICAL_TO_ISO3.get(language, "")
            output_rows.append(output_row)

    out_df = pd.DataFrame(output_rows)
    out_df.to_csv(args.output_path, index=False)


if __name__ == "__main__":
    main()
