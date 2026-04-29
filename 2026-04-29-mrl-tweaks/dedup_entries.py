import argparse
import re
import unicodedata
from collections import OrderedDict

import langcodes
import pandas as pd


ALIAS_TO_QUERY = {
    "agikuyu": "Kikuyu",
    "algerian dialect": "Arabic",
    "belorusian": "Belarusian",
    "brazilian": "Portuguese",
    "do persian": "Persian",
    "fongbe": "Fon",
    "hindia": "Hindi",
    "hokkien": "nan",
    "ijesa dialect": "Yoruba",
    "indonesia": "Indonesian",
    "kashmiri": "Kashmiri",
    "lithuania": "Lithuanian",
    "marwadi": "Marwari",
    "najdi-saudi": "Arabic",
    "najdi saudi": "Arabic",
    "nigerian pidigin english": "Nigerian Pidgin",
    "odiya": "Odia",
    "olelo hawaii": "Hawaiian",
    "oriya": "Odia",
    "pushto": "Pashto",
    "saraiki": "Saraiki",
    "also work on swedish": "Swedish",
    "work on german in a group": "German",
    "swahilli": "Swahili",
    "thailand": "Thai",
    "tshivenda": "Venda",
}

CODE_TO_QUERY = {
    "bal": "Balochi",
    "brh": "Brahui",
    "chv": "Chuvash",
    "kas": "Kashmiri",
    "pus": "Pashto",
    "skr": "Saraiki",
}

NOISE_PATTERNS = [
    r"\bas i am not native in it\b",
    r"\bi can\b",
    r"\bi can also\b",
    r"\bi would like to\b",
    r"\bpreferable\b",
    r"\bnative\b",
    r"\bnative speaker\b",
    r"\bpossibly\b",
    r"\bif needed\b",
    r"\bi am working with several collaborators on this\b",
    r"\bi have already registered for farsi\b",
    r"\bbut i know the language\b",
    r"\bwhich is another name for persian\b",
    r"\blow-resourced middle eastern languages\b",
    r"\blow-resource pakistani languages\b",
    r"\blow-resource pakistani\b",
    r"\blow-resource\b",
    r"\blow-resourced\b",
    r"\bincluding varieties of\b",
    r"\bthere romanized versions\b",
    r"\bthese are not currently represented there\b",
    r"\bwould extend coverage in south asia\b",
    r"\blanguage spoken in madagascar\b",
    r"\bspoken in the faroe islands\b",
    r"\bafrican country\b",
    r"\bdravidian language family\b",
    r"\bindo-aryan language related to bengali\b",
    r"\biso 639-3\b",
    r"\biso 15924\b",
    r"\bglottocode\b",
]

NOISE_TOKENS = {
    "",
    "all dialects",
    "as i am not in it",
    "contribute culturally specific global piqa datasets for low-resource pakistani languages such as saraiki",
    "cyrl",
    "dardic language of northern pakistan",
    "khak1248",
    "will discuss",
    "python",
    "speaker",
    "middleeast",
    "msa",
}


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", default="contact_leads.csv")
    parser.add_argument("-o", "--output_path", default="contact_leads_dedup.csv")
    # fmt: on
    return parser.parse_args()


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text))
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"[^a-z0-9_;/,&() -]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _split_candidates(value: str) -> list[str]:
    text = _normalize(value)
    if not text:
        return []

    for pattern in NOISE_PATTERNS:
        text = re.sub(pattern, " ", text)

    text = re.sub(r"\b(and/or|and|or)\b", ",", text)
    text = text.replace("&", ",")
    text = text.replace("/", ",")
    text = text.replace(";", ",")
    text = re.sub(r"[()]", ",", text)
    text = re.sub(r"\s*,\s*", ",", text)
    text = re.sub(r"\s+", " ", text)

    return [chunk.strip(" .,-") for chunk in text.split(",") if chunk.strip(" .,-")]


def _resolve_with_langcodes(query: str) -> tuple[str, str] | None:
    try:
        language = langcodes.find(query)
    except Exception:
        return None

    try:
        iso3 = language.to_alpha3()
    except Exception:
        iso3 = ""

    display_name = language.display_name()
    display_name = re.sub(r"\s*\([^)]*\)", "", display_name).strip()
    return display_name, iso3


def _resolve_candidate(candidate: str) -> tuple[str, str] | None:
    if candidate in NOISE_TOKENS:
        return None

    code_match = re.fullmatch(r"([a-z]{3})(?:_[a-z0-9]+)?", candidate)
    if code_match:
        code = code_match.group(1)
        if code in CODE_TO_QUERY:
            return _resolve_with_langcodes(CODE_TO_QUERY[code])

    if candidate in ALIAS_TO_QUERY:
        return _resolve_with_langcodes(ALIAS_TO_QUERY[candidate])

    resolved = _resolve_with_langcodes(candidate)
    if resolved:
        return resolved

    for alias, query in ALIAS_TO_QUERY.items():
        if re.search(rf"\b{re.escape(alias)}\b", candidate):
            return _resolve_with_langcodes(query)

    fallback = " ".join(part.capitalize() for part in candidate.split())
    return fallback, ""


def _extract_languages(value: str) -> list[tuple[str, str]]:
    deduped = OrderedDict()
    for candidate in _split_candidates(value):
        resolved = _resolve_candidate(candidate)
        if not resolved:
            continue
        name, iso3 = resolved
        if name:
            deduped[name] = iso3
    return list(deduped.items())


def main():
    args = get_args()
    df = pd.read_csv(args.input_path)

    output_rows: list[dict] = []
    for row in df.to_dict(orient="records"):
        raw_language = row.get("Language", "")
        raw_language = "" if pd.isna(raw_language) else str(raw_language)
        languages = _extract_languages(raw_language)
        if not languages:
            output_row = dict(row)
            output_row["Language"] = ""
            output_row["Language_Original"] = raw_language
            output_row["Language_ISO3"] = ""
            output_rows.append(output_row)
            continue

        for language, iso3 in languages:
            output_row = dict(row)
            output_row["Language"] = language
            output_row["Language_Original"] = raw_language
            output_row["Language_ISO3"] = iso3
            output_rows.append(output_row)

    pd.DataFrame(output_rows).to_csv(args.output_path, index=False)


if __name__ == "__main__":
    main()
