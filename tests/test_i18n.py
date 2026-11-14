"""Static checks that the five interface languages stay complete."""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
I18N = (ROOT / "frontend" / "i18n.js").read_text(encoding="utf-8")
INDEX = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
APP = (ROOT / "frontend" / "app.js").read_text(encoding="utf-8")

LANGUAGES = ["en", "sq", "it", "de", "es"]


def parse_dictionary(name: str) -> dict:
    marker = f"const {name} = {{"
    start = I18N.index(marker) + len(marker) - 1
    end = I18N.index("\n  };", start)
    body = I18N[start:end].rstrip().rstrip(",")
    return json.loads(body + "\n}")


DICTIONARIES = {name: parse_dictionary(name) for name in LANGUAGES}


def test_all_languages_share_the_same_keys():
    reference = set(DICTIONARIES["en"])
    for name, table in DICTIONARIES.items():
        missing = reference - set(table)
        extra = set(table) - reference
        assert not missing, f"{name} is missing keys: {sorted(missing)[:8]}"
        assert not extra, f"{name} has unknown keys: {sorted(extra)[:8]}"


def test_no_empty_translations():
    for name, table in DICTIONARIES.items():
        empty = [key for key, value in table.items() if not str(value).strip()]
        assert not empty, f"{name} has empty translations: {empty}"


def test_every_static_annotation_is_translated():
    keys = set(re.findall(r'data-i18n(?:-html|-alt|-aria-label|-title)?="([^"]+)"', INDEX))
    missing = keys - set(DICTIONARIES["en"])
    assert not missing, f"index.html references missing keys: {sorted(missing)}"


def test_every_literal_key_used_by_scripts_exists():
    keys = set(re.findall(r'\bt\("([^"]+)"', APP))
    keys |= set(re.findall(r'\bt\(\n?\s*"([^"]+)"', APP))
    missing = keys - set(DICTIONARIES["en"])
    assert not missing, f"app.js references missing keys: {sorted(missing)}"


def test_dynamic_key_families_exist_in_every_language():
    fields = ("short", "full", "description", "advice", "urgencyLabel")
    families = [f"class.{code}.{field}" for code in ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc") for field in fields]
    families += [f"status.{status}" for status in ("confident", "ambiguous", "uncertain", "out_of_distribution", "low_quality")]
    families += [f"rejected.{status}" for status in ("low_quality", "out_of_distribution")]
    families += [f"next.{step}" for step in ("retake_focused", "even_lighting", "consult_if_changing", "dermoscopy_only", "consult_clinician")]
    families += [f"sample.tags.{tag}" for tag in ("clear_melanoma", "clear_nevus", "ambiguous", "bcc", "bkl", "non_skin")]
    for name, table in DICTIONARIES.items():
        missing = [key for key in families if key not in table]
        assert not missing, f"{name} is missing dynamic keys: {missing}"


def test_english_and_albanian_cover_the_safety_copy():
    for key in ("result.safety", "result.setExplainer", "evidence.warningMel", "evidence.howToCoverage"):
        assert key in DICTIONARIES["en"] and key in DICTIONARIES["sq"]
