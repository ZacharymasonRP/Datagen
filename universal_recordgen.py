#!/usr/bin/env python3
"""
universal_recordgen.py — Generate many synthetic records into ONE JSON file from a flexible template.

Key guarantees (aligned to your rpsubmit mapping):
- Always emits the fields required by your mapping with THESE EXACT NAMES:
  ExternalID, Author, SourceCreatedDate, SourceCreatedBy,
  SourceLastModifiedDate, SourceLastModifiedBy, SourceURL.
- Also emits Category and ProjectCode if present or constructible from the template.
- All other template-defined fields are included as source properties.

Template highlights
- Works with simple "options" lists for choices (names, positions, notes, etc.).
- Randomizes dates and IDs. ID pattern supports {YYYY}{MM}{DD} and {####} width.
- Supports title/source URL templating using any fields you emit.
- Compatible with earlier templates (e.g., your board minutes) via light aliases.

Usage
  python universal_recordgen.py template.json --output out.json --count 500 --id-mode seq --seed 42

Python: 3.9+; stdlib only.
"""

import argparse
import json
import random
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple
import importlib.util

# -----------------------------
# CLI
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Universal JSON record generator (mapping-safe).")
    p.add_argument("template", help="Path to template JSON file")
    p.add_argument("--output", default="out.json", help="Output JSON file path")
    p.add_argument("--count", type=int, default=None, help="Override template.count")
    p.add_argument("--seed", type=int, default=None, help="Random seed; omit for nondeterministic runs")
    p.add_argument("--id-mode", choices=["seq", "random"], default="random",
                   help="Numeric suffix strategy for {####}: sequential or randomized")
    p.add_argument("--text-mode", choices=["grammar", "llm-plugin", "none"], default="grammar",
               help="How to create Title and Notes. 'grammar' is fast & offline; 'llm-plugin' calls a plugin; 'none' disables.")
    p.add_argument("--llm-plugin", default=None,
               help="Path to a Python file exporting generate_text(record, template) -> {'title': str, 'notes': str}")
    return p.parse_args()

# -----------------------------
# Utilities
# -----------------------------
ISOZ = "%Y-%m-%dT%H:%M:%S.%fZ"

def iso_z(dt: datetime) -> str:
    """Return ISO 8601 with Z."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime(ISOZ)

def parse_dt(s: str) -> datetime:
    """Parse YYYY-MM-DD or full ISO; force UTC."""
    if "T" in s:
        # rely on fromisoformat; coerce to UTC if naive
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)

def day_sequence(start: datetime, end: datetime, n: int) -> List[datetime]:
    """Roughly even spread of n dates between start and end."""
    total_days = (end - start).days + 1
    step = max(1, total_days // max(1, n))
    dates, cur = [], start.replace(hour=14, minute=3, second=0, microsecond=0, tzinfo=timezone.utc)
    while cur <= end and len(dates) < n:
        dates.append(cur)
        cur += timedelta(days=step)
    # pad if short
    while len(dates) < n:
        cur += timedelta(days=1)
        dates.append(cur)
    return dates[:n]

def month_sequence(start: datetime, end: datetime, n: int) -> List[datetime]:
    """Roughly monthly spacing."""
    total_months = (end.year - start.year) * 12 + (end.month - start.month) + 1
    step = max(1, total_months // max(1, n))
    dates = []
    y, m = start.year, start.month
    cur = start.replace(day=15, hour=14, minute=3, second=0, microsecond=0)
    while cur <= end and len(dates) < n:
        dates.append(cur)
        # advance by "step" months
        m += step
        y += (m - 1) // 12
        m = (m - 1) % 12 + 1
        cur = cur.replace(year=y, month=m)
    while len(dates) < n:
        cur += timedelta(days=1)
        dates.append(cur)
    return dates[:n]

def placeholders_from_string(s: str) -> List[str]:
    if not isinstance(s, str):
        return []
    # unique, preserving order
    seen, out = set(), []
    for m in re.findall(r"{([^}]+)}", s):
        if m not in seen:
            out.append(m); seen.add(m)
    return out

def apply_hashes(pattern: str, idx: int) -> str:
    """Replace {####...} tokens with zero-padded idx of that width."""
    def repl(m):
        hashes = m.group(1)
        return f"{idx:0{len(hashes)}d}"
    return re.sub(r"{(#+)}", repl, pattern)

def apply_tokens(pattern: str, tokens: Dict[str, Any]) -> str:
    """Replace {Key} tokens with values from tokens dict; leave unknowns intact."""
    def repl(m):
        k = m.group(1)
        return str(tokens.get(k, f"{{{k}}}"))
    s = re.sub(r"{([^#}][^}]*)}", repl, pattern)  # avoid {###} which we handle separately
    return s
def _pick(seq):  # safe random choice
    return random.choice(seq) if isinstance(seq, list) and seq else ""

def _word(wordlist, default):
    return _pick(wordlist) or default

def _load_llm_plugin(path):
    spec = importlib.util.spec_from_file_location("llm_plugin", path)
    if not spec or not spec.loader:
        raise RuntimeError(f"Cannot import llm plugin at {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "generate_text"):
        raise RuntimeError("LLM plugin must define generate_text(record, template) -> {'title','notes'}")
    return mod.generate_text

TITLE_PATTERNS = [
    "{Category}: {BoardCommittee} {MeetingType} meeting — {YYYY}-{MM}",
    "{Category}: {ProjectCode} update — {BoardCommittee}",
    "{BoardCommittee} minutes — {MeetingLocation}",
    "{ProjectCode} | {Category} — {BoardCommittee}"
]

NOTE_TEMPLATES = [
    "The {BoardCommittee} held a {MeetingType} session to review {topic}. {resolutions} were {action}.",
    "Discussion focused on {topic}. {resolutions} {action}; follow-ups assigned to {Author}.",
    "Committee reviewed {topic} and noted key risks. {resolutions} {action}."
]

ACTIONS = ["approved", "adopted", "endorsed", "ratified", "tabled for review"]
TOPIC_FALLBACKS = [
    "quarterly financial statements", "SOX audit findings",
    "risk appetite statement", "technology roadmap", "regulatory changes"
]

def synthesize_title_and_notes(rec: Dict[str, Any], tpl: Dict[str, Any]) -> Tuple[str, str]:
    # Pull commonly-used fields safely
    y = rec["SourceCreatedDate"][0:4]
    m = rec["SourceCreatedDate"][5:7]
    tokens = {
        **rec,
        "YYYY": y,
        "MM": m,
        "topic": _word(tpl.get("options", {}).get("Topics"), _pick(TOPIC_FALLBACKS)),
        "action": _pick(ACTIONS),
        "resolutions": f"{rec.get('ResolutionCount', _pick(['3','4','5']))} resolutions"
    }
    # Title
    title_raw = _pick(tpl.get("titlePatterns", [])) or _pick(TITLE_PATTERNS) or "{Category} — {ProjectCode}"
    title = apply_tokens(title_raw, tokens)
    title = apply_hashes(title, rec.get("_Seq_", 1))

    # Notes (1–2 short sentences)
    sent = apply_tokens(_pick(NOTE_TEMPLATES), tokens)
    # keep it brief
    if len(sent) > 220:
        sent = sent[:200].rsplit(" ", 1)[0] + "."

    return title, sent

# -----------------------------
# Template loading & validation
# -----------------------------
def load_template(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# Field name aliases to make older templates work. Best effort only.
AUTHOR_ALIASES = ["Author", "PreparedBy", "MinuteTaker", "MinuteTakers"]
CREATED_BY_ALIASES = ["SourceCreatedBy", "CreatedBy", "PreparedBy", "MinuteTaker"]
MODIFIED_BY_ALIASES = ["SourceLastModifiedBy", "ModifiedBy", "PreparedBy", "MinuteTaker"]

def pick_from_options(opts: Dict[str, Any], keys: List[str], default_pool: List[str]) -> str:
    """Pick one value from the first matching options list by key, else from default_pool."""
    for k in keys:
        if isinstance(opts.get(k), list) and opts[k]:
            return random.choice(opts[k])
    return random.choice(default_pool)

# -----------------------------
# Record synthesis
# -----------------------------
def build_external_id(pattern: str, class_code: str, when: datetime, idx: int, prefix: str) -> str:
    tokens = {
        "CLASSCODE": class_code,
        "YYYY": when.strftime("%Y"),
        "MM": when.strftime("%m"),
        "DD": when.strftime("%d"),
        "PREFIX": prefix or "",
    }
    s = apply_tokens(pattern, tokens)
    s = apply_hashes(s, idx)
    # collapse stray separators
    s = re.sub(r"--+", "-", s)
    s = s.strip("-")
    return s

def build_url(pattern: str, record: Dict[str, Any]) -> str:
    s = apply_tokens(pattern, record)
    s = apply_hashes(s, record.get("_Seq_", 1))
    # accept either connector-style ("Databricks:/...") or http(s)
    return s

def generate_records(tpl: Dict[str, Any], count: int, id_mode: str) -> List[Dict[str, Any]]:
    # Base parameters
    class_code = tpl.get("classCode", "COM-GEN")
    category_name = tpl.get("categoryName", "General")
    connector = tpl.get("connector", "Databricks")
    id_field = tpl.get("idField")  # optional legacy, e.g., "MinuteID"
    author_field = tpl.get("authorField")  # optional legacy
    title_template = tpl.get("titleTemplate", "")  # optional
    source_url_pattern = tpl.get("sourceUrlPattern", "{SourceURL}")
    id_pattern = tpl.get("idPattern", "{PREFIX}-{CLASSCODE}-{YYYY}{MM}{DD}-{####}")
    id_prefix = tpl.get("idPrefix", "MNM")
    cadence = tpl.get("cadence", "monthly")  # daily | monthly | mixed

    # Date range
    dr = tpl.get("dateRange", {"start": "2016-01-01", "end": "2025-10-31"})
    start = parse_dt(dr["start"]); end = parse_dt(dr["end"])
    if cadence == "daily":
        when_list = day_sequence(start, end, count)
    elif cadence == "mixed":
        # half monthly, half daily for variation
        half = max(1, count // 2)
        when_list = month_sequence(start, end, count - half) + day_sequence(start, end, half)
        random.shuffle(when_list)
        when_list = when_list[:count]
    else:
        when_list = month_sequence(start, end, count)

    # Options pool
    opts: Dict[str, Any] = tpl.get("options", {})

    # Prebuild numeric indices for {####}
    if id_mode == "random":
        indices = random.sample(range(1, count + 1), count)
    else:
        indices = list(range(1, count + 1))

    records: List[Dict[str, Any]] = []

    for i in range(count):
        when = when_list[i]
        idx = indices[i]

        # --- Choices (you can extend options freely in each template) ---
        # Category / ProjectCode
        category = opts.get("Category", [category_name])[0] if isinstance(opts.get("Category"), list) else category_name
        project_code = pick_from_options(opts, ["ProjectCode", "Projects", "ProjectCodes"], [f"{class_code}-CORE"])

        # Author / CreatedBy / ModifiedBy — use explicit "Author" if present; else fall back to common legacy keys
        author = pick_from_options(opts, AUTHOR_ALIASES, ["Hannah Kim", "Chloe Davies", "Lara Bennett"])
        created_by = pick_from_options(opts, CREATED_BY_ALIASES, [author])
        modified_by = pick_from_options(opts, MODIFIED_BY_ALIASES, [author])

        # External ID (unique)
        external_id = build_external_id(id_pattern, class_code, when, idx, id_prefix)

        # Dates: Created ≤ Modified
        created_date = (when + timedelta(days=random.randint(0, 7), hours=random.randint(0, 6))).astimezone(timezone.utc)
        modified_date = created_date + timedelta(days=random.randint(0, 14), hours=random.randint(0, 18))

        # Start record with mandatory mapping fields (exact names)
        rec: Dict[str, Any] = {
            "ExternalID": external_id,
            "Author": author,
            "SourceCreatedDate": iso_z(created_date),
            "SourceCreatedBy": created_by,
            "SourceLastModifiedDate": iso_z(modified_date),
            "SourceLastModifiedBy": modified_by,
            # We'll compute SourceURL after we know everything else:
            # "SourceURL": "...",
            # Helpful extra mapping inputs:
            "Category": category,
            "ProjectCode": project_code,
            # Useful context fields (becomes source properties in rpsubmit):
            "ClassCode": class_code,
            "CategoryName": category_name,
            "Connector": connector,
            "_Seq_": idx  # internal helper for {####} in patterns
        }

        # For legacy templates that expect an internal ID field (e.g., "MinuteID"), mirror ExternalID.
        if id_field and isinstance(id_field, str) and id_field.strip():
            rec[id_field] = external_id

        # Include additional option-driven fields (positions, notes, etc.)
        for k, v in opts.items():
            if isinstance(v, list) and v:
                # Avoid clobbering mandatory mapping fields
                if k not in rec:
                    rec[k] = random.choice(v)

        # Title (optional, for readability in JSON; the loader will compute title from mapping independently)
        if title_template:
            rec["Title"] = apply_hashes(apply_tokens(title_template, rec), rec["_Seq_"])

        # Source URL (supports connector-style or https). Allow {ExternalID} or other placeholders.
        rec["SourceURL"] = build_url(source_url_pattern, rec)

        # Done
        records.append(rec)

    return records

# -----------------------------
# Main
# -----------------------------
def main():
    args = parse_args()
    tpl = load_template(args.template)

    if args.seed is not None:
        random.seed(args.seed)

    count = args.count or int(tpl.get("count", 100))
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    recs = generate_records(tpl, count, id_mode=args.id_mode)

    # Final validation: ensure required mapping names exist on every record
    required = [
        "ExternalID", "Author",
        "SourceCreatedDate", "SourceCreatedBy",
        "SourceLastModifiedDate", "SourceLastModifiedBy",
        "SourceURL"
    ]
    for n, r in enumerate(recs, start=1):
        missing = [k for k in required if k not in r or r[k] in ("", None)]
        if missing:
            raise ValueError(f"Record {n} missing required mapping fields: {missing}")

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(recs, f, ensure_ascii=False, indent=2)

    print(f"✅ Wrote {len(recs)} records to {out_path}")
    print("Tip: submit with rpsubmit.py using your config mapping.")


if __name__ == "__main__":
    main()
