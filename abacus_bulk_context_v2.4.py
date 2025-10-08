#!/usr/bin/env python3
r"""
Abacus.AI Multi-FG Context Updater — v2.2
-----------------------------------------
Purpose
  Update a ChatLLM/DataLLM model's structured-data *context* for many Feature Groups (FGs)
  using a single JSON “source of truth” (global/table/column maps + template families).

What this script CAN change (additive & safe):
  1) data_prompt_context            (append-only; de-duped; blank line separator)
  2) data_prompt_table_context      (merge by tableName; OVERWRITE targeted values)
  3) data_prompt_column_context     (merge by tableName.column; OVERWRITE targeted values)
  4) data_columns_to_ignore         (set-union)
  5) data_feature_group_ids         (set-union; unless --only-contexts)

What this script NEVER changes:
  • Any other model settings. Unstructured/vector retrievers and unrelated config remain untouched.

Default behavior (IMPORTANT):
  • Running with NO FLAGS applies changes immediately to the target model.
  • Use --dry-run to preview without applying.
  • You can restrict scope with --company, --tables, or --fg-ids.

Credentials (MANDATORY – paste or use env vars):
  At the top of this file, set API_KEY, PROJECT_ID, MODEL_ID OR set env vars:
    ABACUS_API_KEY, ABACUS_PROJECT_ID, ABACUS_MODEL_ID

Windows default JSON path (can override with --json):
  F:\Documents\docs\AI\Databases\Kbase_Prime\Platform_Config\model config update\structured table context descriptions_optimized.json

SDK calls (verified against Abacus.AI Python SDK):
  • ApiClient(api_key=...)                    → create client
  • client.describe_model(model_id=...)       → read current config
  • ChatLLMTrainingConfig(**kwargs)           → build training config payload
  • client.set_model_training_config(model_id=..., training_config=...) → apply

Usage examples:
  python abacus_bulk_context_v2.2.py                                # APPLY (default)
  python abacus_bulk_context_v2.2.py --dry-run                      # preview only (read-only)
  python abacus_bulk_context_v2.2.py --dry-run --dump-contexts      # preview + write ./contexts_preview/*
  python abacus_bulk_context_v2.2.py --company UNH                  # target UNH_* tables
  python abacus_bulk_context_v2.2.py --tables UNH_XBRL_Facts_Enriched UNH_forms345_ownership_lines --dry-run
  python abacus_bulk_context_v2.2.py --list-all-fgs                 # list (id, tableName, name)

Key flags:
  --dry-run         Preview only; no model updates (read-only).
  --dump-contexts   Write resolved contexts to ./contexts_preview/ for audit.
  --only-contexts   Update only contexts (no FG attachment changes).
  --only-fgs        Update only FG attachments (no context changes).
  --company PREFIX  Filter targets by company prefix (e.g., UNH).
  --tables ...      Explicit tableName targets (space-separated).
  --fg-ids ...      Explicit FG id targets (space-separated).
  --verbose, -v     Verbose logging.

"""
from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
import re

# Abacus.AI SDK
from abacusai import ApiClient, ChatLLMTrainingConfig

# =========================
# Configuration placeholders
# =========================
API_KEY    = os.getenv("ABACUS_API_KEY")    or ""
PROJECT_ID = os.getenv("ABACUS_PROJECT_ID") or ""
MODEL_ID   = os.getenv("ABACUS_MODEL_ID")   or ""

DEFAULT_JSON_WINDOWS = r"F:\Documents\docs\AI\Databases\Kbase_Prime\Platform_Config\model config update\structured table context descriptions_optimized.json"

# --- DEFAULTS used when no flags are provided ---
DEFAULTS = {
    "COMPANY": None,            # e.g., "UNH" or None for ALL in JSON
    "TABLES": None,             # e.g., ["UNH_XBRL_Facts_Enriched", ...]
    "FG_IDS": None,             # e.g., ["fg_123", "fg_456"]
    "DUMP_CONTEXTS": False,     # True -> always write contexts_preview/*
    "VERBOSE": True,            # default console verbosity
    "ENABLE_TEMPLATES": True,   # template fallback for SEC_XBRL & SEC_345
}

# =====================
# Logging setup helpers
# =====================

def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        level=level,
    )

# =====================
# JSON loading & checks
# =====================

@dataclass
class ContextPackage:
    feature_group_map: Dict[str, str]
    table_context_map: Dict[str, str]
    column_context_map: Dict[str, str]
    global_context: List[str]
    templates: Dict[str, dict]

def validate_context_package(pkg: ContextPackage) -> Dict[str, object]:
    """Unit-ish checks for key hygiene and basic consistency."""
    issues: Dict[str, object] = {}
    fg_tables = set(pkg.feature_group_map.keys())

    # table_context_map keys present in FG map
    table_keys = set(pkg.table_context_map.keys())
    miss_fg = sorted(table_keys - fg_tables)
    miss_table = sorted(fg_tables - table_keys)
    if miss_fg:
        issues["table_context_without_fg"] = miss_fg
    if miss_table:
        issues["fg_without_table_context"] = miss_table

    # column_context_map key format and table presence
    bad_cols: List[str] = []
    missing_tbls: Set[str] = set()
    for k in (pkg.column_context_map or {}).keys():
        if "." not in k:
            bad_cols.append(k)
        else:
            t = k.split(".", 1)[0]
            if t not in fg_tables:
                missing_tbls.add(t)
    if bad_cols:
        issues["invalid_column_keys"] = bad_cols[:50]
    if missing_tbls:
        issues["column_tables_missing_in_fg_map"] = sorted(missing_tbls)

    return issues


def load_context_package(path: str) -> ContextPackage:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return ContextPackage(
        feature_group_map=raw.get("feature_group_map", {}),
        table_context_map=raw.get("table_context_map", {}),
        column_context_map=raw.get("column_context_map", {}),
        global_context=raw.get("global_context", []),
        templates=raw.get("templates", {}),
    )

# =====================
# Abacus SDK utilities
# =====================

# Defensive attribute fetcher to handle snakeCase/camelCase variants in SDK objects
def _safe_attr(obj, *names, default=None):
    for n in names:
        if obj is None:
            break
        if isinstance(obj, dict) and n in obj:
            return obj[n]
        if hasattr(obj, n):
            return getattr(obj, n)
    return default


def list_all_feature_groups(client: ApiClient, project_id: str) -> List[dict]:
    """List FGs visible to the *project* (id, tableName, name)."""
    fgs = []
    # Prefer the project-scoped method if available
    list_proj_fn = getattr(client, "list_project_feature_groups", None)
    if callable(list_proj_fn):
        page = list_proj_fn(project_id=project_id, limit=200)
    else:
        # Fallback for older SDKs (may return all org FGs)
        list_fg_fn = getattr(client, "list_feature_groups", None)
        if callable(list_fg_fn):
            page = list_fg_fn(limit=200)
        else:
            logging.error("SDK does not expose list_project_feature_groups or list_feature_groups")
            page = []
    fgs.extend(page)

    return [
        {
            "feature_group_id": _safe_attr(x, "feature_group_id", "featureGroupId", "id"),
            "tableName": _safe_attr(x, "tableName", "table_name", "name"),
            "name": _safe_attr(x, "name"),
        }
        for x in fgs
    ]

# ==================================
# Target resolution (company/tables)
# ==================================

def prefix_from_table(table_name: str) -> str:
    return table_name.split("_", 1)[0] if "_" in table_name else ""

def extract_ticker_from_xbrl_table(table_name: str) -> Optional[str]:
    # Only treat names like TICKER_XBRL_* as XBRL families, and return TICKER
    m = re.match(r"^([A-Z0-9]+)_XBRL_", table_name)
    return m.group(1) if m else None


def filter_targets(
    fg_map: Dict[str, str],
    company: Optional[str],
    tables: Optional[List[str]],
    fg_ids: Optional[List[str]],
) -> Dict[str, str]:
    """
    Returns subset_fg_map_by_tableName
    """
    subset: Dict[str, str] = {}

    if tables:
        for t in tables:
            if t in fg_map:
                subset[t] = fg_map[t]
    elif fg_ids:
        # Inverse lookup
        inv = {v: k for k, v in fg_map.items()}
        for fid in fg_ids:
            t = inv.get(fid)
            if t:
                subset[t] = fid
    elif company:
        for t, fid in fg_map.items():
            if prefix_from_table(t).upper() == (company or "").upper():
                subset[t] = fid
    else:
        # default: all fgs in JSON
        subset = dict(fg_map)

    return subset

# =============================
# Template expansion (families)
# =============================

@dataclass
class ResolvedContexts:
    global_text: str
    table_ctx: Dict[str, str]
    column_ctx: Dict[str, str]
    ignore_cols: List[str]
    template_log: Dict[str, Dict[str, List[str]]]  # family → {tables|columns|ignore}


def _detect_family_tokens(family: dict, target_tables: Set[str]) -> Set[str]:
    """Return detected tokens (e.g., company tickers) for a template family."""
    placeholder = family.get("placeholder", "{company_ticker}")
    tokens: Set[str] = set()

    # Option 1: explicit table_regex with one capturing group
    regex = family.get("table_regex")
    if isinstance(regex, str):
        try:
            cre = re.compile(regex)
            for t in target_tables:
                m = cre.match(t)
                if m and m.groups():
                    tokens.add(m.group(1))
        except re.error:
            logging.warning("Invalid table_regex in template family; skipping regex detection.")

    # Option 2: infer from templated keys
    def capture_from_template_key(tkey: str) -> Set[str]:
        # Build a regex by replacing the placeholder with a capturing group
        esc = re.escape(placeholder)
        pat = "^" + re.escape(tkey).replace(esc, r"([A-Za-z0-9]+)") + "$"
        cre = re.compile(pat)
        found: Set[str] = set()
        for t in target_tables:
            m = cre.match(t)
            if m:
                found.add(m.group(1))
        return found

    for section in ("table", "columns", "ignore"):
        sec = family.get(section) or {}
        # Support both dict (table/columns) and list (ignore)
        if isinstance(sec, dict):
            iterable = sec.keys()
        elif isinstance(sec, list):
            iterable = sec
        else:
            iterable = []
        for tkey in iterable:
            # tkey can be "TABLE" or "TABLE.col"; take table portion for token detection
            table_part = tkey.split(".", 1)[0]
            tokens |= capture_from_template_key(table_part)

    return tokens


def resolve_contexts(
    pkg: ContextPackage,
    target_tables: Set[str],
    enable_templates: bool = True,
) -> ResolvedContexts:

    templates = pkg.templates or {}
    families = {k: v for k, v in templates.items()} if enable_templates else {}

    # --- Detect tokens per family and a global union for {company_ticker} expansion in global context ---
    family_tokens: Dict[str, Set[str]] = {name: _detect_family_tokens(fam, target_tables) for name, fam in families.items()}
    detected_tokens_union: List[str] = sorted({tok for toks in family_tokens.values() for tok in toks})

    # --- Global context ---
    lines = list(pkg.global_context or [])
    joined: List[str] = []
    for s in lines:
        if "{company_ticker}" in s and detected_tokens_union:
            for t in detected_tokens_union:
                joined.append(s.replace("{company_ticker}", t))
        else:
            joined.append(s)

    # de-duplicate while preserving order
    seen = set()
    deduped = []
    for s in joined:
        if s not in seen:
            seen.add(s)
            deduped.append(s)
    global_text = "\n".join(deduped).strip()

    # --- Start from explicit entries (limited to target tables) ---
    table_ctx: Dict[str, str] = {k: v for k, v in (pkg.table_context_map or {}).items() if k in target_tables}
    column_ctx: Dict[str, str] = {k: v for k, v in (pkg.column_context_map or {}).items() if k.split(".", 1)[0] in target_tables}

    # --- Template fallback (fill-only) ---
    template_log: Dict[str, Dict[str, List[str]]] = {}

    for fam_name, fam in families.items():
        placeholder = fam.get("placeholder", "{company_ticker}")
        tokens = sorted(family_tokens.get(fam_name, set()))
        log_entry = {"tables": [], "columns": [], "ignore": []}

        # TABLES
        for t in sorted(target_tables):
            if t in table_ctx:
                continue  # explicit wins
            for tk, tv in (fam.get("table") or {}).items():
                for token in tokens:
                    concrete_key = tk.replace(placeholder, token)
                    if concrete_key == t:
                        table_ctx[t] = tv.replace(placeholder, token)
                        log_entry["tables"].append(t)

        # COLUMNS
        for ck, cv in (fam.get("columns") or {}).items():
            for token in tokens:
                concrete_key = ck.replace(placeholder, token)
                table_part = concrete_key.split(".", 1)[0]
                if table_part in target_tables and concrete_key not in column_ctx:
                    column_ctx[concrete_key] = cv.replace(placeholder, token)
                    log_entry["columns"].append(concrete_key)

        # IGNORE
        ignore_cols: List[str] = []
        for ik in (fam.get("ignore") or []):
            for token in tokens:
                concrete_key = ik.replace(placeholder, token)
                if concrete_key.split(".", 1)[0] in target_tables:
                    ignore_cols.append(concrete_key)

        template_log[fam_name] = log_entry
        # Aggregate ignore later with others
        # (we add ignore from all families outside the loop)
    # Gather ignore from all families
    ignore_all: List[str] = []
    for fam_name, fam in families.items():
        placeholder = fam.get("placeholder", "{company_ticker}")
        tokens = sorted(family_tokens.get(fam_name, set()))
        for ik in (fam.get("ignore") or []):
            for token in tokens:
                concrete_key = ik.replace(placeholder, token)
                if concrete_key.split(".", 1)[0] in target_tables:
                    ignore_all.append(concrete_key)

    # De-dup ignore list while preserving order
    seen_i = set()
    dedup_ignore: List[str] = []
    for k in ignore_all:
        if k not in seen_i:
            seen_i.add(k)
            dedup_ignore.append(k)

    return ResolvedContexts(
        global_text=global_text,
        table_ctx=table_ctx,
        column_ctx=column_ctx,
        ignore_cols=dedup_ignore,
        template_log=template_log,
    )

# ==============================
# Merging & diffing (idempotent)
# ==============================

@dataclass
class ExistingConfig:
    data_prompt_context: str
    data_prompt_table_context: Dict[str, str]
    data_prompt_column_context: Dict[str, str]
    data_columns_to_ignore: List[str]
    existing_feature_group_ids: List[str]


def read_existing_config(client: ApiClient, model_id: str) -> ExistingConfig:
    desc = client.describe_model(model_id=model_id)
    # NEW: pull the nested training config first (snake or camel)
    training_cfg = _safe_attr(desc, "training_config", "trainingConfig", default={}) or {}

    existing_global  = _safe_attr(training_cfg, "data_prompt_context", "dataPromptContext", default="") or ""
    existing_tables  = _safe_attr(training_cfg, "data_prompt_table_context", "dataPromptTableContext", default={}) or {}
    existing_columns = _safe_attr(training_cfg, "data_prompt_column_context", "dataPromptColumnContext", default={}) or {}
    existing_ignore  = _safe_attr(training_cfg, "data_columns_to_ignore", "dataColumnsToIgnore", default=[]) or []

    # FG ids: keep your existing logic (it’s on the model object)
    raw_fgs = _safe_attr(desc, "data_llm_feature_groups", "dataLlmFeatureGroups", default=[]) or []
    existing_fg_ids = []
    for fg in raw_fgs:
        fgid = _safe_attr(fg, "feature_group_id", "featureGroupId", "id")
        if fgid:
            existing_fg_ids.append(fgid)

    return ExistingConfig(
        data_prompt_context=existing_global,
        data_prompt_table_context=existing_tables,
        data_prompt_column_context=existing_columns,
        data_columns_to_ignore=existing_ignore,
        existing_feature_group_ids=existing_fg_ids,
    )



@dataclass
class DiffPreview:
    global_added: bool
    table_added: List[str]
    column_added: List[str]
    ignore_added: List[str]
    fg_ids_added: List[str]


def _merge_lists(existing: List[str], new: List[str]) -> List[str]:
    seen = set(existing or [])
    out = list(existing or [])
    for x in (new or []):
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _merge_dicts(existing: dict, new: dict) -> dict:
    out = dict(existing or {})
    for k, v in (new or {}).items():
        if v is None:
            continue
        out[k] = v  # always prefer our new value for targeted keys
    return out


def _merge_global(existing: str, new: str) -> str:
    ex = (existing or "").strip()
    nn = (new or "").strip()
    if not nn:
        return ex
    if ex and nn and nn not in ex:
        return ex + "\n\n" + nn
    if not ex:
        return nn
    return ex

# ----------------------------
# Word-count guardrail helpers
# ----------------------------
def _wc_str(s: str) -> int:
    # split on any whitespace to mimic backend "word" counting
    return len(re.findall(r"\S+", s or ""))

def _wc_dict(d: dict) -> int:
    if not d:
        return 0
    # Emulate server-side serialization ("key: value" lines), then count words
    joined = "\n".join(f"{k}: {v}" for k, v in d.items())
    return _wc_str(joined)

def _log_word_counts(kwargs: dict, limit: int | None = None) -> tuple[int, int, int, int]:
    g = kwargs.get("data_prompt_context", "") or ""
    t = kwargs.get("data_prompt_table_context", {}) or {}
    c = kwargs.get("data_prompt_column_context", {}) or {}
    g_wc, t_wc, c_wc = _wc_str(g), _wc_dict(t), _wc_dict(c)
    total = g_wc + t_wc + c_wc
    logging.debug(
        "Word counts — global=%d, table=%d, column=%d, total=%d%s",
        g_wc, t_wc, c_wc, total, f" (limit={limit})" if limit else ""
    )
    return g_wc, t_wc, c_wc, total

def diff_preview(existing: ExistingConfig, resolved: ResolvedContexts, new_fg_ids: List[str]) -> DiffPreview:
    global_added = bool(resolved.global_text and (resolved.global_text not in (existing.data_prompt_context or "")))
    table_added = [k for k in resolved.table_ctx.keys() if existing.data_prompt_table_context.get(k) != resolved.table_ctx[k]]
    column_added = [k for k in resolved.column_ctx.keys() if existing.data_prompt_column_context.get(k) != resolved.column_ctx[k]]

    e_ignore = set(existing.data_columns_to_ignore or [])
    n_ignore = set(resolved.ignore_cols or [])
    ignore_added = sorted(list(n_ignore - e_ignore))

    e_fg = set(existing.existing_feature_group_ids or [])
    n_fg = set(new_fg_ids or [])
    fg_ids_added = sorted(list(n_fg - e_fg))

    return DiffPreview(
        global_added=global_added,
        table_added=table_added,
        column_added=column_added,
        ignore_added=ignore_added,
        fg_ids_added=fg_ids_added,
    )

# =========================
# Validation (non-fatal)
# =========================

def validate_keys_against_targets(pkg: ContextPackage, target_tables: Set[str], resolved: ResolvedContexts) -> None:
    """Log non-fatal issues like keys outside target scope."""
    # Ensure every resolved table context key is in target set
    for k in resolved.table_ctx.keys():
        if k not in target_tables:
            logging.warning("Resolved table context key not in target scope: %s", k)

    # Ensure resolved column contexts have a valid table prefix
    for k in resolved.column_ctx.keys():
        t = k.split(".", 1)[0]
        if t not in target_tables:
            logging.warning("Resolved column context key outside target scope: %s", k)

    # Ensure ignore entries parse and are in scope
    for k in resolved.ignore_cols:
        if "." not in k:
            logging.warning("Ignore entry missing dot (table.column): %s", k)
            continue
        t = k.split(".", 1)[0]
        if t not in target_tables:
            logging.warning("Ignore entry references table not in scope: %s", k)

# =========================
# Main execution flow
# =========================

def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Bulk update an Abacus.AI ChatLLM/DataLLM model's structured-data contexts "
            "from a canonical JSON package (apply by default; use --dry-run to preview)."
        )
    )
    p.add_argument("--api-key", default=API_KEY, help="Abacus.AI API key (or ABACUS_API_KEY env var)")
    p.add_argument("--project-id", default=PROJECT_ID, help="Target Abacus.AI project id (or ABACUS_PROJECT_ID)")
    p.add_argument("--model-id", default=MODEL_ID, help="Target ChatLLM/DataLLM model id (or ABACUS_MODEL_ID)")
    p.add_argument("--json", default=DEFAULT_JSON_WINDOWS, help="Path to optimized context JSON package")
    p.add_argument("--company", help="Filter by company prefix (e.g., UNH)")
    p.add_argument("--tables", nargs="*", help="Explicit list of tableName targets")
    p.add_argument("--fg-ids", nargs="*", help="Explicit list of feature_group_id targets")
    p.add_argument("--only-contexts", action="store_true", help="Update only contexts (no FG attachments)")
    p.add_argument("--only-fgs", action="store_true", help="Update only FG attachments (no contexts)")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--dry-run", action="store_true", help="Preview changes only; do not update the model")
    group.add_argument("--apply", action="store_true", help="Apply changes (DEFAULT if neither flag is given)")
    p.add_argument("--dump-contexts", action="store_true", help="Dump resolved contexts into ./contexts_preview/")
    p.add_argument("--list-all-fgs", action="store_true", help="List feature groups in the project (id, tableName, name)")
    p.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    p.add_argument("--word-limit", type=int, default=5625,
                   help="Soft cap for prompt word count (global + table + column). Default: 5625.")
    p.add_argument("--enforce-word-limit", action="store_true",
                   help="If set, abort apply when computed word count exceeds --word-limit.")

    args = p.parse_args(argv)

    # Apply-by-default semantics
    if not args.dry_run and not args.apply:
        args.apply = True

    # Optionally override with DEFAULTS when user didn't pass flags
    if args.company is None:
        args.company = DEFAULTS["COMPANY"]
    if args.tables is None:
        args.tables = DEFAULTS["TABLES"]
    if args.fg_ids is None:
        args.fg_ids = DEFAULTS["FG_IDS"]
    if not args.dump_contexts:
        args.dump_contexts = DEFAULTS["DUMP_CONTEXTS"]
    if not args.verbose:
        args.verbose = DEFAULTS["VERBOSE"]

    setup_logging(args.verbose)

    if not args.api_key or args.api_key.startswith("<PASTE_"):
        logging.warning("No API key provided. Set --api-key or ABACUS_API_KEY.")
    if not args.project_id or args.project_id.startswith("<PASTE_"):
        logging.warning("No PROJECT_ID provided. Set --project-id or ABACUS_PROJECT_ID.")
    if not args.model_id or args.model_id.startswith("<PASTE_"):
        logging.warning("No MODEL_ID provided. Set --model-id or ABACUS_MODEL_ID.")

    # Load JSON
    json_path = args.json
    if not os.path.exists(json_path):
        # try local mirror path in this environment
        alt = str(Path(__file__).parent / "structured table context descriptions_optimized.json")
        if os.path.exists(alt):
            json_path = alt
            logging.info("Using local JSON at %s", alt)
        else:
            logging.error("JSON not found: %s", args.json)
            return 2

    pkg = load_context_package(json_path)
    logging.info("JSON loaded: %s | FG entries=%d | table keys=%d | column keys=%d | template families=%d",
                 json_path, len(pkg.feature_group_map), len(pkg.table_context_map),
                 len(pkg.column_context_map), len(pkg.templates or {}))

    # Basic validation
    issues = validate_context_package(pkg)
    if issues:
        logging.warning("Validation issues detected: %s", list(issues.keys()))
        for k, v in issues.items():
            logging.warning("  %s: %s", k, (v if isinstance(v, list) else v))
    else:
        logging.info("Validation checks passed (no key mismatches detected)")

    # Init client (NOTE: ApiClient only needs the API key; project is passed per-call when needed)
    client = ApiClient(args.api_key)

    if args.list_all_fgs:
        if not args.project_id or args.project_id.startswith("<PASTE_"):
            logging.error("--list-all-fgs requires a valid --project-id (or ABACUS_PROJECT_ID).")
            return 2
        fgs = list_all_feature_groups(client, project_id=args.project_id)
        logging.info("Feature Groups visible in project: %d", len(fgs))
        for f in fgs:
            logging.info("FG %s | table=%s | name=%s", f.get("feature_group_id"), f.get("tableName"), f.get("name"))

    # Determine targets
    target_map = filter_targets(pkg.feature_group_map, args.company, args.tables, args.fg_ids)
    target_tables = set(target_map.keys())

    # Informational logs
    xbrl_tickers = sorted({t for t in (extract_ticker_from_xbrl_table(tn) for tn in target_tables) if t})
    prefix_groups = sorted({prefix_from_table(tn) for tn in target_tables})
    if xbrl_tickers:
        logging.info("Detected SEC XBRL tickers in target scope: %s", ", ".join(xbrl_tickers))
    logging.info("Prefix groups present in target scope: %s", ", ".join(prefix_groups))

    # Resolve contexts (incl. template fallback: SEC_XBRL, SEC_345 if present)
    resolved = resolve_contexts(pkg, target_tables, enable_templates=DEFAULTS["ENABLE_TEMPLATES"])
    validate_keys_against_targets(pkg, target_tables, resolved)

    # Dump contexts if requested
    if args.dump_contexts:
        outdir = Path("contexts_preview")
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "global.txt").write_text(resolved.global_text, encoding="utf-8")
        with open(outdir / "table_context.json", "w", encoding="utf-8") as f:
            json.dump(resolved.table_ctx, f, ensure_ascii=False, indent=2)
        with open(outdir / "column_context.json", "w", encoding="utf-8") as f:
            json.dump(resolved.column_ctx, f, ensure_ascii=False, indent=2)
        with open(outdir / "ignore.json", "w", encoding="utf-8") as f:
            json.dump(resolved.ignore_cols, f, ensure_ascii=False, indent=2)
        logging.info("Dumped resolved contexts to %s", outdir)


    # Compute desired FG set (union of existing + targets) unless --only-contexts
    existing = read_existing_config(client, args.model_id)
    final_fg_ids = existing.existing_feature_group_ids
    if not args.only_contexts:
        desired = list(target_map.values())
        final_fg_ids = _merge_lists(existing.existing_feature_group_ids, desired)

    # Preview changes
    preview = diff_preview(existing, resolved, new_fg_ids=list(target_map.values()))
    logging.info(
        "Preview — global_added=%s | +tables=%d | +columns=%d | +ignore=%d | +fg_ids=%d",
        preview.global_added,
        len(preview.table_added),
        len(preview.column_added),
        len(preview.ignore_added),
        len(preview.fg_ids_added),
    )
    logging.debug("  Table keys changed: %s", preview.table_added)
    logging.debug("  Column keys changed: %s", preview.column_added)
    logging.debug("  Ignore keys added: %s", preview.ignore_added)

    # Build kwargs to apply (additively and *only* the allowed fields)
    kwargs = {}
    if not args.only_fgs:
        kwargs.update(dict(
            data_prompt_context=_merge_global(existing.data_prompt_context, resolved.global_text),
            data_prompt_table_context=_merge_dicts(existing.data_prompt_table_context, resolved.table_ctx),
            data_prompt_column_context=_merge_dicts(existing.data_prompt_column_context, resolved.column_ctx),
            data_columns_to_ignore=_merge_lists(existing.data_columns_to_ignore, resolved.ignore_cols),
        ))
    if not args.only_contexts and final_fg_ids:
        kwargs["data_feature_group_ids"] = final_fg_ids

    # --- OPTIONAL GUARDRAIL: compute & optionally enforce word limits ---
    limit = args.word_limit or 0
    g_wc, t_wc, c_wc, total_wc = _log_word_counts(kwargs, limit)

    if args.enforce_word_limit and limit > 0:
        # Enforce both per-field and combined (the backend error names ColumnContext,
        # but experience suggests it may be counting a serialization of all prompts).
        if c_wc > limit:
            logging.error("ColumnContext count %d exceeds limit %d.", c_wc, limit)
            return 3
        if total_wc > limit:
            logging.error("Combined prompt count %d exceeds limit %d. Use --company to scope or trim global.", total_wc, limit)
            return 3

    # Dry-run vs apply
    if args.apply:
        logging.info("Applying training config update...")
        cfg = ChatLLMTrainingConfig(**kwargs)
        client.set_model_training_config(model_id=args.model_id, training_config=cfg)
        logging.info("✅ Training config updated.")
    else:
        logging.info("DRY RUN — no changes applied. Use --apply (or omit --dry-run) to push.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
