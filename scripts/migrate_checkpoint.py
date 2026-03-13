#!/usr/bin/env python3
"""Migrate existing checkpoints so they are compatible with torch.load(weights_only=True).

Fixes two issues in lora_adapters.pt:
  1. peft_type: PeftType enum -> str (e.g. "LORA")
  2. target_modules: set -> sorted list

zcl_links.pt contains only state_dicts (pure tensors) and needs no migration,
but the script will verify it loads cleanly with weights_only=True.

Usage:
    # Single checkpoint directory
    python scripts/migrate_checkpoint.py training/checkpoints/step-000005

    # Multiple checkpoints
    python scripts/migrate_checkpoint.py training/checkpoints/step-*

    # Dry-run (report only, no writes)
    python scripts/migrate_checkpoint.py --dry-run training/checkpoints/step-000005
"""

import argparse
import os
import sys
import shutil

import torch


def _sanitize_peft_config(cfg: dict) -> tuple[dict, list[str]]:
    """Return (sanitized_cfg, list_of_changes)."""
    changes = []
    out = dict(cfg)
    for k, v in out.items():
        if isinstance(v, set):
            out[k] = sorted(v)
            changes.append(f"  {k}: set -> sorted list")
        elif hasattr(v, "value"):  # enum
            out[k] = v.value
            changes.append(f"  {k}: {type(v).__name__}({v!r}) -> {v.value!r}")
    return out, changes


def migrate_lora(path: str, *, dry_run: bool) -> bool:
    """Migrate lora_adapters.pt.  Returns True if changes were made (or would be)."""
    # First try safe load — if it works, nothing to do.
    try:
        torch.load(path, map_location="cpu", weights_only=True)
        print(f"  [OK] {path} already compatible")
        return False
    except Exception:
        pass

    # Load with weights_only=False to access the original payload.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    peft_config = payload.get("peft_config")
    if peft_config is None:
        print(f"  [SKIP] {path}: no peft_config key, nothing to sanitize")
        return False

    sanitized, changes = _sanitize_peft_config(peft_config)
    if not changes:
        print(f"  [SKIP] {path}: peft_config already clean, issue is elsewhere")
        return False

    print(f"  [FIX] {path}:")
    for c in changes:
        print(f"       {c}")

    if dry_run:
        print("         (dry-run, not writing)")
        return True

    # Backup then overwrite.
    backup = path + ".bak"
    if not os.path.exists(backup):
        shutil.copy2(path, backup)
        print(f"         backup -> {backup}")

    payload["peft_config"] = sanitized
    torch.save(payload, path)

    # Verify round-trip.
    torch.load(path, map_location="cpu", weights_only=True)
    print("         verified: weights_only=True OK")
    return True


def verify_zcl(path: str) -> bool:
    """Verify zcl_links.pt loads with weights_only=True."""
    try:
        torch.load(path, map_location="cpu", weights_only=True)
        print(f"  [OK] {path} compatible")
        return True
    except Exception as e:
        print(f"  [WARN] {path} NOT compatible: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Migrate checkpoints for weights_only=True")
    parser.add_argument("dirs", nargs="+", help="Checkpoint directory(ies), e.g. training/checkpoints/step-000005")
    parser.add_argument("--dry-run", action="store_true", help="Report issues without modifying files")
    args = parser.parse_args()

    any_fixed = False
    any_warn = False

    for d in args.dirs:
        if not os.path.isdir(d):
            print(f"[SKIP] not a directory: {d}")
            continue
        print(f"\n=== {d} ===")

        lora_file = os.path.join(d, "lora_adapters.pt")
        if os.path.exists(lora_file):
            if migrate_lora(lora_file, dry_run=args.dry_run):
                any_fixed = True
        else:
            print(f"  [SKIP] lora_adapters.pt not found")

        zcl_file = os.path.join(d, "zcl_links.pt")
        if os.path.exists(zcl_file):
            if not verify_zcl(zcl_file):
                any_warn = True
        else:
            print(f"  [SKIP] zcl_links.pt not found")

    print()
    if args.dry_run and any_fixed:
        print("Re-run without --dry-run to apply fixes.")
    elif any_fixed:
        print("Done. Original files backed up as *.bak.")
    elif any_warn:
        print("Some files have issues that need manual inspection.")
    else:
        print("All checkpoints are already compatible.")


if __name__ == "__main__":
    main()
