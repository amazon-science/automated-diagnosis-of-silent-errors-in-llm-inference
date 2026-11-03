#!/usr/bin/env python3
import argparse
import os
import re
import sys
from typing import Dict, List

import torch

from tqdm import tqdm


ACTIVATION_PREFIX_RE = re.compile(r"^.*activation_rank\d+_")


def load_call_counts(call_sequence_path: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    with open(call_sequence_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("root."):
                line = line[len("root."):]
            module_key = line.replace(".", "_")
            counts[module_key] = counts.get(module_key, 0) + 1
    return counts


def module_key_from_filename(filename: str, prefix: str) -> str:
    if filename.endswith("_input.pt"):
        stem = filename[: -len("_input.pt")]
    elif filename.endswith("_output.pt"):
        stem = filename[: -len("_output.pt")]
    else:
        return ""
    if prefix:
        if not stem.startswith(prefix):
            return ""
        stem = stem[len(prefix) :]
    else:
        stem = ACTIVATION_PREFIX_RE.sub("", stem)
    return stem


def prepare_pickle_shims() -> None:
    try:
        import transformers.cache_utils as cache_utils
    except Exception:
        return

    extra_names = ["DynamicLayer", "EncoderDecoderCache"]
    for name in extra_names:
        if not hasattr(cache_utils, name):
            dummy = type(name, (), {"__module__": cache_utils.__name__})
            setattr(cache_utils, name, dummy)

    safe_globals = [
        obj for obj in vars(cache_utils).values() if isinstance(obj, type)
    ]
    if safe_globals:
        torch.serialization.add_safe_globals(safe_globals)


def slice_activation_list(
    activations: List, step_idx: int, call_count: int, filename: str
) -> List:
    total = len(activations)
    if call_count <= 0:
        call_count = 1

    if total % call_count == 0:
        steps = total // call_count
        if step_idx >= steps:
            raise IndexError(
                f"{filename}: step_idx {step_idx} out of range (steps={steps})."
            )
        start = step_idx * call_count
        end = start + call_count
        return activations[start:end]

    if step_idx >= total:
        raise IndexError(
            f"{filename}: step_idx {step_idx} out of range (len={total})."
        )
    return [activations[step_idx]]


def process_folder(
    input_dir: str,
    call_sequence_path: str,
    output_root: str,
    step_idx: int,
    prefix: str,
) -> None:
    call_counts = load_call_counts(call_sequence_path)

    output_dir = os.path.join(output_root, os.path.basename(os.path.normpath(input_dir)))
    os.makedirs(output_dir, exist_ok=True)

    pt_files = [
        f for f in os.listdir(input_dir) if f.endswith(".pt") and os.path.isfile(os.path.join(input_dir, f))
    ]
    if not pt_files:
        raise FileNotFoundError(f"No .pt files found in {input_dir}")

    skipped = 0
    for fname in tqdm(sorted(pt_files), desc="Processing activation files"):
        try:
            in_path = os.path.join(input_dir, fname)
            module_key = module_key_from_filename(fname, prefix)
            if not module_key:
                # print(f"Skipped {fname} because it doesn't match the prefix {prefix}", file=sys.stderr)
                skipped += 1
                continue
            call_count = call_counts.get(module_key, 1)
            prepare_pickle_shims()
            activations = torch.load(in_path, map_location="cpu", weights_only=False)
            if not isinstance(activations, list):
                raise TypeError(f"{fname}: expected list, got {type(activations)}")
            sliced = slice_activation_list(activations, step_idx, call_count, fname)
            out_path = os.path.join(output_dir, fname)
            torch.save(sliced, out_path)
        except Exception as e:
            print(f"Error processing {fname}: {e}", file=sys.stderr)
            skipped += 1
            continue

    if skipped:
        print(
            f"Skipped {skipped} files without _input.pt/_output.pt suffix "
            f"or without prefix match.",
            file=sys.stderr,
        )
        print(f"processed {len(pt_files) - skipped} files")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Postprocess activation .pt files to keep a single generation step."
    )
    parser.add_argument("--input-dir", required=True, help="Folder with activation .pt files.")
    parser.add_argument(
        "--call-sequence",
        required=True,
        help="Path to call_sequence_rank0.log from prefill.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Output root; a folder named after input-dir will be created inside.",
    )
    parser.add_argument(
        "--step-idx",
        type=int,
        required=True,
        help="Zero-based generation step index to keep.",
    )
    parser.add_argument(
        "--prefix",
        default="",
        help=(
            "Optional filename prefix to strip, e.g. "
            "'hf_activation_rank0_' or 'sglang_activation_rank0_'."
        ),
    )
    args = parser.parse_args()
    process_folder(
        args.input_dir,
        args.call_sequence,
        args.output_root,
        args.step_idx,
        args.prefix,
    )


if __name__ == "__main__":
    main()
