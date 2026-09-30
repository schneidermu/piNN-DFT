"""Stream the Diet-clean grouped Minnesota pickle into a canonical view.

The source artifact is too large to load in the constrained WSL environment.
This reader makes two streaming passes over the protocol-4 pickle: a small
opcode pass records when each memo entry is last referenced, then a pure-Python
unpickler releases expired memo objects and keeps one augmentation variant per
top-level reaction group.  The original pickle is never rewritten.

Pickle is executable input.  Use this utility only with the trusted local
``data_train_grouped.pickle`` supplied for this project.
"""

from __future__ import annotations

import argparse
import array
import gc
import hashlib
import heapq
import json
import os
import pickle
import pickletools
import posixpath
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, BinaryIO

from prepare_data import DIET_HELD_OUT_MN_REACTIONS, TRAINING_PROTOCOL

EXPECTED_GROUPS = 268
EXPECTED_VARIANTS = 8
EXPECTED_CANONICAL_SUFFIX = "level2"
EXPECTED_FIELDS = frozenset(
    {
        "Database",
        "ReactionID",
        "Components",
        "Coefficients",
        "Energy",
        "component_paths",
        "Grid",
        "Weights",
        "Densities",
        "Gradients",
        "HF_energies",
        "backsplit_ind",
        "PBE_local_energies",
    }
)
class SourceFormatError(ValueError):
    """The input artifact does not match the audited grouped-pickle schema."""


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _memo_opcode_index(op_name: str, arg: Any) -> tuple[str, int] | None:
    if op_name in {"MEMOIZE", "BINPUT", "LONG_BINPUT", "PUT"}:
        if op_name == "MEMOIZE":
            return "assign_next", -1
        return "assign", int(arg)
    if op_name in {"BINGET", "LONG_BINGET", "GET"}:
        return "get", int(arg)
    return None


def scan_memo_last_uses(source: BinaryIO) -> tuple[array.array, int]:
    """Return per-slot final BINGET opcode number without retaining pickle data."""
    last_use = array.array("q")
    next_slot = 0
    opcode_count = 0
    first_opcode = True
    for opcode, arg, _position in pickletools.genops(source):
        if first_opcode:
            if opcode.name != "PROTO" or arg != 4:
                raise SourceFormatError("Expected a protocol-4 pickle input.")
            first_opcode = False
        event = _memo_opcode_index(opcode.name, arg)
        if event is not None:
            kind, slot = event
            if kind == "assign_next":
                slot = next_slot
                next_slot += 1
                last_use.append(-1)
            elif kind == "assign":
                if slot != next_slot:
                    raise SourceFormatError(
                        "Non-sequential memo assignment encountered; refusing "
                        "to prune memo values safely."
                    )
                next_slot += 1
                while len(last_use) < next_slot:
                    last_use.append(-1)
            else:
                if slot < 0 or slot >= next_slot:
                    raise SourceFormatError(
                        f"Pickle references memo slot {slot} before assignment."
                    )
                last_use[slot] = opcode_count
        opcode_count += 1
        if opcode.name == "STOP":
            if source.read(1):
                raise SourceFormatError(
                    "Unexpected bytes after the pickle STOP opcode."
                )
            break
    if first_opcode:
        raise SourceFormatError("Input is empty, not a pickle.")
    if not last_use:
        raise SourceFormatError("Input contains no pickle memo entries.")
    return last_use, opcode_count


def _path_suffix(path: Any) -> str:
    if not isinstance(path, (str, os.PathLike)):
        raise SourceFormatError(f"component path is not text: {type(path).__name__}")
    text = os.fspath(path)
    basename = re.split(r"[/\\]", text)[-1]
    stem = posixpath.splitext(basename)[0]
    if "__" in stem:
        return stem.split("__", 1)[1]
    return "default"


def _reaction_suffix(reaction: dict[str, Any]) -> str:
    paths = reaction.get("component_paths")
    if paths is None or len(paths) == 0:
        raise SourceFormatError("A variant has no component_paths.")
    suffixes = {_path_suffix(path) for path in paths}
    if len(suffixes) != 1:
        raise SourceFormatError(
            f"A variant has inconsistent component suffixes: {sorted(suffixes)}"
        )
    return next(iter(suffixes))


def _point_count(reaction: dict[str, Any]) -> int:
    grid = reaction["Grid"]
    shape = getattr(grid, "shape", None)
    if shape is None or len(shape) != 2 or int(shape[1]) != 9:
        raise SourceFormatError(f"Grid must have shape (N, 9); got {shape!r}.")
    return int(shape[0])


def _validate_variants(
    group_key: int, variants: list[Any]
) -> tuple[dict[str, Any], str, list[str], int]:
    if len(variants) != EXPECTED_VARIANTS:
        raise SourceFormatError(
            f"Group {group_key} has {len(variants)} variants; expected "
            f"{EXPECTED_VARIANTS}."
        )
    suffix_to_variant: dict[str, dict[str, Any]] = {}
    suffixes: list[str] = []
    identity: tuple[Any, Any] | None = None
    for variant in variants:
        if not isinstance(variant, dict):
            raise SourceFormatError(
                f"Group {group_key} contains a non-dictionary variant."
            )
        actual_fields = frozenset(variant)
        if actual_fields != EXPECTED_FIELDS:
            missing = sorted(EXPECTED_FIELDS - actual_fields)
            extra = sorted(actual_fields - EXPECTED_FIELDS)
            raise SourceFormatError(
                f"Group {group_key} field schema mismatch; missing={missing}, "
                f"extra={extra}."
            )
        variant_identity = (variant["Database"], variant["ReactionID"])
        if identity is None:
            identity = variant_identity
        elif variant_identity != identity:
            raise SourceFormatError(
                f"Group {group_key} variants disagree on reaction identity."
            )
        database, reaction_id = variant_identity
        if reaction_id in DIET_HELD_OUT_MN_REACTIONS.get(str(database), set()):
            raise SourceFormatError(
                f"Group {group_key} contains held-out Diet reaction "
                f"{database}:{reaction_id}."
            )
        suffix = _reaction_suffix(variant)
        if suffix in suffix_to_variant:
            raise SourceFormatError(
                f"Group {group_key} repeats augmentation suffix {suffix!r}."
            )
        suffixes.append(suffix)
        suffix_to_variant[suffix] = variant
    if len(suffix_to_variant) != EXPECTED_VARIANTS:
        raise SourceFormatError(
            f"Group {group_key} has {len(suffix_to_variant)} distinct suffixes."
        )
    selected_suffix = "default" if "default" in suffix_to_variant else min(suffixes)
    if selected_suffix != EXPECTED_CANONICAL_SUFFIX:
        raise SourceFormatError(
            f"Group {group_key} canonical suffix is {selected_suffix!r}; the "
            f"audited source expects {EXPECTED_CANONICAL_SUFFIX!r}."
        )
    selected = suffix_to_variant[selected_suffix]
    points = _point_count(selected)
    if identity is None:
        raise SourceFormatError(f"Group {group_key} is empty.")
    return selected, selected_suffix, sorted(suffixes), points


class _PruningUnpickler(pickle._Unpickler):
    """Pure-Python unpickler that discards expired memo refs and 7 variants."""

    dispatch = pickle._Unpickler.dispatch.copy()

    def __init__(
        self,
        file: BinaryIO,
        last_use: array.array,
        pilot_group: int | None = None,
    ):
        super().__init__(file)
        self._last_use = last_use
        self._pilot_group = pilot_group
        self._next_slot = 0
        self._opcode_index = -1
        self._expiry_heap: list[tuple[int, int]] = []
        self._active_group_lists: dict[int, int] = {}
        self._selected: dict[int, dict[str, Any]] = {}
        self._metadata: dict[int, dict[str, Any]] = {}
        self._root_dict: dict[Any, Any] | None = None
        self._pilot_variants: list[dict[str, Any]] | None = None

    def _store_memo_slot(self, slot: int, value: Any) -> None:
        if slot >= len(self._last_use):
            raise SourceFormatError("Memo scan and unpickler became inconsistent.")
        expiry = int(self._last_use[slot])
        if expiry < 0 or expiry <= self._opcode_index:
            self.memo.pop(slot, None)
        else:
            self.memo[slot] = value
            heapq.heappush(self._expiry_heap, (expiry, slot))

    def _expire_memo(self) -> None:
        while self._expiry_heap and self._expiry_heap[0][0] <= self._opcode_index:
            _expiry, slot = heapq.heappop(self._expiry_heap)
            self.memo.pop(slot, None)

    def load(self) -> Any:
        if not hasattr(self, "_file_read"):
            raise pickle.UnpicklingError("Unpickler was not initialized correctly")
        self._unframer = pickle._Unframer(self._file_read, self._file_readline)
        self.read = self._unframer.read
        self.readinto = self._unframer.readinto
        self.readline = self._unframer.readline
        self.metastack = []
        self.stack = []
        self.append = self.stack.append
        self.proto = 0
        read = self.read
        try:
            while True:
                key = read(1)
                if not key:
                    raise EOFError
                self._opcode_index += 1
                self.dispatch[key[0]](self)
                self._expire_memo()
        except pickle._Stop as stopinst:
            self._expire_memo()
            return stopinst.value

    def load_memoize(self) -> None:
        slot = self._next_slot
        self._next_slot += 1
        self._store_memo_slot(slot, self.stack[-1])

    def load_binput(self) -> None:
        slot = self.read(1)[0]
        self._store_explicit_memo(slot)

    def load_long_binput(self) -> None:
        slot = pickle.unpack("<I", self.read(4))[0]
        self._store_explicit_memo(slot)

    def load_put(self) -> None:
        slot = int(self.readline()[:-1])
        self._store_explicit_memo(slot)

    def _store_explicit_memo(self, slot: int) -> None:
        if slot != self._next_slot:
            raise SourceFormatError(
                "Non-sequential memo assignment encountered during unpickle."
            )
        self._next_slot += 1
        self._store_memo_slot(slot, self.stack[-1])

    def load_empty_list(self) -> None:
        value: list[Any] = []
        if self.stack and type(self.stack[-1]) is int:
            parent_is_root = (
                len(self.stack) >= 2 and self.stack[-2] is self._root_dict
            ) or (
                self.metastack
                and self.metastack[-1]
                and self.metastack[-1][-1] is self._root_dict
            )
            if parent_is_root:
                self._active_group_lists[id(value)] = self.stack[-1]
        self.append(value)

    def load_empty_dict(self) -> None:
        value: dict[Any, Any] = {}
        if self._root_dict is None and not self.stack and not self.metastack:
            self._root_dict = value
        self.append(value)

    def load_appends(self) -> None:
        items = self.pop_mark()
        list_obj = self.stack[-1]
        group_key = self._active_group_lists.get(id(list_obj))
        if group_key is None:
            list_obj.extend(items)
            return
        if group_key in self._selected:
            raise SourceFormatError(f"Group {group_key} list was appended twice.")
        selected, suffix, suffixes, points = _validate_variants(group_key, items)
        if group_key == self._pilot_group:
            self._pilot_variants = list(items)
        list_obj.append(selected)
        self._selected[group_key] = selected
        self._metadata[group_key] = {
            "selected_suffix": suffix,
            "available_suffixes": suffixes,
            "variant_count": len(items),
            "point_count": points,
            "database": str(selected["Database"]),
            "reaction_id": int(selected["ReactionID"]),
        }

    def load_append(self) -> None:
        value = self.stack.pop()
        list_obj = self.stack[-1]
        group_key = self._active_group_lists.get(id(list_obj))
        if group_key is None:
            list_obj.append(value)
            return
        if group_key in self._selected:
            raise SourceFormatError(f"Group {group_key} has extra list items.")
        list_obj.append(value)
        if len(list_obj) == EXPECTED_VARIANTS:
            selected, suffix, suffixes, points = _validate_variants(group_key, list_obj)
            if group_key == self._pilot_group:
                self._pilot_variants = list(list_obj)
            list_obj[:] = [selected]
            self._selected[group_key] = selected
            self._metadata[group_key] = {
                "selected_suffix": suffix,
                "available_suffixes": suffixes,
                "variant_count": EXPECTED_VARIANTS,
                "point_count": points,
                "database": str(selected["Database"]),
                "reaction_id": int(selected["ReactionID"]),
            }


_PruningUnpickler.dispatch[pickle.APPENDS[0]] = _PruningUnpickler.load_appends
_PruningUnpickler.dispatch[pickle.APPEND[0]] = _PruningUnpickler.load_append
_PruningUnpickler.dispatch[pickle.EMPTY_LIST[0]] = _PruningUnpickler.load_empty_list
_PruningUnpickler.dispatch[pickle.EMPTY_DICT[0]] = _PruningUnpickler.load_empty_dict
_PruningUnpickler.dispatch[pickle.MEMOIZE[0]] = _PruningUnpickler.load_memoize
_PruningUnpickler.dispatch[pickle.BINPUT[0]] = _PruningUnpickler.load_binput
_PruningUnpickler.dispatch[pickle.LONG_BINPUT[0]] = _PruningUnpickler.load_long_binput
_PruningUnpickler.dispatch[pickle.PUT[0]] = _PruningUnpickler.load_put


def _atomic_pickle_dump(value: Any, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as output:
            pickle.dump(value, output, protocol=4)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return sha256_file(target)


def _validate_root(
    data: Any, metadata: dict[int, dict[str, Any]]
) -> dict[int, list[dict[str, Any]]]:
    if not isinstance(data, dict):
        raise SourceFormatError("Pickle root must be a dictionary.")
    if len(data) != EXPECTED_GROUPS:
        raise SourceFormatError(
            f"Expected {EXPECTED_GROUPS} base reactions, got {len(data)}."
        )
    if any(type(key) is not int for key in data):
        raise SourceFormatError("Top-level reaction keys must all be integers.")
    if set(data) != set(metadata):
        raise SourceFormatError("Some top-level groups were not canonicalized.")
    identities: set[tuple[str, int]] = set()
    for key, group in data.items():
        if (
            not isinstance(group, list)
            or len(group) != 1
            or not isinstance(group[0], dict)
        ):
            raise SourceFormatError(f"Group {key} did not reduce to one reaction.")
        identity = (str(group[0]["Database"]), int(group[0]["ReactionID"]))
        if identity in identities:
            raise SourceFormatError(f"Duplicate base reaction identity {identity}.")
        identities.add(identity)
    return data


def _verify_written_view(path: Path, expected_groups: int) -> None:
    """Verify ordinary pickle.load can read the compact output schema."""
    with path.open("rb") as source:
        data = pickle.load(source)
    if not isinstance(data, dict) or len(data) != expected_groups:
        raise SourceFormatError(
            f"Written view {path} did not reload as {expected_groups} groups."
        )
    if list(data) != list(range(expected_groups)):
        raise SourceFormatError("Written canonical view keys are not contiguous.")
    for key, group in data.items():
        if not isinstance(group, list) or len(group) != 1:
            raise SourceFormatError(
                f"Written canonical group {key} does not contain one variant."
            )
        if frozenset(group[0]) != EXPECTED_FIELDS:
            raise SourceFormatError(
                f"Written canonical group {key} has an unexpected field schema."
            )
    del data
    gc.collect()


def _verify_written_pilot(path: Path, original_key: int) -> None:
    with path.open("rb") as source:
        pilot = pickle.load(source)
    if not isinstance(pilot, dict) or list(pilot) != [original_key]:
        raise SourceFormatError("Written pilot view is not a one-key grouped dataset.")
    variants = pilot[original_key]
    if not isinstance(variants, list) or len(variants) != EXPECTED_VARIANTS:
        raise SourceFormatError(
            "Written pilot view does not contain all eight variants."
        )
    _validate_variants(original_key, variants)
    del pilot
    gc.collect()


def build_canonical_view(
    source_path: Path,
    output_path: Path,
    manifest_path: Path,
    pilot_group: int | None = None,
    pilot_output_path: Path | None = None,
) -> dict[str, Any]:
    source_path = source_path.resolve()
    output_path = output_path.resolve()
    manifest_path = manifest_path.resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    source_stat = source_path.stat()
    source_size = source_stat.st_size
    source_hash = sha256_file(source_path)
    with source_path.open("rb") as source:
        memo_last_use, _opcode_count = scan_memo_last_uses(source)
    with source_path.open("rb") as source:
        unpickler = _PruningUnpickler(source, memo_last_use, pilot_group=pilot_group)
        data = _validate_root(unpickler.load(), unpickler._metadata)
    final_stat = source_path.stat()
    if (
        final_stat.st_size != source_size
        or final_stat.st_mtime_ns != source_stat.st_mtime_ns
    ):
        raise SourceFormatError("Source artifact changed while it was being read.")

    original_keys = sorted(data)
    canonical_data = {
        new_key: data[old_key] for new_key, old_key in enumerate(original_keys)
    }

    if pilot_group is not None:
        if pilot_group not in data:
            raise SourceFormatError(
                f"Pilot group {pilot_group} is not a key in the grouped dataset."
            )
        if pilot_output_path is None:
            pilot_output_path = output_path.with_name(
                f"{output_path.stem}_group_{pilot_group}{output_path.suffix}"
            )
        pilot_variants = unpickler._pilot_variants
        if pilot_variants is None or len(pilot_variants) != EXPECTED_VARIANTS:
            raise SourceFormatError(
                f"Pilot group {pilot_group} was not retained with all variants."
            )
        pilot_view = {pilot_group: pilot_variants}
        pilot_hash = _atomic_pickle_dump(pilot_view, pilot_output_path.resolve())
        pilot_info = {
            "group_key": pilot_group,
            "selected_suffix": unpickler._metadata[pilot_group]["selected_suffix"],
            "selected_reaction_count": 1,
            "variant_count": len(pilot_variants),
            "point_count": unpickler._metadata[pilot_group]["point_count"],
            "output_path": str(pilot_output_path.resolve()),
            "output_sha256": pilot_hash,
        }
    else:
        pilot_info = None

    total_points = sum(row["point_count"] for row in unpickler._metadata.values())
    group_count = len(canonical_data)
    group_rows = [
        {
            "group_key": key,
            "output_index": output_index,
            **unpickler._metadata[key],
        }
        for output_index, key in enumerate(original_keys)
    ]
    output_hash = _atomic_pickle_dump(canonical_data, output_path)
    del canonical_data, data
    del unpickler
    if pilot_group is not None:
        del pilot_view, pilot_variants
    gc.collect()
    _verify_written_view(output_path, EXPECTED_GROUPS)
    if pilot_group is not None:
        _verify_written_pilot(pilot_output_path.resolve(), pilot_group)
    manifest = {
        "schema": "canonical-minnesota-view-v1",
        "training_protocol": TRAINING_PROTOCOL,
        "selection_rule": "default if present; otherwise lexicographically minimum component-path suffix",
        "source": {
            "path": str(source_path),
            "size_bytes": source_size,
            "sha256": source_hash,
        },
        "output": {
            "path": str(output_path),
            "sha256": output_hash,
            "group_count": group_count,
            "selected_reaction_count": group_count,
            "total_grid_points": total_points,
            "variant_count_per_group": EXPECTED_VARIANTS,
            "grid_columns": 9,
            "key_policy": "contiguous insertion-order indices 0..267",
        },
        "source_sha256": source_hash,
        "output_sha256": output_hash,
        "base_reaction_count": group_count,
        "pilot_view": pilot_info,
        "groups": group_rows,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="trusted grouped MN pickle")
    parser.add_argument("output", type=Path, help="canonical 268-reaction pickle")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pilot-group", type=int)
    parser.add_argument("--pilot-output", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.pilot_output is not None and args.pilot_group is None:
        raise SystemExit("--pilot-output requires --pilot-group")
    manifest = build_canonical_view(
        args.source,
        args.output,
        args.manifest,
        pilot_group=args.pilot_group,
        pilot_output_path=args.pilot_output,
    )
    print(
        json.dumps(
            {
                "source_sha256": manifest["source"]["sha256"],
                "output_sha256": manifest["output"]["sha256"],
                "groups": manifest["output"]["group_count"],
                "total_grid_points": manifest["output"]["total_grid_points"],
                "manifest": str(args.manifest.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
