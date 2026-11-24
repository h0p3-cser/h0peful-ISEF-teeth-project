"""Assemble per-sample training rows from identity and per-QLF lookup tables."""

from __future__ import annotations

from collections.abc import Mapping

_QLF = "qlf_path"
_SAMPLE = "sample_id"


def _is_nonempty(value: object) -> bool:
    return isinstance(value, str) and value.strip() != ""


def _preview(values: list[object], limit: int = 3) -> str:
    text = ", ".join(repr(value) for value in values[:limit])
    if len(values) > limit:
        text += f", ... ({len(values)} total)"
    return text


def _identity(rows: list[dict]) -> list[dict]:
    seen = {_QLF: set(), _SAMPLE: set()}
    duplicates = {_QLF: [], _SAMPLE: []}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"identity_rows[{index}] is not a dictionary")
        missing = [name for name in (_QLF, _SAMPLE) if name not in row]
        if missing:
            raise ValueError(
                f"identity_rows[{index}] is missing {len(missing)} required "
                f"column(s) (e.g. {_preview(missing)})"
            )
        for name in (_QLF, _SAMPLE):
            value = row[name]
            if not _is_nonempty(value):
                raise ValueError(
                    f"identity_rows[{index}] has an empty or non-string "
                    f"{name}: {value!r}"
                )
            if value in seen[name] and value not in duplicates[name]:
                duplicates[name].append(value)
            seen[name].add(value)
    problems = [
        f"{name}: {len(duplicates[name])} duplicate value(s) "
        f"(e.g. {_preview(duplicates[name])})"
        for name in (_QLF, _SAMPLE)
        if duplicates[name]
    ]
    if problems:
        raise ValueError("identity_rows must be unique: " + "; ".join(problems))
    return rows


def _mapping(name: str, value: object) -> Mapping:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping keyed by qlf_path")
    return value


def _check_keys(name: str, table: Mapping, expected: list[str]) -> None:
    expected_set = set(expected)
    keys = list(table.keys())
    key_set = set(keys)
    missing = [path for path in expected if path not in key_set]
    extra = [path for path in keys if path not in expected_set]
    if missing or extra:
        problems = []
        if missing:
            problems.append(f"{len(missing)} missing (e.g. {_preview(missing)})")
        if extra:
            problems.append(f"{len(extra)} extra (e.g. {_preview(extra)})")
        raise ValueError(
            f"{name} qlf_path keys do not match identity_rows: "
            + ", ".join(problems)
        )


def _check_columns(left_name: str, left: Mapping, right_name: str, right: Mapping) -> None:
    overlap = [column for column in left if column in right]
    if overlap:
        raise ValueError(
            f"{left_name} and {right_name} have {len(overlap)} colliding "
            f"column(s) (e.g. {_preview(overlap)})"
        )


def assemble(
    identity_rows: list[dict],
    single_by_path: Mapping,
    longitudinal_by_path: Mapping,
) -> list[dict]:
    """Merge each identity row with its single and longitudinal lookup values.

    Rows are returned in identity order. Every nonempty ``qlf_path`` and
    ``sample_id`` must be unique, both mappings must have exactly the identity
    qlf_path set, and no column name may occur in more than one of the three
    dictionaries. Values are copied as-is; blanks are neither filled nor coerced.
    """
    rows = _identity(list(identity_rows))
    single = _mapping("single_by_path", single_by_path)
    longitudinal = _mapping("longitudinal_by_path", longitudinal_by_path)
    expected = [row[_QLF] for row in rows]
    _check_keys("single_by_path", single, expected)
    _check_keys("longitudinal_by_path", longitudinal, expected)

    output = []
    for index, row in enumerate(rows):
        path = row[_QLF]
        single_value = single[path]
        longitudinal_value = longitudinal[path]
        if not isinstance(single_value, Mapping):
            raise TypeError(f"single_by_path[{path!r}] is not a dictionary")
        if not isinstance(longitudinal_value, Mapping):
            raise TypeError(f"longitudinal_by_path[{path!r}] is not a dictionary")
        _check_columns(f"identity_rows[{index}]", row, "single_by_path", single_value)
        _check_columns(
            f"identity_rows[{index}]", row, "longitudinal_by_path", longitudinal_value
        )
        _check_columns(
            f"single_by_path[{path!r}]",
            single_value,
            f"longitudinal_by_path[{path!r}]",
            longitudinal_value,
        )
        merged = dict(row)
        merged.update(single_value)
        merged.update(longitudinal_value)
        output.append(merged)
    return output
