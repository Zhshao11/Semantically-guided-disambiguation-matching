"""Dataset decoding and machine-readable report helpers."""

import csv
import json
import pickle
import struct
from pathlib import Path

import numpy as np


class NumpyUnpickler(pickle.Unpickler):
    """Read the legacy WHU object arrays with a small NumPy-only allowlist."""

    def find_class(self, module, name):
        allowed = {
            ("numpy", "ndarray"): np.ndarray,
            ("numpy", "dtype"): np.dtype,
            ("numpy.core.multiarray", "_reconstruct"): np.core.multiarray._reconstruct,
            ("numpy._core.multiarray", "_reconstruct"): np.core.multiarray._reconstruct,
            ("numpy.core.multiarray", "scalar"): np.core.multiarray.scalar,
            ("numpy._core.multiarray", "scalar"): np.core.multiarray.scalar,
        }
        if (module, name) not in allowed:
            raise pickle.UnpicklingError(f"Unsupported metadata type {module}.{name}")
        return allowed[module, name]


def load_datum(path):
    with open(path, "rb") as stream:
        version = np.lib.format.read_magic(stream)
        shape, _, dtype = np.lib.format._read_array_header(stream, version)
        if shape != (6,) or not dtype.hasobject:
            raise ValueError(f"Unexpected WHU datum header {shape}, {dtype}")
        data = NumpyUnpickler(stream).load()
    if not isinstance(data, np.ndarray) or data.shape != (6,):
        raise ValueError("Expected six-element WHU datum")
    return data


def read_flo(path):
    with open(path, "rb") as stream:
        magic, width, height = struct.unpack("<fii", stream.read(12))
        if magic != 202021.25 or not (0 < width <= 100000 and 0 < height <= 100000):
            raise ValueError("Invalid Middlebury flow header")
        if Path(path).stat().st_size != 12 + height * width * 2 * 4:
            raise ValueError("Flow payload size does not match header")
        return np.fromfile(stream, dtype="<f4").reshape(height, width, 2)


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def save_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def save_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or sorted({key for row in rows for key in row})
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
