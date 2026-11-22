"""
scripts/export_weights_for_cpp.py
Estrae i pesi e i parametri di scaling (median, iqr) da `models/modello_calcio_v1.pt`
in formati pronti per C++ puro (`.txt` lineare leggibile con `std::ifstream` e `.json`).
Funziona sia con PyTorch installato che in pura standard library Python (`zipfile` + `pickle` + `struct`).
"""

import io
import json
import os
import pickle
import struct
import zipfile
from collections import OrderedDict
from typing import Any, Dict, List

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PT_PATH = os.path.join(BASE_DIR, "models", "modello_calcio_v1.pt")
OUT_JSON = os.path.join(BASE_DIR, "models", "modello_calcio_v1_weights.json")
OUT_TXT = os.path.join(BASE_DIR, "models", "modello_calcio_v1_weights.txt")


class _RawStorage:
    def __init__(self, key: str, size: int, raw_bytes: bytes):
        self.key = key
        self.size = size
        self.data = raw_bytes


def load_checkpoint_arrays(pt_path: str) -> Dict[str, Any]:
    """Estrae shape e valori float32 dei tensori del checkpoint senza richiedere LibTorch/PyTorch."""
    try:
        import torch  # type: ignore

        ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
        tensors = {
            "median": {"shape": list(ckpt["median"].shape), "data": ckpt["median"].detach().cpu().float().tolist()},
            "iqr": {"shape": list(ckpt["iqr"].shape), "data": ckpt["iqr"].detach().cpu().float().tolist()},
        }
        for name, t in ckpt["model_state"].items():
            tensors[name] = {"shape": list(t.shape), "data": t.detach().cpu().contiguous().float().tolist()}
        return {"n_features": int(ckpt.get("n_features", 56)), "tensors": tensors}
    except ImportError:
        pass

    with zipfile.ZipFile(pt_path, "r") as z:
        prefix = [n for n in z.namelist() if n.endswith("data.pkl")][0].rsplit("data.pkl", 1)[0]
        pkl_bytes = z.read(prefix + "data.pkl")

        class _Unpickler(pickle.Unpickler):
            def persistent_load(self, pid):
                _, _, key, _, size = pid
                return _RawStorage(key, size, z.read(f"{prefix}data/{key}"))

            def find_class(self, module, name):
                if module == "torch._utils" and name == "_rebuild_tensor_v2":
                    def _rebuild(storage, storage_offset, size, stride, requires_grad, backward_hooks, metadata=None):
                        vals = list(struct.unpack("<" + ("f" * storage.size), storage.data[: storage.size * 4]))
                        total_elems = 1
                        for s in size:
                            total_elems *= s
                        return {"shape": list(size), "data": vals[storage_offset : storage_offset + total_elems]}
                    return _rebuild
                if module == "collections" and name == "OrderedDict":
                    return OrderedDict
                if module == "torch" and name in ("FloatStorage", "HalfStorage", "DoubleStorage"):
                    return name
                raise pickle.Unpickler.find_class(self, module, name)

        ckpt = _Unpickler(io.BytesIO(pkl_bytes)).load()

    tensors = {
        "median": ckpt["median"],
        "iqr": ckpt["iqr"],
    }
    for name, t in ckpt["model_state"].items():
        tensors[name] = t
    return {"n_features": int(ckpt.get("n_features", 56)), "tensors": tensors}


def main():
    extracted = load_checkpoint_arrays(PT_PATH)
    tensors = extracted["tensors"]

    # 1. JSON strutturato
    json_payload = {
        "n_features": extracted["n_features"],
        "architecture": [56, 25, 16, 3],
        "activation": "tanh",
        "tensors": {k: {"shape": v["shape"], "data": v["data"]} for k, v in tensors.items()},
    }
    with open(OUT_JSON, "w", encoding="utf-8") as fj:
        json.dump(json_payload, fj, indent=2)

    # 2. TXT flat C++-friendly (tag rows cols seguito dai float separati da spazio)
    order = ["median", "iqr", "fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias", "fc3.weight", "fc3.bias"]
    with open(OUT_TXT, "w", encoding="utf-8") as ft:
        for name in order:
            shape = tensors[name]["shape"]
            data: List[float] = tensors[name]["data"]
            rows = shape[0]
            cols = shape[1] if len(shape) > 1 else 1
            ft.write(f"{name} {rows} {cols}\n")
            ft.write(" ".join(f"{x:.9g}" for x in data) + "\n")

    print(f"[OK] Esportato JSON: {OUT_JSON}")
    print(f"[OK] Esportato TXT:  {OUT_TXT}")
    for name in order:
        print(f"  - {name:<12} shape={tensors[name]['shape']} elems={len(tensors[name]['data'])}")


if __name__ == "__main__":
    main()
