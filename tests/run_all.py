"""Minimal stdlib test runner so the suite needs zero dependencies.

    python -m tests.run_all        (from the repo root, or python tests/run_all.py)

The test functions are also plain pytest-compatible (tmp_path included), so
`pytest tests/` works identically once pytest is installed.
"""
import importlib
import inspect
import sys
import tempfile
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

MODULES = ("test_env", "test_nets", "test_agents")


def main():
    failures = []
    total = 0
    for mod_name in MODULES:
        mod = importlib.import_module(f"tests.{mod_name}")
        for name in sorted(dir(mod)):
            if not name.startswith("test_"):
                continue
            fn = getattr(mod, name)
            if not callable(fn):
                continue
            total += 1
            kwargs = {}
            if "tmp_path" in inspect.signature(fn).parameters:
                kwargs["tmp_path"] = Path(tempfile.mkdtemp(prefix="serslice_test_"))
            try:
                fn(**kwargs)
                print(f"PASS {mod_name}.{name}")
            except Exception:
                failures.append((mod_name, name, traceback.format_exc()))
                print(f"FAIL {mod_name}.{name}")
    for mod_name, name, tb in failures:
        print(f"\n--- {mod_name}.{name} ---\n{tb}")
    print(f"\n{total - len(failures)}/{total} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
