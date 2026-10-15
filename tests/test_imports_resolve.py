"""
Static check that every project-internal import in the maintained code resolves to a real module and a real name.

Nothing is executed (scripts here run experiments at import time); the files are parsed and each `import` / `from ... import`
is resolved the way Python would find it:
  - a dotted or plain module name from the repository root (every script puts the root on sys.path),
  - a plain module name next to the importing file (script-directory semantics, e.g. training/diffusion/*.py),
and for `from m import name` the name must exist at the top level of m (def / class / assignment / import) or be a submodule.
Anything that is not part of the repository (numpy, torch, ...) is ignored.

    python tests/test_imports_resolve.py            # prints every problem, also for legacy/
    python -m unittest tests.test_imports_resolve   # fails on problems in maintained code (everything except legacy/)
"""
import ast
import sys
import unittest
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKIP_PARTS = {".git", "__pycache__", ".venv", "node_modules"}
LEGACY = "legacy"


def py_files():
    return sorted(p for p in ROOT.rglob("*.py") if not (set(p.relative_to(ROOT).parts) & SKIP_PARTS))


@lru_cache(maxsize=None)
def project_top_level():
    names = set()
    for child in ROOT.iterdir():
        if child.name.startswith(".") or child.name in SKIP_PARTS:
            continue
        if child.is_file() and child.suffix == ".py":
            names.add(child.stem)
        elif child.is_dir() and any(child.rglob("*.py")):          # data folders (csv/, checkpoints/, ...) are not packages
            names.add(child.name)
    return names


def resolve_module(dotted, importer: Path):
    """Path of the file (or directory, for a namespace package) a module name refers to, or None."""
    parts = dotted.split(".")
    for base in (ROOT, importer.parent):
        target = base.joinpath(*parts)
        if target.with_suffix(".py").is_file():
            return target.with_suffix(".py")
        if (target / "__init__.py").is_file():
            return target / "__init__.py"
        if target.is_dir() and any(target.glob("*.py")):
            return target                                   # namespace package
    return None


@lru_cache(maxsize=None)
def top_level_names(path: Path):
    """Names defined at the top level of a module (None if it cannot be parsed)."""
    if path.is_dir():
        return None                                         # namespace package: any submodule name is fine
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    except SyntaxError:
        return None
    names = set()

    def collect(body):
        for n in body:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(n.name)
            elif isinstance(n, ast.Assign):
                for t in n.targets:
                    for x in ast.walk(t):
                        if isinstance(x, ast.Name):
                            names.add(x.id)
            elif isinstance(n, (ast.AnnAssign, ast.AugAssign)) and isinstance(n.target, ast.Name):
                names.add(n.target.id)
            elif isinstance(n, ast.Import):
                for a in n.names:
                    names.add((a.asname or a.name).split(".")[0])
            elif isinstance(n, ast.ImportFrom):
                for a in n.names:
                    names.add(a.asname or a.name)
            elif isinstance(n, (ast.If, ast.Try, ast.With, ast.For, ast.While)):
                for field in ("body", "orelse", "finalbody"):
                    collect(getattr(n, field, []))
                for h in getattr(n, "handlers", []):
                    collect(h.body)
    collect(tree.body)
    if "__getattr__" in names:
        return None                                         # lazy attributes (PEP 562): cannot be enumerated statically
    return names


def check_file(path: Path):
    problems = []
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    except SyntaxError as e:
        return [f"{path.relative_to(ROOT).as_posix()}:{e.lineno}: syntax error: {e.msg}"]
    rel = path.relative_to(ROOT).as_posix()
    local = project_top_level() | {p.stem for p in path.parent.glob("*.py")}
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                top = a.name.split(".")[0]
                if top in local and resolve_module(a.name, path) is None:
                    problems.append(f"{rel}:{n.lineno}: `import {a.name}` does not resolve")
        elif isinstance(n, ast.ImportFrom) and n.level == 0 and n.module:
            top = n.module.split(".")[0]
            if top not in local:
                continue
            target = resolve_module(n.module, path)
            if target is None:
                problems.append(f"{rel}:{n.lineno}: `from {n.module} import ...` does not resolve")
                continue
            defined = top_level_names(target)
            if defined is None:
                continue
            for a in n.names:
                if a.name == "*" or a.name in defined:
                    continue
                if resolve_module(f"{n.module}.{a.name}", path) is not None:
                    continue                                # a submodule
                problems.append(f"{rel}:{n.lineno}: `{a.name}` is not defined in {n.module}")
    return problems


def check_all(include_legacy=True):
    problems = []
    for p in py_files():
        if not include_legacy and p.relative_to(ROOT).parts[0] == LEGACY:
            continue
        problems.extend(check_file(p))
    return problems


class ImportsResolve(unittest.TestCase):
    def test_maintained_code(self):
        problems = check_all(include_legacy=False)
        self.assertEqual(problems, [], "\n" + "\n".join(problems))


if __name__ == "__main__":
    found = check_all(include_legacy=True)
    maintained = [p for p in found if not p.startswith(LEGACY + "/")]
    print(f"{len(py_files())} files checked: {len(maintained)} problems in maintained code, "
          f"{len(found) - len(maintained)} in legacy/")
    for p in found:
        print(("  " if p.startswith(LEGACY + "/") else "  ** ") + p)
    sys.exit(1 if maintained else 0)
