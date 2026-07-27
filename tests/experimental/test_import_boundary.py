from __future__ import annotations

import ast
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src" / "maloq"
EXPERIMENTAL_ROOT = SOURCE_ROOT / "experimental"
VALID_SLUG = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")
AMBIGUOUS_SLUG_PARTS = re.compile(r"(?:^|_)(?:new|test|v[0-9]+)(?:_|$)")


def _canonical_python_files() -> list[Path]:
    return sorted(
        path
        for path in SOURCE_ROOT.rglob("*.py")
        if EXPERIMENTAL_ROOT not in path.parents
    )


def _import_target_is_experimental(target: str) -> bool:
    return (
        target == "maloq.experimental"
        or target.startswith("maloq.experimental.")
        or target == "experimental"
        or target.startswith("experimental.")
        or target.startswith(".experimental")
    )


def _violations(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    violations: list[tuple[int, str]] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _import_target_is_experimental(alias.name):
                    violations.append((node.lineno, alias.name))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if _import_target_is_experimental(module):
                violations.append((node.lineno, module))
            elif module == "maloq" and any(
                alias.name == "experimental" for alias in node.names
            ):
                violations.append((node.lineno, "maloq.experimental"))
            elif node.level and any(
                alias.name == "experimental" for alias in node.names
            ):
                violations.append((node.lineno, "relative experimental"))
        elif isinstance(node, ast.Call) and node.args:
            first_arg = node.args[0]
            if not isinstance(first_arg, ast.Constant):
                continue
            if not isinstance(first_arg.value, str):
                continue
            function_name = ""
            if isinstance(node.func, ast.Name):
                function_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                function_name = node.func.attr
            if function_name in {"import_module", "__import__"} and (
                _import_target_is_experimental(first_arg.value)
            ):
                violations.append((node.lineno, first_arg.value))

    return violations


def test_canonical_maloq_does_not_import_experimental() -> None:
    violations = [
        f"{path.relative_to(PROJECT_ROOT)}:{line}: {target}"
        for path in _canonical_python_files()
        for line, target in _violations(path)
    ]
    assert not violations, (
        "Canonical MALOQ must not import experimental features. "
        "Import the feature explicitly from its dated experiment runner:\n"
        + "\n".join(violations)
    )


def test_experimental_feature_directories_follow_contract() -> None:
    if not EXPERIMENTAL_ROOT.exists():
        raise AssertionError("src/maloq/experimental must exist")

    errors: list[str] = []
    for path in sorted(EXPERIMENTAL_ROOT.iterdir()):
        if not path.is_dir() or path.name.startswith((".", "_")):
            continue
        if not VALID_SLUG.fullmatch(path.name):
            errors.append(f"{path.name}: use lowercase snake_case")
        if AMBIGUOUS_SLUG_PARTS.search(path.name):
            errors.append(f"{path.name}: use a semantic slug, not new/test/vN")
        for required in ("__init__.py", "FEATURE.md"):
            if not (path / required).is_file():
                errors.append(f"{path.name}: missing {required}")

    assert not errors, "\n".join(errors)
