# Copyright (c) 2024-2026 ETH Zurich and the authors of the MALOQ package.
"""Top-level MALOQ package.

The public API is organized as immediate subpackages:
``maloq.dataset_utils``, ``maloq.fock_utils``, ``maloq.helm``, and
``maloq.train_utils``.
"""

from importlib import import_module

__all__ = ["dataset_utils", "fock_utils", "helm", "train_utils"]


def __getattr__(name):
	if name in __all__:
		module = import_module(f".{name}", __name__)
		globals()[name] = module
		return module
	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
	return sorted(list(globals().keys()) + __all__)
