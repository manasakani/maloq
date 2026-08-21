# Copyright (c) 2024-2026 ETH Zurich and the authors of the MALOQ package.
"""torch.compile plumbing shared by the backbone and the head.

Both models take a PyTorch Geometric ``Batch`` and return dictionaries, which
looks like it ought to be a problem -- Dynamo has to trace ``Data.__getattr__``
and ``Data.__contains__`` to reach the tensors. Measured, it is not: on torch
2.13 the backbone traces to one graph with zero breaks (705 ops) and the head
likewise (140 ops), and ``fullgraph=True`` succeeds on both. So there is no
need to hoist the graph-object reads out of the traced region, and this module
does not ask the models to.

What does matter is *what* gets compiled. ``torch.compile(module)`` returns an
``OptimizedModule`` whose ``state_dict`` prefixes every key with
``_orig_mod.``, and ``SplitTrainer.save_training_state`` writes
``model.state_dict()`` straight to disk -- so compiling the modules would make
a compiled run's checkpoints stop loading into an eager one.
``torch.compile(bound_method)`` has no such effect.

That is the only reason the models carry a ``_forward_impl``:
``nn.Module.__call__`` looks up ``self.forward`` by name, so a compiled
``forward`` would simply never be called. ``forward`` is a one-line dispatch to
``_core_fn``, which is ``_forward_impl`` until ``enable_compile`` swaps in the
compiled version.

Nothing here runs unless a run asks for it: ``enable_compile`` is called only
from ``TrainingWorkflow.build_model``, when ``compile`` is set in the config.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch._dynamo  # noqa: F401  -- torch._dynamo.config is read below

#: torch.compile's own backend modes, plus maloq's spelling for "off".
_MODES = ("default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs")


@dataclass(frozen=True)
class CompileSpec:
    """A resolved request to compile, or ``None`` for "run eager"."""

    mode: str = "default"

    def wrap(self, fn: Callable) -> Callable:
        # dynamic and fullgraph are not settings. Molecules differ in size: a
        # 184k-batch nablaDFT run visits 54 distinct (node, edge) shapes, and
        # static shapes compile one graph each -- past the cache limit, at
        # which point Dynamo gives up and runs eager for the rest of training.
        # Dynamic holds that at two graphs and costs nothing in steady state
        # (3.60 vs 3.69 ms/step measured). fullgraph makes a graph break an
        # error instead of a silent performance cliff, which is the whole point
        # of splitting the core out; a run that cannot take that should set
        # compile: False rather than compile around the break.
        return torch.compile(fn, mode=self.mode, dynamic=True, fullgraph=True)


def resolve(compile_: bool | str | None) -> CompileSpec | None:
    """Turn the ``compile`` config value into a spec, or ``None`` for eager.

    Accepts ``False``/``None`` for off, ``True`` for on with defaults, or one of
    torch's mode strings to pick a backend mode directly.
    """

    if compile_ is None or compile_ is False:
        return None
    if compile_ is True:
        return CompileSpec()
    mode = str(compile_).strip().lower()
    if mode in ("false", "off", "none", ""):
        return None
    if mode in ("true", "on"):
        return CompileSpec()
    if mode not in _MODES:
        raise ValueError(
            f"compile must be a bool or one of {list(_MODES)}; got {compile_!r}"
        )
    return CompileSpec(mode=mode)


def raise_cache_limit(minimum: int = 32) -> None:
    """Give Dynamo room for the recompiles this model legitimately needs.

    Even marked dynamic, the backbone traces more than once: the head runs
    separately, ``include_edges`` and the flash path take different branches,
    and the first batch is specialised before dynamic shapes kick in. The
    default limit of 8 is low enough that a normal run can exhaust it and fall
    back to eager silently. Raising it is free -- it caps nothing that would
    otherwise be hit.
    """

    cfg = torch._dynamo.config
    limit = getattr(cfg, "cache_size_limit", None)
    if limit is not None and limit < minimum:
        cfg.cache_size_limit = minimum
    # Recompiles are per-frame; the backbone core is one frame with several
    # legitimate variants, so the per-frame limit needs the same headroom.
    accumulated = getattr(cfg, "accumulated_cache_size_limit", None)
    if accumulated is not None and accumulated < minimum * 2:
        cfg.accumulated_cache_size_limit = minimum * 2


#: Rebinding two names in a torch namespace is a global act, so do it once and
#: only when a run has actually asked to compile.
_NVTX_NEUTRALISED = False


def neutralise_nvtx_for_dynamo() -> None:
    """Hide ``torch.cuda.nvtx.*`` from Dynamo without changing eager behaviour.

    The message-passing path is annotated with ``torch.cuda.nvtx.range_push`` /
    ``range_pop``. Those are ``torch.*`` calls returning an int, which Dynamo
    cannot place in an FX graph (gb0208): measured, they cost 23 graph breaks in
    a trace of the backbone, and under the shipped ``fullgraph=True`` they do not
    merely split the graph -- they raise ``torch._dynamo.exc.Unsupported``.

    Replacing them with plain-Python wrappers moves the decision inside the
    traced frame. ``torch.compiler.is_compiling()`` folds to a constant during
    tracing, so Dynamo keeps only the ``return 0`` branch and no ``torch.*`` call
    survives into the graph; outside tracing the original C function is called
    exactly as before, so an nsys profile of an eager run is unchanged. A
    compiled region loses the markers, which a fused graph could not have
    honoured anyway.

    Done here rather than at the call sites so that the annotations themselves
    stay as they were written.
    """

    global _NVTX_NEUTRALISED
    if _NVTX_NEUTRALISED:
        return

    nvtx = torch.cuda.nvtx
    original_push, original_pop = nvtx.range_push, nvtx.range_pop

    def range_push(msg):
        if torch.compiler.is_compiling():
            return 0
        return original_push(msg)

    def range_pop():
        if torch.compiler.is_compiling():
            return 0
        return original_pop()

    nvtx.range_push, nvtx.range_pop = range_push, range_pop
    _NVTX_NEUTRALISED = True


def describe(spec: CompileSpec | None) -> str:
    if spec is None:
        return "compile: off"
    return f"compile: mode={spec.mode} dynamic fullgraph"


class CompiledCoreMixin:
    """Gives a model a compiled ``_core`` without touching its ``state_dict``.

    ``forward`` calls ``self._core_fn(...)``, which is ``self._core`` until
    ``enable_compile`` swaps in the compiled version. Subclasses only have to
    provide ``_core``.
    """

    #: Set by ``enable_compile``; ``None`` means run eager.
    _compiled_core: Callable | None = None
    _compile_spec: CompileSpec | None = None

    def enable_compile(
        self, compile_: bool | str | None = True
    ) -> CompileSpec | None:
        """Compile this model's tensor core. Returns the spec actually used."""

        spec = resolve(compile_)
        # Written straight into __dict__, bypassing nn.Module.__setattr__. A
        # compiled callable is not an nn.Module today, so __setattr__ would
        # store it here anyway -- but if that ever changed it would land in
        # _modules instead and start showing up in state_dict, which is the one
        # thing this design exists to prevent.
        self.__dict__["_compile_spec"] = spec
        if spec is None:
            self.__dict__["_compiled_core"] = None
            return None
        self._check_compilable()
        raise_cache_limit()
        neutralise_nvtx_for_dynamo()
        self.__dict__["_compiled_core"] = spec.wrap(self._forward_impl)
        return spec

    def _check_compilable(self) -> None:
        """Refuse configurations that cannot be traced. Override as needed."""

    @property
    def _core_fn(self) -> Callable[..., Any]:
        if self._compiled_core is None:
            return self._forward_impl
        return self._compiled_core

    def _forward_impl(self, *args, **kwargs):  # pragma: no cover - interface
        raise NotImplementedError


__all__ = [
    "CompileSpec",
    "CompiledCoreMixin",
    "describe",
    "neutralise_nvtx_for_dynamo",
    "raise_cache_limit",
    "resolve",
]
