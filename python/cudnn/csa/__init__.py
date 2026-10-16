# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``cudnn.csa`` — experimental CSA/HCA kernels and training composition.

Symbols resolve lazily on first attribute access, so importing ``cudnn`` never
pulls in the optional PyTorch or CuTe DSL dependency stack.
"""

from importlib import import_module

_SYMBOLS = {
    "CompressedSparseAttention": (".training", "CompressedSparseAttention"),
    "CSACompressorForward": (".compressor", "CSACompressorForward"),
    "CSACompressorBackward": (".compressor", "CSACompressorBackward"),
    "csa_compressor_forward_wrapper": (".compressor", "csa_compressor_forward_wrapper"),
    "csa_compressor_backward_wrapper": (".compressor", "csa_compressor_backward_wrapper"),
}


def _load_symbol(name):
    """Import the symbol behind lazy attribute ``name`` and cache it in module globals."""
    module_name, symbol_name = _SYMBOLS[name]
    module = import_module(module_name, package=__name__)
    symbol = getattr(module, symbol_name)
    globals()[name] = symbol
    return symbol


def __getattr__(name):
    """Resolve the lazily exported symbols and the ``CSA`` namespace (PEP 562)."""
    if name == "CSA":
        return CSA
    if name in _SYMBOLS:
        return _load_symbol(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class CSANamespace:
    """Namespace object mirroring the package's lazy symbols (``cudnn.CSA.<symbol>``)."""

    def __getattr__(self, name):
        """Lazily resolve ``CSA.<name>`` through the package's symbol table."""
        if name in _SYMBOLS:
            return _load_symbol(name)
        raise AttributeError(f"CSA has no attribute {name!r}")


CSA = CSANamespace()

__all__ = [
    "CSA",
    "CompressedSparseAttention",
    "CSACompressorBackward",
    "CSACompressorForward",
    "csa_compressor_backward_wrapper",
    "csa_compressor_forward_wrapper",
]
