#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Traced sizes for tensors whose shape depends on their values.

``torch.nonzero``, boolean masks, ``torch.unique`` and similar operations
return tensors whose size is only known at run time. Reading such a size as a
plain Python ``int`` would bake the traced value into the exported graph, so a
later input with a different count fails (for example in ``Reshape``).
:class:`TracedInt` is an ``int`` that also carries the FX proxy of the size
query, so shapes built from it stay symbolic in the exported graph.
"""

import operator

import torch


class TracedInt(int):
    """An ``int`` recorded in the FX graph as the result of a size query.

    It behaves like the integer value it holds during tracing; arithmetic with
    other integers records the operation so derived sizes stay symbolic.
    """

    def __new__(cls, value: int, proxy, context, owner):
        instance = super().__new__(cls, value)
        instance.proxy = proxy
        instance._context = context
        # Traced tensor the size was read from; tensors built from this size
        # are created through it so they join the same graph.
        instance._owner = owner
        return instance

    def _record(self, op, lhs, rhs) -> "TracedInt":
        value = op(int(lhs), int(rhs))
        if not isinstance(value, int):
            return value
        proxy = self._context.tracer.create_proxy(
            "call_function", op, (proxy_of(lhs), proxy_of(rhs)), {}
        )
        return TracedInt(value, proxy, self._context, self._owner)

    def __add__(self, other):
        return self._record(operator.add, self, other) if isinstance(other, int) else NotImplemented

    def __radd__(self, other):
        return self._record(operator.add, other, self) if isinstance(other, int) else NotImplemented

    def __sub__(self, other):
        return self._record(operator.sub, self, other) if isinstance(other, int) else NotImplemented

    def __rsub__(self, other):
        return self._record(operator.sub, other, self) if isinstance(other, int) else NotImplemented

    def __mul__(self, other):
        return self._record(operator.mul, self, other) if isinstance(other, int) else NotImplemented

    def __rmul__(self, other):
        return self._record(operator.mul, other, self) if isinstance(other, int) else NotImplemented

    def __floordiv__(self, other):
        return self._record(operator.floordiv, self, other) if isinstance(other, int) else NotImplemented

    def __rfloordiv__(self, other):
        return self._record(operator.floordiv, other, self) if isinstance(other, int) else NotImplemented

    def __mod__(self, other):
        return self._record(operator.mod, self, other) if isinstance(other, int) else NotImplemented

    def __neg__(self):
        return self._record(operator.sub, 0, self)

    def __repr__(self) -> str:
        return f"TracedInt({int(self)})"


def proxy_of(value):
    """FX argument for ``value``: the proxy of a :class:`TracedInt`, else ``value``."""
    return value.proxy if isinstance(value, TracedInt) else value


def find_traced_int(obj):
    """Return the first :class:`TracedInt` in a nested list/tuple, else None."""
    if isinstance(obj, TracedInt):
        return obj
    if isinstance(obj, (list, tuple)):
        for item in obj:
            found = find_traced_int(item)
            if found is not None:
                return found
    return None


def record_tensor_from_sizes(tracer, data, dtype, device):
    """Record ``torch.tensor(data)`` for nested sizes so it stays dynamic.

    Exporters that trace the FX graph (TorchScript) keep ``torch.tensor`` of
    traced sizes as a constant, but they do trace ``scalar_tensor`` and
    ``stack``; the tensor is rebuilt from those.
    """
    if isinstance(data, (list, tuple)):
        items = [record_tensor_from_sizes(tracer, item, dtype, device) for item in data]
        return tracer.create_proxy("call_function", torch.stack, (items,), {})
    return tracer.create_proxy(
        "call_function",
        torch.scalar_tensor,
        (proxy_of(data),),
        {"dtype": dtype, "device": device},
    )


def plain_sizes(obj):
    """Return ``obj`` with any ``torch.Size`` turned into a tuple.

    FX records sizes as proxies, which ``torch.Size`` cannot hold, so sizes are
    rebuilt as tuples before their elements are replaced by proxies.
    """
    if isinstance(obj, torch.Size):
        return tuple(obj)
    if isinstance(obj, (list, tuple)) and not hasattr(obj, "_fields"):
        return type(obj)(plain_sizes(item) for item in obj)
    if isinstance(obj, dict):
        return {key: plain_sizes(value) for key, value in obj.items()}
    return obj
