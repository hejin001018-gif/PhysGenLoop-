"""Deprecated compatibility namespace for :mod:`physgenloop.learning_repair`.

No implementation lives here at runtime.  Existing imports are redirected to
the canonical package so old scripts and frozen experiment manifests continue
to work during the migration window.
"""

from __future__ import annotations

from importlib import import_module
import sys
import warnings

from physgenloop.learning_repair import *  # noqa: F401,F403
from physgenloop.learning_repair import __all__


_MODULES = (
    "action_value_cli",
    "baselines",
    "campaign",
    "cli",
    "cloud_campaign",
    "compatibility",
    "contracts",
    "evaluation",
    "executors",
    "manifests",
    "memory_policy",
    "proxy_adapter",
    "recording",
    "review",
    "runner",
    "selector",
    "value_policy",
    "value_training",
)

for _name in _MODULES:
    sys.modules[f"{__name__}.{_name}"] = import_module(
        f"physgenloop.learning_repair.{_name}"
    )

warnings.warn(
    "physgenloop.learning_repair_pipeline is deprecated; use "
    "physgenloop.learning_repair",
    DeprecationWarning,
    stacklevel=2,
)
