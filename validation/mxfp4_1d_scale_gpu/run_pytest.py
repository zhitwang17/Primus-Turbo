#!/usr/bin/env python3
"""Run repository pytest cases against this worktree and a prebuilt extension."""

from __future__ import annotations

import sys

import pytest

from run_validation import DEFAULT_EXTENSION, DEFAULT_REPO, _bootstrap


torch = _bootstrap(DEFAULT_REPO, DEFAULT_EXTENSION)
from primus_turbo.pytorch.core import low_precision

pytorch_package = sys.modules["primus_turbo.pytorch"]
pytorch_package.float8_e4m3 = low_precision.float8_e4m3
pytorch_package.float8_e5m2 = low_precision.float8_e5m2
pytorch_package.float4_e2m1fn_x2 = low_precision.float4_e2m1fn_x2

raise SystemExit(pytest.main(sys.argv[1:]))
