# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Make an installed `lerobot` use THIS folder's modified SmolVLA modules.

Call `use_local_smolvla()` BEFORE importing anything from `lerobot`. It:

1. Replaces the `lerobot.policies` package with a lazy stub so its eager `__init__.py`
   (which imports every policy family and can crash on unrelated policies such as GROOT
   with newer transformers) never runs. The stub still resolves the same public names
   (`from lerobot.policies import SmolVLAConfig, ACTConfig, ...`) on demand, so lerobot
   internals like `async_inference.helpers` keep working.
2. Pre-seeds `sys.modules` with the four SmolVLA modules from this directory so all
   `lerobot.policies.smolvla.*` imports resolve to the modified code, and wires the
   parent-package attribute chain (`lerobot.policies.smolvla.modeling_smolvla` works
   with any import syntax).

The installed lerobot package on disk is never touched.
"""

import importlib
import importlib.util
import sys
import types
from pathlib import Path

_HERE = Path(__file__).resolve().parent

# Dependency order: configuration first (everything imports it), modeling last.
_MODULES = [
    "configuration_smolvla",
    "smolvlm_with_expert",
    "processor_smolvla",
    "modeling_smolvla",
]

# Public names of lerobot.policies.__init__ -> the submodule that defines them.
# Resolved lazily so a broken policy family (e.g. GROOT under transformers 5.x) only
# fails if something actually asks for it. SmolVLA names resolve to the injected modules.
_LAZY_EXPORTS = {
    "ACTConfig": "lerobot.policies.act.configuration_act",
    "DiffusionConfig": "lerobot.policies.diffusion.configuration_diffusion",
    "GrootConfig": "lerobot.policies.groot.configuration_groot",
    "PI0Config": "lerobot.policies.pi0.configuration_pi0",
    "PI0FastConfig": "lerobot.policies.pi0_fast.configuration_pi0_fast",
    "PI05Config": "lerobot.policies.pi05.configuration_pi05",
    "SmolVLAConfig": "lerobot.policies.smolvla.configuration_smolvla",
    "SmolVLANewLineProcessor": "lerobot.policies.smolvla.processor_smolvla",
    "TDMPCConfig": "lerobot.policies.tdmpc.configuration_tdmpc",
    "VQBeTConfig": "lerobot.policies.vqbet.configuration_vqbet",
    "WallXConfig": "lerobot.policies.wall_x.configuration_wall_x",
    "XVLAConfig": "lerobot.policies.xvla.configuration_xvla",
}


def use_local_smolvla() -> None:
    # Idempotent: if our stub is already installed (e.g. two modules in one process both
    # call this), the injection is done - just return.
    existing = sys.modules.get("lerobot.policies")
    if existing is not None and getattr(existing, "_local_smolvla_stub", False):
        return

    already = [
        m
        for m in ["lerobot.policies", *(f"lerobot.policies.smolvla.{n}" for n in _MODULES)]
        if m in sys.modules
    ]
    if already:
        raise RuntimeError(
            f"{already[0]} was already imported. Call use_local_smolvla() before importing lerobot."
        )

    import lerobot  # top-level package is lightweight (no policy imports)

    policies_dir = Path(lerobot.__file__).resolve().parent / "policies"

    # Lazy stub for lerobot.policies: submodules stay importable by full path, the fragile
    # eager __init__ never executes, and the public names resolve on first access.
    policies_stub = types.ModuleType("lerobot.policies")
    policies_stub.__path__ = [str(policies_dir)]
    policies_stub.__package__ = "lerobot.policies"

    def _lazy_getattr(name: str):
        target = _LAZY_EXPORTS.get(name)
        if target is None:
            raise AttributeError(f"module 'lerobot.policies' (lightweight stub) has no attribute {name!r}")
        return getattr(importlib.import_module(target), name)

    policies_stub.__getattr__ = _lazy_getattr
    policies_stub.__all__ = list(_LAZY_EXPORTS)
    policies_stub._local_smolvla_stub = True
    sys.modules["lerobot.policies"] = policies_stub
    lerobot.policies = policies_stub  # parent attribute chain

    # Intermediate package so `import lerobot.policies.smolvla` and submodule-as-module
    # imports both work.
    smolvla_pkg = types.ModuleType("lerobot.policies.smolvla")
    smolvla_pkg.__path__ = [str(_HERE)]
    smolvla_pkg.__package__ = "lerobot.policies.smolvla"
    sys.modules["lerobot.policies.smolvla"] = smolvla_pkg
    policies_stub.smolvla = smolvla_pkg

    specs = {}
    for name in _MODULES:
        fqn = f"lerobot.policies.smolvla.{name}"
        spec = importlib.util.spec_from_file_location(fqn, _HERE / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[fqn] = module  # pre-seed so cross-imports resolve to these
        setattr(smolvla_pkg, name, module)
        specs[fqn] = spec

    for fqn, spec in specs.items():
        spec.loader.exec_module(sys.modules[fqn])
