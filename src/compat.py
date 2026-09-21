"""Shims for environments we do not control — in practice, Google Colab."""

from __future__ import annotations

import sys


def neutralize_peft_torchao_check() -> int:
    """Stop peft refusing to load beside an outdated torchao.

    `peft.import_utils.is_torchao_available()` returns False when torchao is missing, but
    *raises* ImportError when torchao is installed and older than peft's minimum. Colab
    preinstalls torchao 0.10 while current peft wants 0.16+, so `get_peft_model` fails
    there — even though nothing in this repository quantises anything and torchao is
    never used.

    Uninstalling torchao is the real fix and the notebooks do that first. This is the
    fallback for a session where that did not take effect: a clone made before the fix, a
    runtime restored from a snapshot, a preinstall that reappears.

    Several peft modules bind the helper by name (`from peft.import_utils import
    is_torchao_available`), so rebinding it in one module is not enough. Every loaded peft
    module holding a reference is updated, including `peft.import_utils` itself, which
    covers modules imported later.

    Returns the number of references patched; 0 means nothing needed doing.
    """
    try:
        import peft.import_utils as import_utils
    except ImportError:
        return 0

    try:
        import_utils.is_torchao_available()
    except ImportError:
        pass  # the incompatible-version path — this is what we are here to defuse
    else:
        return 0  # torchao is absent or acceptable; leave peft alone

    def _torchao_unavailable() -> bool:
        return False

    patched = 0
    for module in list(sys.modules.values()):
        name = getattr(module, "__name__", "")
        if name.startswith("peft") and hasattr(module, "is_torchao_available"):
            module.is_torchao_available = _torchao_unavailable
            patched += 1
    return patched
