"""The peft/torchao shim exists because of a distinction that is easy to miss.

peft skips torchao when it is *absent* but raises when it is *present and old*, which is
why this repository's tests passed locally and Colab failed. The shim has to reach every
module that bound the helper by name, not just the one that defines it.
"""

import pytest


def _peft_modules():
    pytest.importorskip("peft")
    import peft.import_utils  # noqa: F401
    import peft.tuners.lora.torchao  # noqa: F401
    import peft.utils.quantization_utils  # noqa: F401
    import sys

    return [
        module
        for module in sys.modules.values()
        if getattr(module, "__name__", "").startswith("peft")
        and hasattr(module, "is_torchao_available")
    ]


def test_shim_does_nothing_when_torchao_is_fine():
    """On a healthy environment it must not touch peft at all."""
    from src.compat import neutralize_peft_torchao_check

    _peft_modules()
    assert neutralize_peft_torchao_check() == 0


def test_shim_reaches_every_module_that_bound_the_helper():
    from src.compat import neutralize_peft_torchao_check

    modules = _peft_modules()
    assert len(modules) > 1, "the interesting case is the helper being bound in several modules"
    originals = {module: module.is_torchao_available for module in modules}

    def raise_like_colab():
        raise ImportError(
            "Found an incompatible version of torchao. Found version 0.10.0, "
            "but only versions above 0.16.0 are supported"
        )

    try:
        for module in modules:
            module.is_torchao_available = raise_like_colab

        patched = neutralize_peft_torchao_check()

        assert patched == len(modules)
        for module in modules:
            assert module.is_torchao_available() is False
    finally:
        for module, original in originals.items():
            module.is_torchao_available = original
