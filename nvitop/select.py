# This file is part of nputop, the interactive Huawei Ascend NPU process viewer.
# License: GNU GPL version 3.

"""Deprecated CUDA device selection helper.

The original nvitop project ships ``nvisel`` for CUDA-visible GPU selection. This
fork monitors Ascend NPUs through DCMI, so CUDA visibility is intentionally not
part of the public CLI surface.
"""

from __future__ import annotations

import sys
from typing import Any, NoReturn

from nvitop.api import colored


__all__ = ['main', 'select_devices']


def _unsupported() -> NoReturn:
    raise RuntimeError(
        'nvitop.select/nvisel is not supported in nputop. '
        'Use `nputop --only INDEX [...]` to filter physical NPU indices.',
    )


def select_devices(*args: object, **kwargs: object) -> Any:
    """Raise an explicit error for the removed CUDA selector."""
    del args, kwargs
    _unsupported()


def main() -> int:
    """Return a clear CLI error for the removed CUDA selector."""
    try:
        _unsupported()
    except RuntimeError as ex:
        print(
            '{} {}'.format(colored('ERROR:', color='red', attrs=('bold',)), ex),
            file=sys.stderr,
        )
        return 1


if __name__ == '__main__':
    sys.exit(main())
