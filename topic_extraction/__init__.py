from __future__ import annotations

import sys
if sys.version_info >= (3, 8):
    from importlib.metadata import distribution
else:
    from importlib_metadata import distribution

__version__ = distribution('topic_extraction').version
