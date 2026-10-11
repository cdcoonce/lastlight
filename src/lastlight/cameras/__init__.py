"""Tier A truth: the registry of public air-quality webcams and the polite collector.

`registry` says which cameras exist and what may be published from them; `collector` waits for
each evening's twilight window and archives the frames privately on the capture host;
`transport`, `clock` and `archive` are its injectable HTTP layer, time source and storage.
"""

from lastlight.cameras.registry import (
    Camera,
    Position,
    RegistryError,
    load_registry,
    parse_registry,
)

__all__ = ["Camera", "Position", "RegistryError", "load_registry", "parse_registry"]
