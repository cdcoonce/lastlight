"""The privacy scanner (docs/design.md section 3): `lastlight privacy-scan`.

The rules are P001 clock-time, P002 sun-event-time, P003 coordinates, P004 image-metadata and
P005 private-site. `scan` applies them to files; `load_allowlist` reads the reviewed
exceptions; `contains_clock_time` lets other modules reuse the P001 pattern.
"""

from lastlight.privacy.allowlist import AllowEntry, Allowlist, load_allowlist
from lastlight.privacy.findings import Finding
from lastlight.privacy.rules import RULES, contains_clock_time
from lastlight.privacy.scanner import ScanError, ScanResult, find_root, scan

__all__ = [
    "RULES",
    "AllowEntry",
    "Allowlist",
    "Finding",
    "ScanError",
    "ScanResult",
    "contains_clock_time",
    "find_root",
    "load_allowlist",
    "scan",
]
