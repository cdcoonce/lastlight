"""The almanac: a static site built from the ledger (`docs/design.md` section 4, `publish`).

`read_ledger` reads the sealed evenings and re-runs `lastlight verify` on each; `pages` turns
them into HTML; `build_site` writes the site and refuses to finish if the privacy scanner finds a
clock time, a stray coordinate or image metadata in it. `lastlight site build` is the command.

    lastlight site build --ledger-dir ledger --out site [--through YYYY-MM-DD]

An evening between the first sealed one and `--through` (by default the newest evening whose
deadline has passed) that has no seal is listed as missed (`lastlight.publish.missed`).
"""

from lastlight.publish.build import BuildError, BuildReport, build_site
from lastlight.publish.ledger import Evening, Ledger, read_ledger

__all__ = [
    "BuildError",
    "BuildReport",
    "Evening",
    "Ledger",
    "build_site",
    "read_ledger",
]
