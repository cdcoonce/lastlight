"""Seal a forecast before sunset and prove later that it was sealed (`docs/design.md` section 2).

`canonical` turns a `lastlight.forecast.Forecast` into one reproducible byte string and its
SHA-256 digest; `tsa` gets that digest witnessed by an RFC 3161 timestamp authority and checks
the reply is genuine; `trust` pins the authority's certificates; `ledger` writes the sealed files
(forecast, timestamp, and the strip the forecast names) and checks them again, the timestamp's
signature included. The `lastlight seal` and `lastlight verify` commands (in
`lastlight.commands`) are thin wrappers over `ledger`.
"""

from lastlight.seal.canonical import canonical_bytes, digest_hex
from lastlight.seal.ledger import (
    AlreadySealedError,
    LedgerPaths,
    SealResult,
    StripError,
    VerifyResult,
    is_sealed,
    ledger_paths,
    seal,
    verify,
)
from lastlight.seal.trust import Trust, TrustError
from lastlight.seal.tsa import DEFAULT_TSA_URL, Timestamp, TokenRejected, TsaError

__all__ = [
    "DEFAULT_TSA_URL",
    "AlreadySealedError",
    "LedgerPaths",
    "SealResult",
    "StripError",
    "Timestamp",
    "TokenRejected",
    "Trust",
    "TrustError",
    "TsaError",
    "VerifyResult",
    "canonical_bytes",
    "digest_hex",
    "is_sealed",
    "ledger_paths",
    "seal",
    "verify",
]
