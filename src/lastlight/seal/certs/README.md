# Pinned timestamp authority certificates

`lastlight verify` and `lastlight seal` trust an RFC 3161 timestamp only if the pinned authority signed it. The authority is FreeTSA (`lastlight.seal.tsa.DEFAULT_TSA_URL`), and these two files are the only certificates the check trusts. Nothing else is: not the certificates a response carries, not the environment, not a path or a command-line option.

Fetched on 2026-10-11, byte for byte as served, and committed unchanged.

| File | Source | SHA-256 of the DER certificate | Valid from | Valid until |
| --- | --- | --- | --- | --- |
| `freetsa-root.pem` | <https://freetsa.org/files/cacert.pem> (root CA) | `a6379e7cecc05faa3cbf076013d745e327bbbaa38c0b9af22469d4701d18aabc` | 2016-03-13 | 2041-03-07 |
| `freetsa-tsa.pem` | <https://freetsa.org/files/tsa.crt> (TSA certificate, issued by the root) | `32e841a95cc1164101ffde41298ef2fc75c1c4372ef095e88a6bbd47dfb191fc` | 2026-02-15 | 2040-02-02 |

The fingerprint is the SHA-256 of the certificate's DER encoding, what `openssl x509 -in FILE -noout -fingerprint -sha256` prints (without the colons, in lower case). It is also written as a constant in `lastlight/seal/trust.py`, and the loader refuses a file that does not match, so swapping a PEM is an error and not a quiet change of trust.

## Where the pins come from, and why they can be believed

The two files were downloaded from the URLs above. They are byte-identical, as DER, to the two certificates FreeTSA embedded in the response recorded in `tests/fixtures/seal/stub-forecast.tsr`, which came from the service itself on a different occasion; a test pins that. The same recorded response verifies against these pins offline, and the same response with its signature, imprint or time altered does not.

## What is checked

For each timestamp, in this order, and a failure names the first check that fails:

1. the response is granted (status 0; "granted with modifications" is refused);
2. the certificate that signed it is the pinned TSA certificate, which was issued by the pinned root (the certificate chain);
3. both certificates were valid at the time the token states;
4. the TSA certificate carries a critical extended key usage that holds `id-kp-timeStamping` and no other purpose (RFC 3161 section 2.3);
5. the signature verifies.

Then the caller compares the imprint with the bytes it sealed.

## Validity and expiry

The certificates are judged at the time the token names, not today. That is RFC 3161's rule: a timestamp issued while the certificate was valid stays valid after the certificate expires. So the TSA certificate's end date above is when FreeTSA's tokens stop being accepted for new seals; seals made before it still verify afterwards. A token whose time lies outside a certificate's validity fails, and the reason says so; the end instant itself counts as expired, as it does in the library's own check.

## If FreeTSA changes its certificate

The check is a pin, so a new TSA certificate (even one issued by the same root) is refused until it is pinned here. `lastlight seal` then fails loudly, and a missed evening is recorded as missed, never back-filled. To re-pin: fetch the new file from the URL above, compare it with the certificate in a fresh response from the service, replace the PEM, update the fingerprint constant in `trust.py`, this table and the fingerprints written in `tests/test_seal_signature_pins.py`.

Limit: there is one pair, not a list. After a re-pin, seals made under the old TSA certificate verify only with the old pair (check out the commit before the re-pin to verify them). Holding several TSA certificates at once is a small change to `Trust`; it has not been needed, and the current certificate runs to 2040.
