"""Public inputs to the forecast: model fields and observations fetched from open sources.

One module per source (`hrrr` today; GOES and aerosol sources later), plus the small pieces
they share: `http` is the transport, `grib` decodes GRIB2 messages. Nothing here needs an
account, and nothing here writes outside the cache directory a caller names.
"""
