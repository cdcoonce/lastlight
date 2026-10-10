"""Subcommands of the `lastlight` CLI, one module per command.

To add a command, drop a module here that defines a module-level
`COMMAND = Command(name=..., help=..., run=..., configure=...)`
(see `lastlight.cli.Command`). `lastlight.cli.discover` imports every public module
of this package, so slices never edit a shared list and cannot conflict on one.
Keep modules here thin: the logic lives in the module the design assigns it to
(for example `lastlight.privacy`), and the command only parses arguments and calls it.
Modules whose names start with an underscore are skipped.
"""
