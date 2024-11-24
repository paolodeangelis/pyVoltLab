import sys
from typing import TextIO

from mc_mace import __authors__, __license__, __version__, __year__

HEADER = r"""        _\|/_
        (o o)
+----oOO-{_}-OOo-------------------------------------------+
|                                                          |
|                                                          |
|                 ____        __  __  ____                 |
|                |  _ \ _   _|  \/  |/ ___|                |
|                | |_) | | | | |\/| | |                    |
|                |  __/| |_| | |  | | |___                 |
|                |_|    \__, |_|  |_|\____|                |
|                       |___/                              |
|                                                          |
|                                                          |
|                                                          |
+----------------------------------------------------------+
"""


def print_header(output: TextIO | None = sys.stdout) -> None:
    """
    Print a decorative header with copyright and license information.

    The header includes a logo, version of the application, and copyright/license details.
    The output can be directed to the console or a file.

    Args:
        output (Optional[TextIO]): The output stream to write the header to.
            Defaults to sys.stdout for console output.

    Example:
        # Print to the console
        print_header()

        # Write to a file
        with open("header.txt", "w") as file:
            print_header(output=file)
    """
    for line in HEADER.splitlines():
        print(" " * 30 + line, file=output)

    # Print version, copyright, and license
    version_line = " " * 30 + f"version: {__version__}".rjust(60, " ")
    print(version_line, file=output)
    print("", file=output)  # Add a blank line
    print(
        " " * 30 + f"(c) {__year__}, {__authors__[0]['name']}".ljust(60, " "),
        file=output,
    )
    print(
        " " * 30 + f"Distributed under the {__license__} License".ljust(60, " "),
        file=output,
    )
    print(" " * 30 + "Author(s):".ljust(60, " "), file=output)
    for author in __authors__:
        print(
            " " * 30 + f"    {__authors__[0]['name']} ({__authors__[0]['email']})".ljust(60, " "),
            file=output,
        )
    print("", file=output)  # Add a blank line
