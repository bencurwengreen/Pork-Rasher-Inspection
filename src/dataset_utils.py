"""Dataset file validation helpers."""

import hashlib


def calculate_sha256(file_path):
    """Return a SHA-256 digest of a file's exact bytes.

    Parameters
    ----------
    file_path : str or pathlib.Path
        Path to a readable file. No image decoding is performed; any binary
        file is accepted, including an empty file.

    Returns
    -------
    str
        Lowercase, 64-character hexadecimal digest. Equal digests identify
        byte-identical files for this project's duplicate checks, irrespective
        of filename. Re-encoded or exposure-adjusted images may hash differently.

    Raises
    ------
    FileNotFoundError
        If the path does not exist.
    PermissionError
        If the file cannot be read due to access restrictions.
    OSError
        If opening or reading the file otherwise fails.

    Notes
    -----
    Reads in 8192-byte chunks to bound memory use and never modifies the file.
    """

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        # Read in chunks to avoid loading large files into memory.
        for chunk in iter(lambda: file.read(8192), b""):
            sha256.update(chunk)

    return sha256.hexdigest()
