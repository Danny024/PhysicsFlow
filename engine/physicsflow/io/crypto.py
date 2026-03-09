"""
PhysicsFlow — AES-256-GCM Project File Encryption.

Encrypts and decrypts .pfproj files using AES-256-GCM authenticated encryption.
Keys are derived from a user password with PBKDF2-HMAC-SHA256 (600,000 iterations,
random 32-byte salt stored in the file header).

Encrypted file format (.pfproj.enc):
    [4  bytes]  magic number: b'PFEC'
    [1  byte ]  format version: 0x01
    [4  bytes]  PBKDF2 iterations (big-endian uint32)
    [32 bytes]  random salt
    [12 bytes]  AES-GCM nonce
    [16 bytes]  AES-GCM authentication tag
    [N  bytes]  ciphertext

Usage:
    from physicsflow.io.crypto import encrypt_pfproj, decrypt_pfproj

    enc = encrypt_pfproj("study.pfproj", password="s3cr3t!")
    # → PosixPath("study.pfproj.enc")

    dec = decrypt_pfproj("study.pfproj.enc", password="s3cr3t!")
    # → PosixPath("study.pfproj")
"""

from __future__ import annotations

import os
import struct
from pathlib import Path
from typing import Optional

try:
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    _HAS_CRYPTOGRAPHY = True
except ImportError:
    _HAS_CRYPTOGRAPHY = False

# ── Wire format constants ──────────────────────────────────────────────────────

MAGIC           = b"PFEC"
FORMAT_VERSION  = 0x01
PBKDF2_ITERS    = 600_000
SALT_BYTES      = 32
NONCE_BYTES     = 12
TAG_BYTES       = 16
HEADER_SIZE     = len(MAGIC) + 1 + 4 + SALT_BYTES + NONCE_BYTES + TAG_BYTES
ENC_SUFFIX      = ".pfproj.enc"
PLAIN_EXTENSION = ".pfproj"


# ── Public API ─────────────────────────────────────────────────────────────────

def encrypt_pfproj(
    plaintext_path: str | Path,
    password: str,
    output_path: Optional[str | Path] = None,
    remove_original: bool = False,
) -> Path:
    """
    Encrypt a .pfproj file to .pfproj.enc using AES-256-GCM.

    Parameters
    ----------
    plaintext_path  : source .pfproj file
    password        : user passphrase
    output_path     : destination (defaults to <source>.enc)
    remove_original : if True, securely overwrites and deletes the source file

    Returns path to the encrypted .pfproj.enc file.
    Raises ImportError if `cryptography` is not installed.
    """
    _require_crypto()
    src = Path(plaintext_path)
    if not src.exists():
        raise FileNotFoundError(f"Source not found: {src}")

    dst = (Path(output_path) if output_path
           else src.parent / (src.stem + ENC_SUFFIX))
    dst.parent.mkdir(parents=True, exist_ok=True)

    plaintext = src.read_bytes()
    salt      = os.urandom(SALT_BYTES)
    nonce     = os.urandom(NONCE_BYTES)
    key       = _derive_key(password, salt, PBKDF2_ITERS)

    ct_with_tag = AESGCM(key).encrypt(nonce, plaintext, None)
    ciphertext  = ct_with_tag[:-TAG_BYTES]
    tag         = ct_with_tag[-TAG_BYTES:]

    header = (MAGIC
              + bytes([FORMAT_VERSION])
              + struct.pack(">I", PBKDF2_ITERS)
              + salt + nonce + tag)
    dst.write_bytes(header + ciphertext)

    if remove_original:
        _secure_delete(src)

    return dst


def decrypt_pfproj(
    encrypted_path: str | Path,
    password: str,
    output_path: Optional[str | Path] = None,
) -> Path:
    """
    Decrypt a .pfproj.enc file back to .pfproj.

    Parameters
    ----------
    encrypted_path : the .pfproj.enc file
    password       : passphrase used during encryption
    output_path    : destination (defaults to stripping .enc)

    Returns path to the decrypted .pfproj file.
    Raises ValueError on wrong password or tampered file.
    """
    _require_crypto()
    from cryptography.exceptions import InvalidTag

    src = Path(encrypted_path)
    if not src.exists():
        raise FileNotFoundError(f"File not found: {src}")

    data = src.read_bytes()
    if len(data) < HEADER_SIZE:
        raise ValueError("File too short to be a valid PFEC archive")
    if data[:4] != MAGIC:
        raise ValueError(f"Bad magic: expected {MAGIC!r}, got {data[:4]!r}")
    version = data[4]
    if version != FORMAT_VERSION:
        raise ValueError(f"Unsupported PFEC version: {version}")

    offset = 5
    iters  = struct.unpack(">I", data[offset:offset+4])[0]; offset += 4
    salt   = data[offset:offset+SALT_BYTES];                offset += SALT_BYTES
    nonce  = data[offset:offset+NONCE_BYTES];               offset += NONCE_BYTES
    tag    = data[offset:offset+TAG_BYTES];                 offset += TAG_BYTES
    ct     = data[offset:]

    key = _derive_key(password, salt, iters)
    try:
        plaintext = AESGCM(key).decrypt(nonce, ct + tag, None)
    except InvalidTag:
        raise ValueError(
            "Decryption failed — incorrect password or file has been tampered with."
        )

    if output_path:
        dst = Path(output_path)
    elif src.name.endswith(".enc"):
        dst = src.with_name(src.name[:-4])   # strip .enc → .pfproj
    else:
        dst = src.with_suffix(PLAIN_EXTENSION)

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(plaintext)
    return dst


def is_encrypted(path: str | Path) -> bool:
    """Return True if the file is a PFEC-encrypted archive."""
    p = Path(path)
    if not p.exists() or p.stat().st_size < HEADER_SIZE:
        return False
    try:
        return p.read_bytes()[:4] == MAGIC
    except OSError:
        return False


# ── Internal helpers ───────────────────────────────────────────────────────────

def _derive_key(password: str, salt: bytes, iterations: int) -> bytes:
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=iterations,
        backend=default_backend(),
    )
    return kdf.derive(password.encode("utf-8"))


def _secure_delete(path: Path) -> None:
    """Best-effort secure delete: overwrite with zeros then unlink."""
    try:
        size = path.stat().st_size
        with open(path, "r+b") as f:
            f.write(b"\x00" * size)
            f.flush()
            os.fsync(f.fileno())
    except OSError:
        pass
    path.unlink(missing_ok=True)


def _require_crypto() -> None:
    if not _HAS_CRYPTOGRAPHY:
        raise ImportError(
            "The 'cryptography' package is required. Install with: pip install cryptography"
        )
