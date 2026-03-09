"""
PhysicsFlow CLI — Project File Encryption Commands.

    physicsflow-encrypt  study.pfproj
    physicsflow-decrypt  study.pfproj.enc
"""

from __future__ import annotations

import getpass
import sys
from pathlib import Path

import click

from ..io.crypto import decrypt_pfproj, encrypt_pfproj, is_encrypted


@click.command("encrypt-project")
@click.argument("path", type=click.Path(exists=True, dir_okay=False))
@click.option("--output",  "-o", default=None,  help="Output path (default: <input>.enc)")
@click.option("--remove-original", is_flag=True, default=False,
              help="Securely delete the plaintext file after encryption")
@click.option("--password", "-p", default=None,
              help="Password (omit to be prompted securely)")
def encrypt_cmd(path: str, output: str | None, remove_original: bool, password: str | None):
    """Encrypt a .pfproj project file using AES-256-GCM."""
    src = Path(path)

    if is_encrypted(src):
        click.echo(f"ERROR: {src.name} is already encrypted.", err=True)
        sys.exit(1)

    if password is None:
        password = getpass.getpass("Enter encryption password: ")
        confirm  = getpass.getpass("Confirm password: ")
        if password != confirm:
            click.echo("ERROR: Passwords do not match.", err=True)
            sys.exit(1)

    try:
        dst = encrypt_pfproj(src, password=password,
                             output_path=output, remove_original=remove_original)
        click.echo(f"Encrypted: {dst}")
        if remove_original:
            click.echo(f"Original securely deleted: {src}")
    except Exception as exc:
        click.echo(f"ERROR: {exc}", err=True)
        sys.exit(1)


@click.command("decrypt-project")
@click.argument("path", type=click.Path(exists=True, dir_okay=False))
@click.option("--output",  "-o", default=None,  help="Output path (default: strip .enc)")
@click.option("--password", "-p", default=None,
              help="Password (omit to be prompted securely)")
def decrypt_cmd(path: str, output: str | None, password: str | None):
    """Decrypt a .pfproj.enc file back to .pfproj."""
    src = Path(path)

    if not is_encrypted(src):
        click.echo(f"ERROR: {src.name} does not appear to be a PFEC-encrypted file.", err=True)
        sys.exit(1)

    if password is None:
        password = getpass.getpass("Enter decryption password: ")

    try:
        dst = decrypt_pfproj(src, password=password, output_path=output)
        click.echo(f"Decrypted: {dst}")
    except ValueError as exc:
        click.echo(f"ERROR: {exc}", err=True)
        sys.exit(1)
    except Exception as exc:
        click.echo(f"ERROR: {exc}", err=True)
        sys.exit(1)
