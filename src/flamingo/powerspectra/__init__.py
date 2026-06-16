"""Power-spectrum estimation (optional, requires pymaster/NaMaster).

Importing this subpackage raises a clear ImportError if pymaster is missing.
"""
from .namaster import apodize, decoupled_dl

__all__ = ["apodize", "decoupled_dl"]
