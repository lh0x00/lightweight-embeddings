"""Optional Gradio UI mounted at the root path.

Importing this package does not import gradio until :func:`build_demo` is
called, so headless deployments can disable the UI without paying its
import cost.
"""

from .ui import build_demo, mount_demo

__all__ = ["build_demo", "mount_demo"]
