"""syndrome-stream: streaming (sliding-window) decoding for quantum LDPC and surface codes."""

from .window import Problem, Window, decode, from_dem, plan, pymatching_decoder

__all__ = ["Problem", "Window", "decode", "from_dem", "plan", "pymatching_decoder"]
