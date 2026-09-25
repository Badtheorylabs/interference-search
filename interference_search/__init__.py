"""Interference Search: reason over explicit states, many branches at once.

Every live branch expands together, the environment executes the moves, branches that reach the same
state merge, a judge cancels dead ends, and the survivors advance one level at a time.
"""
from .core import Result, linear_search, search

__all__ = ["Result", "search", "linear_search"]
__version__ = "0.1.0"
