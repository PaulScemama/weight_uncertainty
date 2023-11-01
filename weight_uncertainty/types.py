from typing import Any, Iterable, Mapping, Union

import jax
from jax.typing import ArrayLike

"""
This is ported from https://github.com/blackjax-devs/blackjax/blob/main/blackjax/types.py

Following the current best practice (https://jax.readthedocs.io/en/latest/jax.typing.html)
We use:
- `ArrayLike` and `ArrayLikeTree` to annotate function input,
- `Array` and `ArrayTree` to annotate function output.
"""

ArrayTree = Union[jax.Array, Iterable["ArrayTree"], Mapping[Any, "ArrayTree"]]
ArrayLikeTree = Union[
    ArrayLike, Iterable["ArrayLikeTree"], Mapping[Any, "ArrayLikeTree"]
]
