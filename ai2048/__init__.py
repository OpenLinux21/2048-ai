"""2048 genetic-algorithm AI (Python 3.12 + PyTorch CUDA).

Game environment hierarchy, all sharing one logic contract:
  - shim_env.ShimGame2048 : single game, C shim over the unmodified games/2048.c
  - vec_env.VecEnv2048    : batched torch environment (CPU or CUDA), bit-exact
                            with the shim given identical episode seeds
"""

from ai2048.shim_env import DOWN, LEFT, RIGHT, UP, DIRECTIONS, ShimGame2048

__all__ = ["UP", "RIGHT", "DOWN", "LEFT", "DIRECTIONS", "ShimGame2048"]
