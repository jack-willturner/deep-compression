from .l1_pruner import L1Pruner

__all__ = ["L1Pruner", "get_pruner"]


def get_pruner(pruner):
    if pruner == "L1Pruner":
        return L1Pruner()
    else:
        raise NotImplementedError
