from .l1_pruner import L1Pruner

__all__ = ["L1Pruner", "get_pruner"]


def get_pruner(pruner, pruning_type):
    if pruner == "L1Pruner":
        return L1Pruner(pruning_type)
    else:
        raise NotImplementedError
