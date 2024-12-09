import torch
from inspect import signature


def get_target(module: torch.nn.Module) -> bool:
    forward = module.forward
    args = signature(forward)
    return "target" in args.parameters


def get_input(module:torch.nn.Module) -> bool:
    forward = module.forward
    args = signature(forward)
    return "img" in args.parameters
