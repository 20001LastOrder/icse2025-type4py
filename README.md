# Type Checking for Neural Bug Detector

This repository contains the artifacts for the paper "The Power of Types: Exploring the Impact of Type Checking on Neural Bug Detection in Dynamically Typed Languages". Specifically, the files follow the structure below:

- `fine-tuning`: contains the code for fine-tuning pre-trained language models for the variable misuse tasks
- `great_tf`: contains the code for the neural bug detector of GGNN and Great used for the experiments. This code is based on the code from paper: Global Relational Models of Source Code ([18] in the paper)
- `measurements` contains material to reproduce figures and tables of RQ1-4 in the paper, also including the type checking results
- `data.zip` contains the dataset used in the experiments (needs to be decompressed)


## Data
Download the dataset used for the paper [here](https://figshare.com/s/9628d556e58daef5bb84)


# Examples of Variable Misuse Bugs
Below are three examples of variable misuse bugs from the datasets used in the paper. All the the examples are _syntactically correct_ but will cause runtime errors when the program executes. We show one example where `pytype` can detect the bug even when type annotation are not used, one example where `pytype` can detect the bug when type annotations exist and one example where `pytype` cannot detect the bug even when type annotations are used.

## Example 1
```python
def take_last_assignment(source):
    first=True
    last=None
    for assn in source:
        if first:
            last=assn
            first=False
        if (assn[1]!=first[1]):
            (yield last)
        last=assn
    if (last is not None):
        (yield last)
```

The variable misuse bug happens on line 8 ```if (assn[1]!=first[1]):```. `first` is a boolean, which can de deducted from the initial assignment `first=True`. Executing `pytype` check on this line will raise a `unsupported-operands` error. The correct variable should be `last` instead of `first`. Following is the full error message from `pytype`

```
line 8, in take_last_assignment: unsupported operand type(s) for item retrieval: 'first: bool' and '1: int' [unsupported-operands]
  No attribute '__getitem__' on 'first: bool'

For more details, see https://google.github.io/pytype/errors.html#unsupported-operands
ninja: build stopped: subcommand failed.
Leaving directory '.pytype'
```

## Example 2
```python
def multi_scale_cross_entropy2d(input, target, weight=None, size_average=True, scale_weight=None):
    if not isinstance(input, tuple):
        return cross_entropy2d(input=input, target=target, weight=weight, size_average=size_average)

    # Auxiliary training for PSPNet [1.0, 0.4] and ICNet [1.0, 0.4, 0.16]
    if scale_weight is None:  # scale_weight: torch tensor type
        n_inp = len(input)
        scale = 0.4
        scale_weight = torch.pow(scale * torch.ones(n_inp), torch.arange(n_inp).float()).to(
            input.device
        )

    loss = 0.0
    for i, inp in enumerate(input):
        loss = loss + scale_weight[i] * cross_entropy2d(
            input=inp, target=target, weight=weight, size_average=size_average
        )

    return loss
```

This is a real-world variable misuse bug example from the `pytorch-semseg` repository. Here is the [link to the fix commit](https://github.com/meetps/pytorch-semseg/commit/801fb200547caa5b0d91b8dde56b837da029f746).

The variable misuse happens on line 11 `input.device`. However, the `pytype` is not able to detect this bug because it is not able to use the information from the `isinstance` function which operates at runtime. From this line: `if not isinstance(input, tuple):`, we can infer that `input` must be tuple at line `11`. Thus adding annotations to the function signature will help `pytype` to detect this bug.

```python
from typing import Any, Optional, Tuple, Union

import torch
import torch.nn.functional as F


def cross_entropy2d(input: torch.Tensor, target: torch.Tensor, weight: Any = None, size_average: bool = True) -> torch.Tensor:
    # stub for the function
    return torch.zeros(1)

def multi_scale_cross_entropy2d(input: Union[torch.Tensor, Tuple[torch.Tensor]], target: torch.Tensor, weight: Any=None, size_average: bool=True, scale_weight: Optional[torch.Tensor]=None) -> float:
    if not isinstance(input, tuple):
        return cross_entropy2d(input=input, target=target, weight=weight, size_average=size_average)

    # Auxiliary training for PSPNet [1.0, 0.4] and ICNet [1.0, 0.4, 0.16]
    if scale_weight is None:  # scale_weight: torch tensor type
        n_inp: int = len(input)
        scale: float = 0.4
        scale_weight: torch.Tensor = torch.pow(scale * torch.ones(n_inp), torch.arange(n_inp).float()).to(
            input.device
        )

    loss: float = 0.0
    for i, inp in enumerate(input):
        loss = loss + scale_weight[i] * cross_entropy2d(
            input=inp, target=target, weight=weight, size_average=size_average
        )

    return loss
```

In this case, an `attribute-error` will be raised by `pytype` since `Tuple` does not have an attribute `device`. The full error message is as follows:

```
line 19, in multi_scale_cross_entropy2d: No attribute 'device' on Tuple[Any] [attribute-error]
  In Union[Any, Tuple[Any]]

For more details, see https://google.github.io/pytype/errors.html#attribute-error
ninja: build stopped: subcommand failed.
Leaving directory '.pytype'
```

## Example 3
```python
def _recursive_directory_find(path, directory_name):
    if str(path) == expanduser('~'):
        raise FileNotFoundError()
    joined_with_driectory = path.joinpath(directory_name)
    if joined_with_driectory.is_dir():
        return str(path)
    else:
        return _recursive_directory_find(path.parent, directory_name)
```

This error occurs in the `guet` project ([fixing commit](https://github.com/chiptopher/guet/commit/f95c54917b51e65f47789534ab88ecbede1838eb)). The variable misuse location is on line 6, where the string form of `path` is return. 

The purpose of this function is to identify the absolute path. Thus, the correct variable should be `joined_with_directory` instead of `path`. However, `pytype` is not able to detect this bug, even when type annotation is used. This is because both `path` and `joined_with_directory` are of the same type `pathlib.Path`. Thus a neural bug detector will be helpful to identify the bug in this case.