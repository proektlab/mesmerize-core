"""
Useful functions adapted from old mesmerize

GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007
"""


import numpy as np
from functools import wraps
import os
from stat import S_IEXEC
from typing import *
import re as regex
from pathlib import Path
from warnings import warn
import sys
from tempfile import NamedTemporaryFile
from subprocess import check_call
from copy import deepcopy
import pandas as pd

if os.name == "nt":
    IS_WINDOWS = True
    HOME = "USERPROFILE"
else:
    IS_WINDOWS = False
    HOME = "HOME"

if "MESMERIZE_LRU_CACHE" in os.environ.keys():
    MESMERIZE_LRU_CACHE = os.environ["MESMERIZE_LRU_CACHE"]
else:
    MESMERIZE_LRU_CACHE = 10


def warning_experimental(more_info: str = ""):
    """
    decorator to warn the user that the function is experimental
    """
    def catcher(func):
        @wraps(func)
        def fn(self, *args, **kwargs):
            warn(
                f"You are trying to use the following experimental feature, "
                f"this may change in the future without warning:\n"
                f"{func.__qualname__}\n"
                f"{more_info}\n",
                FutureWarning,
                stacklevel=2
            )
            return func(self, *args, **kwargs)
        return fn
    return catcher


def validate_path(path: Union[str, Path]):
    if not regex.match("^[A-Za-z0-9@\/\\\:._-]*$", str(path)):
        raise ValueError(
            "Paths must only contain alphanumeric characters, "
            "hyphens ( - ), underscores ( _ ) or periods ( . )"
        )
    return path


def make_runfile(
    module_path: str, args_str: Optional[str] = None, filename: Optional[str] = None
) -> str:
    """
    Make an executable bash script.
    Used for running python scripts in external processes within the same python environment as the main/parent process.

    Parameters
    ----------
    module_path: str
        absolute path to the python module/script that should be run externally

    args_str: Optional[str]
        optinal str of args that is directly passed to the script specified by ``module_path``

    filename: Optional[str]
        optional, filename of the executable bash script

    Returns
    -------
    str
        path to the shell script that can be executed
    """

    if filename is None:
        if IS_WINDOWS:
            sh_file = os.path.join(os.environ[HOME], "run.ps1")
        else:
            sh_file = os.path.join(os.environ[HOME], "run.sh")
    else:
        if IS_WINDOWS:
            if not filename.endswith(".ps1"):
                filename = filename + ".ps1"

    sh_file = filename

    if args_str is None:
        args_str = ""

    if not IS_WINDOWS:
        # remove file first if it exists - avoid issues with ownership
        if os.path.exists(sh_file):
            os.remove(sh_file)

        with open(sh_file, "w") as f:

            f.write(f"#!/bin/bash\n")

            if "VIRTUAL_ENV" in os.environ.keys():
                f.write(
                    f'export PATH={os.environ["PATH"]}\n'
                    f'export VIRTUAL_ENV={os.environ["VIRTUAL_ENV"]}\n'
                    f'export LD_LIBRARY_PATH={os.environ["LD_LIBRARY_PATH"]}\n'
                )

            if "PYTHONPATH" in os.environ.keys():
                f.write(f'export PYTHONPATH={os.environ["PYTHONPATH"]}\n')

            # for k, v in os.environ.items():  # copy the current environment
            #     if '\n' in v:
            #         continue
            #
            # f.write(f'export {k}="{v}"\n')

            # User-setable n-processes
            if "MESMERIZE_N_PROCESSES" in os.environ.keys():
                f.write(
                    f'export MESMERIZE_N_PROCESSES={os.environ["MESMERIZE_N_PROCESSES"]}\n'
                )

            f.write(
                f"export OPENBLAS_NUM_THREADS=1\n"
                f"export MKL_NUM_THREADS=1\n"
            )

            if "CONDA_PREFIX" in os.environ.keys():
                # add command to run the python script in the conda environment
                # that was active at the time that this shell script was generated
                f.write(f'{os.environ["CONDA_EXE"]} run -p {os.environ["CONDA_PREFIX"]} python {module_path} {args_str}')
            else:
                f.write(f"python {module_path} {args_str}")  # call the script to run

    else:
        with open(sh_file, "w") as f:
            for k, v in os.environ.items():  # copy the current environment
                if regex.match("^.*[\(\)]", str(k)) or regex.match("^.*[\(\)]", str(v)):
                    continue
                with NamedTemporaryFile(suffix=".ps1", delete=False) as tmp:
                    try:  # windows powershell is stupid so make sure all the env var names work
                        tmp.write(f'$env:{k}="{v}";\n')
                        tmp.close()
                        check_call(f"powershell {tmp.name}")
                        os.unlink(tmp.name)
                    except:
                        continue
                f.write(f'$env:{k}="{v}";\n')  # write only env vars that powershell likes
            f.write(f"{sys.executable} {module_path} {args_str}")

    st = os.stat(sh_file)
    os.chmod(sh_file, st.st_mode | S_IEXEC)

    print(sh_file)

    return sh_file


def quick_min_max(data: np.ndarray) -> Tuple[float, float]:
    # from pyqtgraph.ImageView
    # Estimate the min/max values of *data* by subsampling.
    # Returns [(min, max), ...] with one item per channel
    while data.size > 1e6:
        ax = np.argmax(data.shape)
        sl = [slice(None)] * data.ndim
        sl[ax] = slice(None, None, 2)
        data = data[tuple(sl)]

    return float(np.nanmin(data)), float(np.nanmax(data))


def _organize_coordinates(contour: dict):
    coors = contour["coordinates"]
    coors = coors[~np.isnan(coors).any(axis=1)]

    return coors


def flatten_params(params_dict: dict) -> dict:
    """
    Produce a flat dict with one entry for each parameter in the passed dict.
    If params_dict['main'] is nested one level (e.g., {'init': {'K': 5}, 'merging': {'merge_thr': 0.85}}...),
    each key in the output is <outerKey>.<innerKey>, e.g., [(init.K, 5), (merging.merge_thr, 0.85)]
    """
    params = {}
    for key1, val1 in params_dict.items():
        if key1 == "main":
            # recursively step into "main" params
            params.update(flatten_params(val1))
        elif isinstance(val1, dict):  # nested
            for key2, val2 in val1.items():
                params[f"{key1}.{key2}"] = val2
        else:
            params[key1] = val1
    return params
        

def get_params_diffs(params: Sequence[dict]) -> list[dict]:
    """Compute differences between params used for mesmerize"""
    # get flattened parameters for each of the filtered items
    params_flat = list(map(flatten_params, params))

    # build list of params that differ between different parameter sets
    common_params = deepcopy(params_flat[0])  # holds the common value for parameters found in all sets (so far)
    varying_params = set()  # set of parameter keys that appear in not all sets or with varying values

    for this_params in params_flat[1:]:
        # first, anything that's not in both this dict and the common set is considered varying
        common_paramset = set(common_params.keys())
        for not_common_key in common_paramset.symmetric_difference(this_params.keys()):
            varying_params.add(not_common_key)
            if not_common_key in common_paramset:
                del common_params[not_common_key]
                common_paramset.remove(not_common_key)

        # second, look at params in the common set and remove any that differ for this set
        for key in common_paramset:  # iterate over this set rather than dict itself to avoid issues when deleting entries
            if not np.array_equal(common_params[key], this_params[key]):  # (should also work for scalars/arbitrary objects)
                varying_params.add(key)
                del common_params[key]

    # gives a list where each item is a dict that has the unique params that correspond to a row
    return [{key: p[key] if key in p else "<default>" for key in varying_params} for p in params_flat]
