"""Performs CNMF in a separate process"""

import click
import caiman as cm
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.source_extraction.cnmf.params import CNMFParams
from caiman.summary_images import local_correlations_movie_offline
from caiman.paths import decode_mmap_filename_dict
import numpy as np
import traceback
from pathlib import Path, PurePosixPath
from shutil import move as move_file
import time

# prevent circular import
if __name__ in ["__main__", "__mp_main__"]:  # when running in subprocess
    from mesmerize_core import set_parent_raw_data_path, load_batch
    from mesmerize_core.utils import IS_WINDOWS
    from mesmerize_core.algorithms._utils import (
        ensure_server,
        save_projections_parallel,
        setup_logging,
    )
else:  # when running with local backend
    from ..batch_utils import set_parent_raw_data_path, load_batch
    from ..utils import IS_WINDOWS
    from ._utils import ensure_server, save_projections_parallel, setup_logging


def run_algo(batch_path, uuid, data_path: str = None, dview=None, log_level=None):
    if log_level is not None:
        setup_logging(log_level)
    algo_start = time.time()
    set_parent_raw_data_path(data_path)

    df = load_batch(batch_path)
    item = df.caiman.uloc(uuid)

    input_movie_path = item["input_movie_path"]
    # resolve full path
    input_movie_path = str(df.paths.resolve(input_movie_path))

    # make output dir
    output_dir = Path(batch_path).parent.joinpath(str(uuid)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    params = item["params"]
    print(
        f"************************************************************************\n\n"
        f"Starting CNMF item:\n{item}\nWith params:{params}"
    )

    with ensure_server(dview) as (dview, n_processes):
        # merge cnmf and eval kwargs into one dict
        cnmf_params = CNMFParams(params_dict=params["main"])
        # Run CNMF, denote boolean 'success' if CNMF completes w/out error
        try:
            # only re-save memmap if necessary
            save_new_mmap = True
            if Path(input_movie_path).suffix == ".mmap":
                mmap_info = decode_mmap_filename_dict(input_movie_path)
                save_new_mmap = "order" not in mmap_info or mmap_info["order"] != "C"

            if save_new_mmap:
                print("making memmap")
                fname_new = cm.save_memmap(
                    [input_movie_path],
                    base_name=f"{uuid}_cnmf-memmap_",
                    order="C",
                    dview=dview,
                )
                cnmf_memmap_path = output_dir.joinpath(Path(fname_new).name)
                move_file(fname_new, cnmf_memmap_path)
            else:
                cnmf_memmap_path = Path(input_movie_path)

            Yr, dims, T = cm.load_memmap(str(cnmf_memmap_path))
            images = np.reshape(Yr.T, [T] + list(dims), order="F")

            print("computing projections")
            proj_paths = save_projections_parallel(
                uuid=uuid,
                movie_path=cnmf_memmap_path,
                output_dir=output_dir,
                dview=dview,
            )

            print("computing correlation image")
            Cns = local_correlations_movie_offline(
                str(cnmf_memmap_path),
                remove_baseline=True,
                window=1000,
                stride=1000,
                winSize_baseline=100,
                quantil_min_baseline=10,
                dview=dview,
            )
            Cn = Cns.max(axis=0)
            Cn[np.isnan(Cn)] = 0
            corr_img_path = output_dir.joinpath(f"{uuid}_cn.npy")
            np.save(str(corr_img_path), Cn, allow_pickle=False)

            # # in fname new load in memmap order C
            # cm.stop_server(dview=dview)
            # c, dview, n_processes = cm.cluster.setup_cluster(
            #     backend="local", n_processes=None, single_thread=False
            # )

            # load Ain if given
            if "Ain_path" in params and params["Ain_path"] is not None:
                Ain_path_abs = (
                    output_dir / params["Ain_path"]
                )  # resolve relative to output dir
                Ain = np.load(Ain_path_abs, allow_pickle=True)
                if Ain.size == 1:  # sparse array loaded as object
                    Ain = Ain.item()
            else:
                Ain = None

            print("performing CNMF")
            cnm = cnmf.CNMF(n_processes, params=cnmf_params, dview=dview, Ain=Ain)

            print("fitting images")
            cnm = cnm.fit(images)
            #
            if "refit" in params.keys():
                if params["refit"] is True:
                    print("refitting")
                    cnm = cnm.refit(images, dview=dview)

            print("performing eval")
            cnm.estimates.evaluate_components(images, cnm.params, dview=dview)

            output_path = output_dir.joinpath(f"{uuid}.hdf5")

            cnm.save(str(output_path))

            # output dict for dataframe row (pd.Series)
            d = dict()

            if IS_WINDOWS:
                Yr._mmap.close()  # accessing private attr but windows is annoying otherwise

            # save paths as relative path strings with forward slashes
            cnmf_hdf5_path = str(
                PurePosixPath(output_path.relative_to(output_dir.parent))
            )
            cnmf_memmap_path = str(
                PurePosixPath(df.paths.split(cnmf_memmap_path)[1])
            )  # still work if outside output dir
            corr_img_path = str(
                PurePosixPath(corr_img_path.relative_to(output_dir.parent))
            )
            for proj_type in proj_paths.keys():
                d[f"{proj_type}-projection-path"] = str(
                    PurePosixPath(proj_paths[proj_type].relative_to(output_dir.parent))
                )

            d.update(
                {
                    "cnmf-hdf5-path": cnmf_hdf5_path,
                    "cnmf-memmap-path": cnmf_memmap_path,
                    "corr-img-path": corr_img_path,
                    "success": True,
                    "traceback": None,
                }
            )

        except:
            d = {"success": False, "traceback": traceback.format_exc()}

    runtime = round(time.time() - algo_start, 2)
    df.caiman.update_item_with_results(uuid, d, runtime)


@click.command()
@click.option("--batch-path", type=str)
@click.option("--uuid", type=str)
@click.option("--data-path", default=None)
@click.option("--log-level", type=int, default=None)
def main(batch_path, uuid, data_path, log_level):
    run_algo(batch_path, uuid, data_path, log_level=log_level)


if __name__ == "__main__":
    main()
