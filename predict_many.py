import os
import argparse
from datetime import datetime
from glob import glob
from tqdm.contrib.concurrent import process_map
import torch
import numpy as np
import pyrosetta

import deepab
from deepab.models.AbResNet import load_model
from deepab.models.ModelEnsemble import ModelEnsemble
from deepab.build_fv.build_cen_fa import build_initial_fv, get_cst_defs, refine_fv
from deepab.metrics.rosetta_ab import get_ab_metrics
from deepab.util.pdb import renumber_pdb

from functional import seq
from pathlib import Path
def children(p: Path):
    """
    files and subfolders in the folder as sequence
    :param p:
    :return:
    """
    return seq(list(p.iterdir()))

def files(p: Path):
    """
    only files in the folder
    :param p:
    :return:
    """
    return children(p).filter(lambda f: f.is_file())

def with_ext(p: Path, ext: str):
    """
    files in the folder that have appropriate extension
    :param p:
    :param ext:
    :return:
    """
    return files(p).filter(lambda f: ext in f.suffix)

def prog_print(text):
    print("*" * 50)
    print(text)
    print("*" * 50)


def refine_fv_(args):
    in_pdb_file, out_pdb_file, cst_defs = args
    return refine_fv(in_pdb_file, out_pdb_file, cst_defs)

def build_structure(model,
                    fasta_file,
                    cst_defs,
                    out_dir,
                    target="pred",
                    num_decoys=5,
                    num_procs=1,
                    single_chain=False,
                    device=None):
    decoy_dir = os.path.join(out_dir, "decoys")
    os.makedirs(decoy_dir, exist_ok=True)

    prog_print("Creating MDS structure")
    mds_pdb_file = os.path.join(decoy_dir, "{}.mds.pdb".format(target))
    build_initial_fv(fasta_file,
                     mds_pdb_file,
                     model,
                     single_chain=single_chain,
                     device=device)

    prog_print("Creating decoys structures")
    decoy_pdb_pattern = os.path.join(decoy_dir,
                                     "{}.deepab.{{}}.pdb".format(target))
    refine_args = [(mds_pdb_file, decoy_pdb_pattern.format(i), cst_defs)
                   for i in range(num_decoys)]
    decoy_scores = process_map(refine_fv_, refine_args, max_workers=num_procs)

    best_decoy_i = np.argmin(decoy_scores)
    best_decoy_pdb = decoy_pdb_pattern.format(best_decoy_i)
    out_pdb = os.path.join(out_dir, "{}.deepab.pdb".format(target))
    os.system("cp {} {}".format(best_decoy_pdb, out_pdb))

    return out_pdb

def run_many(fasta_files: list, predictions_dir: Path, single_chain: bool = True, decoys: int = 5, num_procs: int = 12,skip_if_exist: bool = True):
    device_type = 'cuda'
    device = torch.device(device_type)
    model_dir = "trained_models/ensemble_abresnet"
    model_files = list(glob(os.path.join(model_dir, "*.pt")))
    if len(model_files) == 0:
        exit("No model files found at: {}".format(model_dir))

    model = ModelEnsemble(model_files=model_files,
                          load_model=load_model,
                          eval_mode=True,
                          device=device)

    init_string = "-mute all -check_cdr_chainbreaks false -detect_disulf true"
    target="pred"
    pyrosetta.init(init_string)
    for fasta_file in fasta_files:
        result_dir = predictions_dir / fasta_file.stem
        if result_dir.exists():
            if skip_if_exist:
                prog_print(f'{result_dir.as_posix()} ALREADY EXIST, skipping!')
                continue
            else:
                prog_print(f'{result_dir.as_posix()} ALREADY EXIST, REWRITING!')
                result_dir.rmdir()
        prog_print("Generating constraints")
        cst_defs = get_cst_defs(model, fasta_file.as_posix(), device=device)
        if decoys > 0:
            pred_pdb = build_structure(model,
                                       fasta_file.as_posix(),
                                       cst_defs,
                                       result_dir.as_posix(),
                                       target=target,
                                       num_decoys=decoys,
                                       num_procs=num_procs,
                                       single_chain=single_chain,
                                       device=device)
            prog_print(f"prediction for {result_dir.name} successfully done!")

def cli():
    desc = ('''
        Script for predicting antibody Fv structures from heavy and light chain sequences.
        ''')
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("--name",
                        type=str,
                        default="test",
                        help="name of the folder inside data/")
    parser.add_argument("--single_chain",
                        default=False,
                        action="store_true",
                        help="If we consider only single chain")
    args = parser.parse_args()
    name = args.name
    single_chain: bool = args.single_chain
    project_path = Path(os.path.abspath(os.path.join(deepab.__file__, "../..")))
    predictions: Path = project_path / "predictions"
    predictions.mkdir(exist_ok=True)
    project_predictions = predictions / name
    prog_print(f'projects predictions are {project_predictions.as_posix()}')
    project_predictions.mkdir(exist_ok=True)
    input: Path = project_path / "data" / name
    input.mkdir(exist_ok=True)
    fastas = with_ext(input, "fasta")
    prog_print(f'inputs are: {fastas.map(lambda f: f.as_posix())}')
    run_many(fastas, predictions_dir=project_predictions, single_chain = single_chain)

if __name__ == '__main__':
    cli()
