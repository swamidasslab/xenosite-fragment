from xenosite.fragment import FragmentNetworkX, RingFragmentNetworkX
from tqdm import tqdm  # type: ignore
import ray
import typer
import ast
import csv
import os
import os.path
from typing import Optional
import logging
from rdkit import Chem

logger = logging.getLogger("xenosite.fragment")
logger.setLevel(logging.INFO)

DATA = os.environ.get("DATA", None)

app = typer.Typer()


@ray.remote(max_calls=200, num_cpus=1)
def one_frag_network(smiles, rids, max_size: int, ring: bool):

    NetworkClass = RingFragmentNetworkX if ring else FragmentNetworkX

    try:
        mol = Chem.MolFromSmiles(smiles)  # type: ignore
        if not mol:
            return NetworkClass(max_size=max_size)
        if mol.GetNumAtoms() < 3:
            return NetworkClass(max_size=max_size)

        return NetworkClass(smiles, marked=rids, max_size=max_size)
    except Exception:
        print("Failed on:", smiles, rids)
        return NetworkClass(max_size=max_size)


def ray_apply(seq, launch_job, concurrent):
    result_refs = []
    ready = []

    with tqdm(total=len(seq), ncols=65) as pbar:

        for s in seq:
            if len(result_refs) >= concurrent:
                ready, result_refs = ray.wait(result_refs, fetch_local=True)

            result_refs.append(launch_job(s))

            while ready:
                pbar.update(1)
                yield ready.pop()

        while result_refs:
            ready, result_refs = ray.wait(result_refs, fetch_local=True)

            while ready:
                pbar.update(1)
                yield ready.pop()

        assert len(result_refs) == 0


def read_data(filename: str):
    data = []
    with open(filename) as f:
        reader = csv.reader(f, dialect="excel-tab")

        for R in reader:
            smi, sor = R
            sor = ast.literal_eval(sor)
            sor = {s[0] for s in sor}

            smi = smi.split(".")[0]
            data.append((smi, sor))

    return data


@app.command()
def main(
    input: str = typer.Argument("bioactivation_dataset.csv"),
    output: str = typer.Argument("network.pkl.gz"),
    max_size: int = 12,
    directory: Optional[str] = DATA,
    concurrent: int = 100,
    ring: bool = False,
    filter_unmarked: bool = True,
):

    NetworkClass = RingFragmentNetworkX if ring else FragmentNetworkX

    if directory:
        input = os.path.join(directory, input)
        output = os.path.join(directory, output)

    logger.info(f" reading data from {input}")
    data = read_data(input)
    result = NetworkClass(max_size=max_size)

    logger.info(f" building network")

    ray.init(address="auto", runtime_env=runtime_env)

    for frag_graph in ray_apply(
        data,
        lambda d: one_frag_network.remote(d[0], d[1], max_size, ring),
        concurrent=concurrent,
    ):
        R = ray.get(frag_graph)
        result.update(R)
        del R

    if filter_unmarked:

        logger.info(f" filtering unmarked fragments")
        unmarked = [
            frag
            for frag in result.network
            if not result.network.nodes[frag]["marked_count"]
        ]
        result.network.remove_nodes_from(unmarked)

    logger.info(f" saving network to {output}")
    result.save(output)


runtime_env = {
    "pip": ["numba", "rdkit", "networkx"],
    "py_modules": ["xenosite"],
    "exclude": ["*.csv", ".git", "*.ipynb", "*.gz"],
    "eager_install": True,
}


if __name__ == "__main__":
    app()
