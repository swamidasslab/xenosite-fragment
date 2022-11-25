from xenosite.fragment import FragmentNetworkX, RingFragmentNetworkX, rdkit_warnings

# from tqdm.rich import tqdm  # type: ignore
import rich.progress
import rich.panel
from rich import print
import rich
import ray
import typer
import itertools
import ast
import csv
from rich.logging import RichHandler
from typing import Optional, Sequence, Any, Callable
import logging
from rdkit import Chem
from pathlib import Path
from enum import Enum


runtime_env = {
    "pip": ["numba", "rdkit", "networkx"],
    "py_modules": ["xenosite"],
    "exclude": ["*.csv", ".git", "*.ipynb", "*.gz"],
    "eager_install": True,
}


logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[
        RichHandler(rich_tracebacks=True, tracebacks_suppress=[typer], markup=True)
    ],
)

logger = logging.getLogger("xenosite.fragment")
logger.setLevel(logging.INFO)

app = typer.Typer(
    name="fragment",
    help="Utilities for computing molecular fragments and fragment networks.",
)


class FragmentType(str, Enum):
    RING = "ring"
    ATOM = "atom"


@app.command()
def network(
    input: Path = typer.Argument(Path("bioactivation_dataset.csv")),
    output: Path = typer.Argument(Path("network.pkl.gz")),
    max_size: int = typer.Option(7, min=1, max=50, help="Maximum nodes of fragments."),
    network_type: FragmentType = typer.Option(
        "ring",
        "-t",
        "--type",
        help="Collapse rings into individual nodes before enumerating fragments.",
    ),
    filter_unmarked: bool = typer.Option(
        True, help="Remove fragments that never contain marked atoms."
    ),
    directory: Optional[Path] = typer.Option(
        None,
        envvar="DATA",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
        help="Directory as base for INPUT and OUTPUT paths.",
    ),
    concurrent: int = typer.Option(100, help="Maximum number of inflight jobs."),
):

    NetworkClass = (
        RingFragmentNetworkX if network_type == FragmentType.RING else FragmentNetworkX
    )

    rdkit_warnings(log=False)

    if directory:
        input = directory / input
        output = directory / output

    logger.info(f"[bold blue]reading {input}[/bold blue]")
    data = read_data(input)

    if output.exists():
        logger.warning(
            f"[bold red]output file exists and will be overwritten {output}[/bold red]",
        )

    logger.info(f"[bold blue]building network[/bold blue]")

    ray.init(runtime_env=runtime_env)

    result = NetworkClass(max_size=max_size)

    for frag_graph in ray_apply_batched(
        data,
        lambda batch: frag_network_batch.remote(batch, max_size, network_type),  # type: ignore
        concurrent=concurrent,
    ):
        result.update(frag_graph)  # type: ignore
        del frag_graph

    if filter_unmarked:

        logger.info(f"[bold blue]filtering unmarked fragments[/bold blue]")
        unmarked = [
            frag
            for frag in result.network
            if not result.network.nodes[frag]["marked_count"]
        ]
        result.network.remove_nodes_from(unmarked)

    logger.info(f"[bold blue]saving network to {output}[/bold blue]")
    result.save(str(output))


@ray.remote(max_calls=100, num_cpus=1)
def frag_network_batch(smiles_rids: list, max_size: int, network_type: str):
    result = None
    for smiles, rids in smiles_rids:
        N = one_frag_network(smiles, rids, max_size, network_type)
        if not result:
            result = N
        else:
            result.update(N)

    return result, len(smiles_rids)


def one_frag_network(smiles: str, rids, max_size: int, network_type: str):
    NetworkClass = (
        RingFragmentNetworkX if network_type == FragmentType.RING else FragmentNetworkX
    )

    try:
        mol = Chem.MolFromSmiles(smiles)  # type: ignore
        if not mol:
            return NetworkClass(max_size=max_size)
        if mol.GetNumAtoms() < 3:
            return NetworkClass(max_size=max_size)

        return NetworkClass(smiles, marked=rids, max_size=max_size)
    except Exception:
        logger.warning(f"Failed on: {smiles} {rids}")
        return NetworkClass(max_size=max_size)


def chunks(seq, chunk_size):
    it = iter(seq)
    while True:
        chunk = tuple(itertools.islice(it, chunk_size))
        if not chunk:
            return
        yield chunk


def ray_apply_batched(
    seq: Sequence[Any],
    launch_job: Callable[[Sequence], Any],
    concurrent: int = 100,
    chunk_size: int = 10,
):
    result_refs = []
    ready = []

    with rich.progress.Progress(
        rich.progress.SpinnerColumn(),
        *rich.progress.Progress.get_default_columns(),
        rich.progress.TimeElapsedColumn(),
        rich.progress.MofNCompleteColumn(),
    ) as pbar:
        task = pbar.add_task("Building", total=len(seq))

        for chunk in chunks(seq, chunk_size):
            if len(result_refs) >= concurrent:
                ready, result_refs = ray.wait(result_refs, fetch_local=True)

            result_refs.append(launch_job(chunk))

            while ready:
                r = ready.pop()
                r = ray.get(r)
                pbar.update(task, advance=r[1])
                yield r[0]

        while result_refs:
            ready, result_refs = ray.wait(result_refs, fetch_local=True)

            while ready:
                r = ready.pop()
                r = ray.get(r)
                pbar.update(task, advance=r[1])
                yield r[0]

        assert len(result_refs) == 0


def read_data(filename: Path):
    data = []
    with rich.progress.open(filename, "rt") as f:
        reader = csv.reader(f, dialect="excel-tab")

        for R in reader:
            smi, sor = R
            sor = ast.literal_eval(sor)
            sor = {s[0] for s in sor}

            smi = smi.split(".")[0]
            data.append((smi, sor))

    return data


if __name__ == "__main__":
    app()
