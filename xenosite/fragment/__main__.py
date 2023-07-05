from xenosite.fragment import FragmentNetworkX, RingFragmentNetworkX, rdkit_warnings

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
import pandas as pd
from xenosite.fragment.stats import FragmentStatistics

runtime_env = {
    "pip": ["numba", "rdkit", "networkx", "rich", "pandas"],
    "py_modules": ["xenosite"],
    "exclude": ["*.csv", ".git", "*.ipynb", "*.gz", "coverage"],
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
    verbose: int = typer.Option(0, "--verbose", "-v", count=True),
):

    NetworkClass = (
        RingFragmentNetworkX if network_type == FragmentType.RING else FragmentNetworkX
    )

    rdkit_warnings(log=False)

    if directory:
        input = directory / input
        output = directory / output

    logger.info(f"[bold blue]reading {input}[/bold blue]")
    

    if output.exists():
        logger.warning(
            f"[bold red]output file exists and will be overwritten {output}[/bold red]",
        )

    logger.info(f"[bold blue]building network[/bold blue]")

    ray_kwargs = dict(runtime_env=runtime_env)

    if not verbose:
      ray_kwargs.update(dict(
        configure_logging=True,
        logging_level=logging.WARN,
        log_to_driver=False
      ))  # type: ignore

    ray.init(**ray_kwargs)

    data = read_data(input)
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

        P = result.to_pandas()
        unmarked = list(P[P.marked_count == 0].index)
        result.network.remove_nodes_from(unmarked)

        unmarked_ref = [
            ref
            for ref, degree in result.network.out_degree
            if isinstance(ref, tuple) and degree == 0
        ]

        result.network.remove_nodes_from(unmarked_ref)

    logger.info(f"[bold blue]saving network to {output}[/bold blue]")
    result.save(str(output))



@app.command()
def smarts(
    input: Path = typer.Argument(Path("bioactivation_dataset.csv")),
    smarts: Path = typer.Argument(Path("structural_alerts.csv")),
    output: Path = typer.Argument(Path("alerts_stats.pkl.gz")),
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

    rdkit_warnings(log=False)

    if directory:
        input = directory / input
        output = directory / output
        smarts = directory / smarts

    logger.info(f"[bold blue]reading {input}[/bold blue]")
    
    if output.exists():
        logger.warning(
            f"[bold red]output file exists and will be overwritten {output}[/bold red]",
        )
    data = read_data(input)
    logger.info(f"[bold blue]building network[/bold blue]")

    ray.init(
        runtime_env=runtime_env  # , configure_logging=True, logging_level=logging.WARN
    )

    smarts_alerts = pd.read_csv(str(smarts))

    smarts = list(set(smarts_alerts["smarts"])) # type: ignore
    smarts_ref = ray.put(smarts)

    result = FragmentStatistics()

    for smarts_stats in ray_apply_batched(
        data,
        lambda batch: smarts_batch.remote(batch, smarts_ref),  # type: ignore
        concurrent=concurrent,
        chunk_size=50
    ):
        result.update(smarts_stats)  # type: ignore
        del smarts_stats


    logger.info(f"[bold blue]saving out to {output}[/bold blue]")

    import gzip
    import pickle
    with gzip.GzipFile(output, "wb") as f:
      pickle.dump(result, f)
  



@ray.remote(max_calls=100, num_cpus=1)
def smarts_batch(smiles_rids: list, smarts : Sequence[Chem.Mol]):  # type: ignore
    result = FragmentStatistics()
    
    mol_smrt = [(Chem.MolFromSmarts(s), s) for s in smarts]  # type: ignore
    for smiles, rids in smiles_rids:
        M = Chem.MolFromSmiles(smiles)  # type: ignore

        R = FragmentStatistics()
        for (mol, smrt) in mol_smrt:
          matches = M.GetSubstructMatches(mol)
          if matches:
            R.add(smrt, matches, rids, M.GetNumAtoms())
            result.update(R)

    return result, len(smiles_rids)




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
        rich.progress.TextColumn("[progress.description]{task.description}"),
        rich.progress.BarColumn(),
        rich.progress.MofNCompleteColumn(),
        rich.progress.TaskProgressColumn(),
        rich.progress.TimeRemainingColumn(),
        rich.progress.TimeElapsedColumn(),
    ) as pbar:
        task = pbar.add_task("Building...", total=len(seq))

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
