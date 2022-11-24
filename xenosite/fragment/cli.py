from xenosite.fragment import FragmentNetworkX, RingFragmentNetworkX
from tqdm import tqdm # type: ignore
import ray
import typer
import ast
import csv
import os
import os.path
from typing import Optional

DATA = os.environ.get("DATA", None)

app = typer.Typer()

@ray.remote(max_calls=200)
def one_frag_network(smiles, rids, max_size : int, ring : bool):

  NetworkClass = RingFragmentNetworkX if ring else FragmentNetworkX

  try:
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
  max_size : int = 12, 
  directory : Optional[str] = DATA,
  concurrent : int = 30,
  ring : bool = False):

  NetworkClass = RingFragmentNetworkX if ring else FragmentNetworkX

  ray.init(runtime_env=runtime_env)

  if directory:
    input = os.path.join(directory, input)
    output = os.path.join(directory, output)

  data = read_data(input)
  result = NetworkClass(max_size=max_size)

  for frag_graph in ray_apply(
    data,
    lambda d: one_frag_network.remote(d[0], d[1], max_size, ring), 
    concurrent=concurrent
  ):
    R = ray.get(frag_graph)
    result.update(R)
    del R
  
  result.save(output)

runtime_env = {
  "pip": ["numba", "rdkit", "networkx"],
  "working_dir": "./",
  "exclude": ["*.csv", ".git", "*.ipynb", "*.gz"],
  "eager_install": True
}


if __name__ == "__main__":
  app()
