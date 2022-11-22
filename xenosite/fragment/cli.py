from xenosite.fragment import FragmentNetworkX
from tqdm import tqdm
import ray
import typer
import ast
import csv

app = typer.Typer()

@ray.remote
def one_frag_network(smiles, rids, max_size):
  try:
    return FragmentNetworkX(smiles, marked=rids, max_size=max_size)
  except Exception:
    print("Failed on:", smiles, rids)
    return FragmentNetworkX(max_size=max_size)
  

def ray_apply(seq, launch_job, concurrent):
  result_refs = []
  ready = []

  for s in tqdm(seq, "Dispatched"):
    if len(result_refs) >= concurrent:
      ready, result_refs = ray.wait(result_refs, fetch_local=True)

    result_refs.append(launch_job(s))

    while ready:
      yield ready.pop()

  for _ in tqdm(range(len(result_refs)), "Last Batch"):
    ready, result_refs = ray.wait(result_refs, fetch_local=True)

    while ready:
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
      data.append((smi, sor))

  return data


@app.command()
def build_network(
  input: str = typer.Argument("tuple_dataset.csv"), 
  output: str = typer.Argument("network.pkl.gz"), 
  max_size : int = 12, 
  concurrent : int = 30):

  ray.init(runtime_env=runtime_env)

  data = read_data(input)
  result = FragmentNetworkX(max_size=max_size)

  for frag_graph in ray_apply(
    data,
    lambda d: one_frag_network.remote(d[0], d[1], max_size), 
    concurrent=concurrent
  ):

    result.update(ray.get(frag_graph))
  
  result.save(output)

runtime_env = {
  "pip": ["numba", "rdkit", "networkx"],
  "working_dir": "./",
  "exclude": ["*.csv", ".git", "*.ipynb", "*.gz"],
  "eager_install": True
}


if __name__ == "__main__":
  app()