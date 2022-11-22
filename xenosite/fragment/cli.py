from xenosite.fragment import FragmentNetworkX
from tqdm import tqdm
import ray
import typer
import ast
import csv
import gc

app = typer.Typer()

@ray.remote(max_calls=200)
def one_frag_network(smiles, rids, max_size):
  try:
    return FragmentNetworkX(smiles, marked=rids, max_size=max_size)
  except Exception:
    print("Failed on:", smiles, rids)
    return FragmentNetworkX(max_size=max_size)
  

def ray_apply(seq, launch_job, concurrent):
  result_refs = []
  ready = []

  with tqdm(total=len(seq)) as pbar:

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
