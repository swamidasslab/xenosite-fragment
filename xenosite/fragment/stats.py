from collections import defaultdict
from .graph import Graph
import numpy as np
from functools import reduce
from scipy.stats import hypergeom
import pandas as pd


class FragmentStatistics:
    def __init__(self):
        self._stats = defaultdict(list)
        self.seen = set()

    def add(self, frag: str, ids: list[list[int]], marked: set[int], mol_atoms: int):
        assert frag not in self.seen

        self.seen.add(frag)

        S = self._stats

        amarked = np.array(list(marked))[None, None, :]

        # normalized count of marked ids by position in fragment
        marked_ids = (np.array(ids)[:, :, None] == amarked).sum(axis=0).sum(axis=1)
        marked_ids = np.where(marked_ids > 0, 1, 0)  # type: ignore

        set_ids = [set(i) for i in ids]

        size = len(set_ids[0])

        covered = reduce(lambda x, y: x | y, set_ids)

        marked_count = (
            len(
                reduce(
                    lambda x, y: x | y,
                    [i for i in set_ids if marked & i],
                    set(),
                )
            )
            / size
        )

        # exp = probability of fragment overlapping with at least one marked atom
        # given: size of molecule, number of atoms matching fragment, number of marked atoms
        exp = 1 - hypergeom.cdf(0, mol_atoms, int(len(covered)), len(marked))
        obs = 1 if marked_count else 0

        S["frag"].append(frag)
        S["count"].append(len(covered) / size)
        S["marked_count"].append(marked_count)
        S["n_mol"].append(1)
        S["n_atom"].append(mol_atoms)
        S["n_mark"].append(len(marked))
        S["n_cover"].append(len(covered))
        S["n_mark_cover"].append(len(covered & marked))
        S["exp"].append(exp)
        S["obs"].append(obs)
        S["marked_ids"].append(marked_ids)

        return {}

    def pack(self) -> pd.DataFrame:
        if self._stats:
            self._dataframe = pd.DataFrame(self._stats).set_index("frag")
            self._stats = {}

        return self._dataframe

    def update(self, other: "FragmentStatistics"):
        other_df = other.pack()
        self_df = self.pack()

        left, right = self_df.align(other_df, join="outer", fill_value=0)

        self._dataframe = left + right  # type: ignore
        return self._dataframe
