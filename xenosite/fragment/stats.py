from collections import defaultdict
from .graph import Graph
import numpy as np
from functools import reduce
from scipy.stats import hypergeom
import pandas as pd
from typing import Optional


class FragmentStatistics:
    def __init__(self):
        self._stats: dict = defaultdict(list)
        self._lookup: dict[str, int] = {}

    def add(
        self, frag: str, ids: list[list[int]], marked: set[int], mol_atoms: int
    ) -> None:
        assert frag not in self._lookup
        assert self._stats is not None

        S = dict()

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

        S["count"] = len(covered) / size
        S["marked_count"] = marked_count
        S["n_mol"] = 1
        S["n_atom"] = mol_atoms
        S["n_mark"] = len(marked)
        S["n_cover"] = len(covered)
        S["n_mark_cover"] = len(covered & marked)
        S["exp"] = exp
        S["obs"] = obs
        S["marked_ids"] = marked_ids

        self.append_one(frag, **S)

    def copy_from(self, other : "FragmentStatistics") -> "FragmentStatistics":
      k1 = set(self._stats['frag'])
      k2 = set(other._stats['frag'])

      out = FragmentStatistics()

      for k in k1 & k2:
        n = other._lookup[k]
        d = {k : other._stats[k][n] for k in other._stats}
        out.append_one(**d)
      
      return out

    def append_one(self, frag, **kwargs):
        assert frag not in self._lookup

        if self._stats: assert set(kwargs) | {"frag"} == set(self._stats), \
            "must always call FragmentStatistics.append_one with the same arguments."

        self._lookup[frag] = len(self._stats["frag"])
        self._stats["frag"].append(frag)

        for k in kwargs:
            self._stats[k].append(kwargs[k])

    def update_one(self, frag, **kwargs):
        if self._stats:
          assert set(kwargs) | {"frag"} == set(self._stats), \
            "must always call FragmentStatistics.update_one with the same arguments."

        if frag in self._lookup:
            n = self._lookup[frag]
            for k in kwargs:
                self._stats[k][n] = self._stats[k][n] + kwargs[k]
        else:
            self.append_one(frag, **kwargs)

    def to_pandas(self) -> pd.DataFrame:
        return pd.DataFrame(self._stats).set_index("frag")

    def update(self, other: "FragmentStatistics"):
        if self._stats and other._stats:
          assert set(other._stats) == set(self._stats), \
            "These FragmentStatistics instances have different statistics!"
            
        other_stats = list(other._stats.items())

        keys = [x[0] for x in other_stats]
        lists = [x[1] for x in other_stats]

        for data in zip(*lists):
            S = dict(zip(keys, data))
            self.update_one(**S)
