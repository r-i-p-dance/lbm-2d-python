import re
from pathlib import Path


def get_safe_filepath(filepath_str, create_parents=True):
    """Return a path that does not yet exist, by appending/incrementing _N.

    'run.gif' -> 'run.gif' if free, else 'run_1.gif', 'run_2.gif', ...

    Guards long simulations against silently overwriting a previous result.

    create_parents: make the containing directory if it is missing, so a
    run never dies at the final write because 'results/' does not exist.
    """
    path = Path(filepath_str)

    if create_parents:
        path.parent.mkdir(parents=True, exist_ok=True)

    while path.exists():
        stem = path.stem
        suffix = path.suffix

        match = re.search(r'_(\d+)$', stem)
        if match:
            current_num = int(match.group(1))
            base_name = stem[:match.start()]
            new_stem = f"{base_name}_{current_num + 1}"
        else:
            new_stem = f"{stem}_1"

        path = path.with_name(new_stem + suffix)

    return str(path)


class RunPaths:
    """Resolve one free run name, then derive every artifact path from it.

    Note: The animation is now an MP4 instead of a GIF.


    Single source of truth for output naming: the animation, still frame,
    run log, field dump and arrays all share one stem, so a run's artifacts
    can never drift apart. Resolving each independently would allow the GIF
    to land on _2 while the log lands on _3.

        results/animations/pipe_bend_2.gif
        results/plots/pipe_bend_2.jpg
        results/logs/pipe_bend_2.txt
        results/logs/pipe_bend_2_fields.txt
        results/arrays/pipe_bend_2.npz
    """

    def __init__(self, name, root="results"):
        root = Path(root)
        stem = name
        n = 0
        while True:
            candidates = self._candidates(root, stem)
            if not any(Path(p).exists() for p in candidates.values()):
                break
            n += 1
            stem = f"{name}_{n}"

        self.stem = stem
        self.root = root
        for key, value in self._candidates(root, stem).items():
            setattr(self, key, value)

        for d in ("animations", "plots", "logs", "arrays"):
            (root / d).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _candidates(root, stem):
        return {
            "animation": str(root / "animations" / f"{stem}.mp4"),
            "plot":      str(root / "plots"      / f"{stem}.jpg"),
            "log":       str(root / "logs"       / f"{stem}.txt"),
            "fields":    str(root / "logs"       / f"{stem}_fields.txt"),
            "arrays":    str(root / "arrays"     / f"{stem}.npz"),
        }