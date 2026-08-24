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