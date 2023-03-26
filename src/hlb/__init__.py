import git
from typing import Final
import pathlib

repo_basepath: Final[pathlib.Path] = pathlib.Path(
    git.Repo(".", search_parent_directories=True).working_dir
)