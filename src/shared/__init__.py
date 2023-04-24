import git
from typing import Final
import pathlib
from .error import relative_tensor_error

repo_basepath: Final[pathlib.Path] = pathlib.Path(
    git.Repo(".", search_parent_directories=True).working_dir
)