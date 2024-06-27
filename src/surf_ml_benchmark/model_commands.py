from types import CodeType
from typing import Any
import click
from pathlib import Path

from click.core import Command, Context


class ModelCommands(click.MultiCommand):
    """hdgdgdg"""

    def __init__(self, model: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.model = model
        if "help" in kwargs:
            self._help = kwargs["help"]

    @property
    def help(self) -> str:
        return self._help

    @help.setter
    def help(self, value: str) -> None:
        self._help = value

    def list_commands(self, ctx: Context) -> list[str]:
        """Get all the supported frameworks for the a model."""
        models_folder: str = f"{ctx.obj["models_folder"]}/{self.model}"
        model: str = f"{models_folder.split('/')[-1]}.py"
        parent_path: Path = Path(models_folder)

        # The __init__.py file is not a framework. Neither is the model name
        # irself. The model name is derived from the current folder.
        frameworks: list[str] = [
            str(p.stem)
            for p in parent_path.iterdir()
            if p.is_file() and p.suffix == ".py" and p.name != "__init__.py" and p.name != model
        ]

        frameworks.sort()

        return frameworks

    def get_command(self, ctx: Context, cmd_name: str) -> Command | None:
        """
        Given the cmd_name, load the corresonding Python file for the actual
        benchmark implementation. The Python file must be: <cmd_namm>.py and
        it must implement a cli() function. This function must be a click
        command.
        """
        models_folder: str = ctx.obj["models_folder"]
        ns: dict[str, Any] = {}
        fn: str = f"{models_folder}/{self.model}/{cmd_name}.py"
        with open(file=fn) as f:
            code: CodeType = compile(source=f.read(), filename=fn, mode="exec")
            eval(code, ns, ns)

        return ns["cli"]
