import importlib
from typing import Any
import click
import os
from pathlib import Path

from click.core import Command, Context


models_folder: str = os.path.join(os.path.dirname(__file__), "models")


class MlBenchmark(click.MultiCommand):
    def list_commands(self, ctx: Context) -> list[str]:
        parent_path: Path = Path(models_folder)
        directories: list[Path] = [
            p for p in parent_path.iterdir() if p.is_dir() and not p.name[0].startswith("_")
        ]
        models: list[str] = sorted([str(p.name) for p in directories])

        return models

    def get_command(self, ctx: Context, cmd_name: str) -> Command | None:
        class_name: str = f"{cmd_name.capitalize()}Commands"
        module_name: str = f"surf_ml_benchmark.models.{cmd_name}.{cmd_name}"
        module: Any = importlib.import_module(module_name)

        model_class = getattr(module, class_name)

        help_text = model_class.help
        model_instance = model_class(model=cmd_name, help=help_text)

        return model_instance


@click.command(cls=MlBenchmark)
@click.pass_context
def cli(ctx: Context) -> None:
    """
    Run benchmarks for different models types and framework implementations.

    Use one of the commands to specify which model you want to run. Each model
    takes a command of it own that defines the framework to use.
    """

    ctx.ensure_object(object_type=dict)
    ctx.obj["models_folder"] = models_folder
    pass


if __name__ == "__main__":
    cli()
