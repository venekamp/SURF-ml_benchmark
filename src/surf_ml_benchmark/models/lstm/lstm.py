from typing import Any
import click


from surf_ml_benchmark.model_commands import ModelCommands
from surf_ml_benchmark.common import get_model_help_text


class LstmCommands(ModelCommands):
    def __init__(self, model: str, **kwargs: Any) -> None:
        super().__init__(model, **kwargs)
        self.help = get_model_help_text(model_name="LSTM")


# cli: LstmCommands = LstmCommands(help="Run LSTM models based on different frameworks.")

if __name__ == "__main__":
    click.echo(message="This script cannot be executed on its own.")
