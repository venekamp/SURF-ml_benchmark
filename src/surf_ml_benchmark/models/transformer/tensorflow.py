import click


@click.group()
@click.option("--b")
def cli() -> None:
    pass
