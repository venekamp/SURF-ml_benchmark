import click


@click.command()
@click.option("--abc")
def cli(abc) -> None:
    click.echo(f"{abc}")
