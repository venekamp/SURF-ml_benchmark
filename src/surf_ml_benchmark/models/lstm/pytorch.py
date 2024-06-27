import click


@click.command()
@click.option("--xyz")
def cli(xyz) -> None:
    click.echo(f"{xyz}")
