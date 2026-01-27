import click

from protab.data.named_data import (TNamedData, download)


@click.command()
@click.argument("names", type=click.UNPROCESSED, required=False)
def main(names: list[TNamedData] | None) -> None:
    if names is None:
        names = list(TNamedData.__args__)

    for name in names:
        download(name)


if __name__ == "__main__":
    main()
