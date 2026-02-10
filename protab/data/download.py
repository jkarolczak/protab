import click

from protab.data.named_data import (download,
                                    TNamedData)


@click.command()
@click.argument("names", nargs=-1, type=click.Choice(TNamedData.__args__))
def main(names: list[TNamedData] | None) -> None:
    if len(names) == 0:
        names = list(TNamedData.__args__)

    for name in names:
        download(name)


if __name__ == "__main__":
    main()
