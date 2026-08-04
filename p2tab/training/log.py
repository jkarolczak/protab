from dataclasses import dataclass


@dataclass
class WandbConfig:
    project: str = "ProTab"
    entity: str = "jacek-karolczak"
    active: bool = False
