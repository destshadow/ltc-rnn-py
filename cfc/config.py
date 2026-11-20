from dataclasses import dataclass


# Configurazione immutabile della cella CfC (Hasani et al., 2022).
@dataclass(frozen=True)
class CfcConfig:
    input_size: int
    hidden_size: int
    backbone_units: int = 64
    backbone_layers: int = 1

    def __post_init__(self):
        for name in ("input_size", "hidden_size", "backbone_units", "backbone_layers"):
            value = getattr(self, name)
            if type(value) is not int:
                raise TypeError(f"{name} deve essere un numero intero.")
            if value <= 0:
                raise ValueError(f"{name} deve essere maggiore di zero.")
