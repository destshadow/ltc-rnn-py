from dataclasses import dataclass


@dataclass(frozen=True) #@dataclass genera automaticamente il costruttore. Possiamo quindi scrivere: config = LTCConfig(input_size=3, hidden_size=8); frozen=True impedisce di modificare accidentalmente la configurazione dopo averla creata


class LTCConfig:
    input_size: int
    hidden_size: int
    substeps: int = 6

    def __post_init__(self):
        fields = {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "substeps": self.substeps,
        }

        for name, value in fields.items():
            if type(value) is not int:
                raise TypeError(f"{name} deve essere un numero intero.")

            if value <= 0:
                raise ValueError(f"{name} deve essere maggiore di zero.")