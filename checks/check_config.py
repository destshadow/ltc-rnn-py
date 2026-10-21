from ltc.config import LTCConfig


def main():
    config = LTCConfig(input_size=3, hidden_size=8)

    print("Configurazione:", config)
    print("Valori per ogni ingresso:", config.input_size)
    print("Neuroni interni:", config.hidden_size)
    print("Passi interni:", config.substeps)

    try:
        LTCConfig(input_size=3, hidden_size=0)
    except ValueError as error:
        print("Configurazione non valida rifiutata:", error)
    else:
        raise AssertionError("hidden_size=0 doveva essere rifiutato.")


if __name__ == "__main__":
    main()