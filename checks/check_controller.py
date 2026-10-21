import torch

from ltc.config import LTCConfig
from models.continuous_controller import ContinuousController


def check_device(device):
    torch.manual_seed(42)

    config = LTCConfig(
        input_size=3,
        hidden_size=8,
        substeps=6,
    )
    model = ContinuousController(config, output_size=1).to(device)

    # Campioni sintetici: non rappresentano ancora sensori del drone.
    samples = torch.randn(2, 5, 3, device=device)

    initial = model.initial_state(batch_size=2)
    initial_copy = initial.clone()
    state = initial
    outputs = []

    for index in range(samples.shape[1]):
        commands, state = model(
            samples[:, index],
            state,
            dt=0.1,
        )

        assert commands.shape == (2, 1)
        assert state.shape == (2, 8)
        assert torch.isfinite(commands).all().item()
        assert torch.isfinite(state).all().item()
        assert ((commands >= 0) & (commands <= 1)).all().item()

        outputs.append(commands)

    torch.testing.assert_close(initial, initial_copy)
    expected = torch.stack(outputs, dim=1).detach()

    # Ripartire dallo stato nullo deve riprodurre le stesse uscite.
    with torch.no_grad():
        replay_state = model.initial_state(batch_size=2)
        replay = []

        for index in range(samples.shape[1]):
            commands, replay_state = model(
                samples[:, index],
                replay_state,
                dt=0.1,
            )
            replay.append(commands)

        torch.testing.assert_close(
            torch.stack(replay, dim=1),
            expected,
        )

        # Ogni sequenza da sola deve dare lo stesso risultato del batch.
        for row in range(samples.shape[0]):
            single_state = model.initial_state(batch_size=1)
            single_outputs = []

            for index in range(samples.shape[1]):
                commands, single_state = model(
                    samples[row:row + 1, index],
                    single_state,
                    dt=0.1,
                )
                single_outputs.append(commands)

            torch.testing.assert_close(
                torch.stack(single_outputs, dim=1),
                expected[row:row + 1],
                rtol=1e-4,
                atol=1e-6,
            )

    # Un obiettivo artificiale serve solo a verificare il backward.
    loss = (outputs[-1] - 0.8).square().mean()
    loss.backward()

    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, f"Gradiente assente: {name}"
        assert torch.isfinite(parameter.grad).all().item(), (
            f"Gradiente non finito: {name}"
        )

    for name, component in (
        ("LTC", model.cell),
        ("uscita", model.readout),
    ):
        gradient_sum = sum(
            parameter.grad.abs().sum().item()
            for parameter in component.parameters()
        )
        assert gradient_sum > 0, f"Gradienti tutti nulli: {name}"

    print(f"{device}: forme, limiti e stati finiti verificati.")
    print("Stato iniziale non modificato.")
    print("Ripartenza e indipendenza del batch verificate.")
    print("Gradienti nella LTC e nello strato di uscita verificati.")


def main():
    check_device(torch.device("cpu"))

    if torch.cuda.is_available():
        check_device(torch.device("cuda"))

    print("Controlli superati.")


if __name__ == "__main__":
    main()