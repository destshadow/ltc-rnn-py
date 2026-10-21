import torch


def main():
    print("Versione PyTorch:", torch.__version__)
    print("CUDA di PyTorch:", torch.version.cuda)
    print("GPU disponibile:", torch.cuda.is_available())

    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch non riesce ad accedere alla GPU.")

    print("Nome GPU:", torch.cuda.get_device_name(0))

    values = torch.tensor([1.0, 2.0, 3.0], device="cuda")
    result = values * 2

    print("Dispositivo del risultato:", result.device)
    print("Risultato:", result.cpu().tolist())


if __name__ == "__main__":
    main()