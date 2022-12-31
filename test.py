import torch

if __name__ == "__main__":
    x = torch.tensor([0.5])
    print(torch.sqrt(1 / x - 1))