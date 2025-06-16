import  torch

from validity import DQN

input_dim = 7
output_dim = 5
model = DQN(input_dim, output_dim)
model.load_state_dict(torch.load("validity.pth"))
model.eval()

with open("weights.txt", "w") as f:
    f.write("Model weights for 1. Validity Task:\n")
    for name, param in model.named_parameters():
        f.write(f"\n\nLayer: {name}\n")
        f.write(f"Shape: {param.shape}\n")
        f.write(f"Values:\n{param.data}")