import torch
from griffin_torch.main import Griffin

# Forward pass
x = torch.randint(0, 10000, (1, 1000))  # Increase the number of tokens

# Model
model = Griffin(
    dim=2048,  # Increase the dimension of the model
    num_tokens=10000,  # Increase the number of tokens in the input
    seq_len=1000,  # Increase the length of the input sequence
    depth=32,  # Increase the number of transformer blocks
    mlp_mult=16,  # Increase the multiplier for the hidden dimension in the MLPs
    dropout=0.1,  # Dropout rate
)

# Forward pass
y = model(x)

print(y.shape)