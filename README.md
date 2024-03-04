[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)

# Griffin

## install
`$ pip install griffin-torch`


## usage
```python
import torch
from griffin_torch.main import Griffin

# Forward pass
x = torch.randint(0, 100, (1, 10))

# Model
model = Griffin(
    dim=512,  # Dimension of the model
    num_tokens=100,  # Number of tokens in the input
    seq_len=10,  # Length of the input sequence
    depth=8,  # Number of transformer blocks
    mlp_mult=4,  # Multiplier for the hidden dimension in the MLPs
    dropout=0.1,  # Dropout rate
)

# Forward pass
y = model(x)

print(y)

```



# License
MIT
