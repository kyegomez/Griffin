[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)

# Griffin
Implementation of Griffin from the paper: "Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models". [PAPER LINK](https://huggingface.co/papers/2402.19427)


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


# Citation
```
@misc{de2024griffin,
    title={Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models}, 
    author={Soham De and Samuel L. Smith and Anushan Fernando and Aleksandar Botev and George Cristian-Muraru and Albert Gu and Ruba Haroun and Leonard Berrada and Yutian Chen and Srivatsan Srinivasan and Guillaume Desjardins and Arnaud Doucet and David Budden and Yee Whye Teh and Razvan Pascanu and Nando De Freitas and Caglar Gulcehre},
    year={2024},
    eprint={2402.19427},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```