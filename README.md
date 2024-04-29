# ProteinMPNN Wrapper
A thin wrapper to enhance developer-friendliness of the original [ProteinMPNN](https://github.com/dauparas/ProteinMPNN) repository by [Justas Dauparas](https://github.com/dauparas).

Clone and install the repository via
```
git clone --recurse-submodules https://github.com/Croydon-Brixton/proteinmpnn_wrapper.git
cd proteinmpnn_wrapper
pip install .
```

Use via
```python
import numpy as np
import torch
import proteinmpnn
from proteinmpnn.run import load_protein_mpnn_model
from proteinmpnn.data import BackboneSample

DEVICE = "cpu"  # set to `cuda` if you want to use the GPU

# load protein mpnn model & weights
model = load_protein_mpnn_model(model_type="ca", device=DEVICE)

# create a dummy backbone structure
backbone = BackboneSample(
    bb_coords=np.random.rand(10, 3), 
    ca_only=True, 
    res_name="MXXXACXGXX", 
    res_mask=np.array([0, 1, 1, 1, 0, 0, 1, 0, 1, 1]),  # 0 = fixed, 1 = will be sampled
)

# sample a sequence for the random structure
sample = model.sample(
    randn=torch.randn(1, backbone.n_residues, device=DEVICE), 
    **backbone.to_protein_mpnn_input("sampling", device=DEVICE)
)
```

See [the example notebook](./example.ipynb) for more examples.
