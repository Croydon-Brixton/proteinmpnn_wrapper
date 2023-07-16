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
import proteinmpnn
from proteinmpnn.run import load_protein_mpnn_model, design_sequences, untokenise_sequence
```

See [the example notebook](./example.ipynb) for more an example.