# DoPFN setup for the RCT_Bench eval

DoPFN is not a pip package; it needs the upstream repo checked out with its
bundled artifacts, and it runs on GPU. Setup used for `rct_bench_dopfn.py`:

1. Clone the project's cited Do-PFN source:
   ```
   git clone https://github.com/jr2021/Do-PFN.git
   ```
   The model weights ship in `artifacts/` (`dopfn_model.pkl`, the `.cpkt`).

2. One-line torch-compat patch (torch 2.11 no longer re-exports `Optional`/`Tensor`
   and the nn classes from `torch.nn.modules.transformer`). In `model/layer.py`,
   replace the `from torch.nn.modules.transformer import ( ... )` block with:
   ```python
   from typing import Optional
   from torch import Tensor
   from torch.nn import Module, MultiheadAttention, Linear, Dropout, LayerNorm
   from torch.nn.modules.transformer import _get_activation_fn
   ```
   Import-only; no functional change.

3. Run (from anywhere; the script chdirs into the repo for DoPFN's relative
   artifact paths):
   ```
   python -m causal_probe.rct_bench_dopfn \
     --dopfn-repo /path/to/Do-PFN \
     --data-dir  /abs/path/to/data/RCT_Bench/cleaned_data \
     --out-json  causal_probe/results/rct_bench_dopfn.json --device cuda
   ```

Ran on a single RTX 4090. DoPFN completes on 123/124 trials but underestimates
effect magnitude (median |d| 0.18 vs RCT 0.35); sign 79%, corr 0.73.
