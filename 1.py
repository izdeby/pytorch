import torch

device = "cuda"
x = torch.tensor([True, True, False, False], device=device)
mask = torch.tensor([0, 0, 1, 1], dtype=torch.bool, device=device)
res = x.masked_fill(mask, 1)
self.assertEqual(res, torch.tensor([True, True, True, True], device=device))
#. ~/.bashrc
