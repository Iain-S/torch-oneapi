"""A simple, linear example with added DDP."""
import os
import warnings

with warnings.catch_warnings():
    # Silence the torchvision warning.
    warnings.simplefilter("ignore")
    import torch
    import torch.nn as nn
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.distributed import init_process_group

import intel_extension_for_pytorch as ipex  # has side-effects
import oneccl_bindings_for_pytorch  # has side-effects

from sklearn.model_selection import train_test_split

# E.g. "xpu" or "cpu".
device_name = os.getenv("TORCH_DEVICE")
assert device_name, "TORCH_DEVICE env var should be set."
device = torch.device(device_name)
print(device)

# Setting up environment variables.
os.environ['RANK'] = str(os.environ.get('PMI_RANK', 0))
os.environ['WORLD_SIZE'] = str(os.environ.get('PMI_SIZE', 1))
os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '29500'

# Required by DDP.
init_process_group(
        backend="ccl",
        )

# Build the model.
model1_reg = nn.Sequential(
    nn.Linear(in_features=1, out_features=10),
    nn.Linear(in_features=10, out_features=10),
    nn.Linear(in_features=10, out_features=1),
).to(device)

torch.manual_seed(4)

model1_reg = DDP(model1_reg)

optim_reg = torch.optim.SGD(model1_reg.parameters(), lr=0.1)

# try
model1_reg, optim_reg = ipex.optimize(model=model1_reg, optimizer=optim_reg)

loss_fn_reg = nn.L1Loss()

# Create some data.
weight = 0.7
bias = 0.3
start = 0
stop = 1
step = 0.01

x_reg = torch.arange(start, stop, step).unsqueeze(dim=1)
y_reg = x_reg * weight + bias

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(x_reg, y_reg)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = (
    X_train_reg.to(device),
    X_test_reg.to(device),
    y_train_reg.to(device),
    y_test_reg.to(device),
)

# Too many epochs may result in file descriptor issues.
EPOCHS = 2
for epoch in range(EPOCHS):
    model1_reg.train()

    y_pred_reg = model1_reg(X_train_reg)

    loss_reg = loss_fn_reg(y_pred_reg, y_train_reg)

    optim_reg.zero_grad()

    loss_reg.backward()

    optim_reg.step()

    if epoch % 100 == 0:
        print(loss_reg)
