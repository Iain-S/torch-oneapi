# torch-oneapi

Utils, tests and docs for using Intel oneAPI and PyTorch.

## Files

### my_ddp.py

A simple linear regression DNN used to check whether we can run DDP on XPUs.

```bash
CCL_ZE_IPC_EXCHANGE=sockets TORCH_DEVICE=xpu mpirun -n 2 python my_ddp.py
```
