import time

import torch
from pytorch_lightning import Callback


class CUDACallback(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        # Reset the memory use counter
        torch.cuda.reset_peak_memory_stats(trainer.root_gpu)
        torch.cuda.synchronize(trainer.root_gpu)
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module, outputs):
        torch.cuda.synchronize(trainer.root_gpu)
        max_memory = torch.cuda.max_memory_allocated(trainer.root_gpu) / 2 ** 20
        epoch_time = time.time() - self.start_time

        max_memory = torch.tensor(max_memory, dtype=torch.int, device=trainer.root_gpu)
        epoch_time = torch.tensor(epoch_time, dtype=torch.int, device=trainer.root_gpu)

        torch.distributed.all_reduce(max_memory, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(epoch_time, op=torch.distributed.ReduceOp.SUM)

        world_size = torch.distributed.get_world_size()

        print(f"Average Epoch time: {epoch_time.item() / float(world_size):.2f} seconds")
        print(f"Average Peak memory {max_memory.item() / float(world_size):.2f}MiB")
