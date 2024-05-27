"""
Common dictionarys, lists and functions.
"""
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
from torch import nn
import MinkowskiEngine as ME

from blip.utils import comm
from blip.utils.common import _reduce

activations = {
    'relu':         nn.ReLU,
    'tanh':         nn.Tanh,
    'sigmoid':      nn.Sigmoid,
    'softmax':      nn.Softmax,
    'leaky_relu':   nn.LeakyReLU,
}

sparse_activations = {
    'relu':     ME.MinkowskiReLU(),
    'prelu':    ME.MinkowskiPReLU(),
    'selu':     ME.MinkowskiSELU(),
    'celu':     ME.MinkowskiCELU(),
    'sigmoid':  ME.MinkowskiSigmoid(),
    'tanh':     ME.MinkowskiTanh(),
    'softmax':  ME.MinkowskiSoftmax(),
    'leaky_relu':   ME.MinkowskiLeakyReLU(),
}

normalizations = {
    'batch_norm',
    'bias',
}


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, input):
        return input


class _CopyToParallelRegion(torch.autograd.Function):
    """Pass the input to the parallel region."""

    @staticmethod
    def symbolic(graph, input_, comm_id_):
        """symbolic method"""
        return input_

    @staticmethod
    def forward(ctx, input_, comm_id_):
        ctx.comm_id = comm_id_
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        if comm.is_distributed(ctx.comm_id):
            return _reduce(grad_output, group=comm.get_group(ctx.comm_id)), None
        else:
            return grad_output, None


class _ReduceFromParallelRegion(torch.autograd.Function):
    """All-reduce the input from the parallel region."""

    @staticmethod
    def symbolic(graph, input_, comm_id_):  # pragma: no cover
        """symbolic method"""
        if comm.is_distributed(comm_id_):
            return _reduce(input_, group=comm.get_group(comm_id_))
        else:
            return input_

    @staticmethod
    def forward(ctx, input_, comm_id_):  # pragma: no cover
        if comm.is_distributed(comm_id_):
            return _reduce(input_, group=comm.get_group(comm_id_))
        else:
            return input_

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        return grad_output, None


# matmul parallel
def copy_to_parallel_region(input_, comm_name):  # pragma: no cover
    """Parallel copy helper"""
    return _CopyToParallelRegion.apply(input_, comm_name)


def reduce_from_parallel_region(input_, comm_name):  # pragma: no cover
    """Parallel reduction helper"""
    return _ReduceFromParallelRegion.apply(input_, comm_name)


def init_ddp_model_and_reduction_hooks(
    model,
    device_ids,
    output_device,
    bucket_cap_mb=25,
    broadcast_buffers=True,
    find_unused_parameters=False,
    gradient_as_bucket_view=True,
    static_graph=False
):
    """Early exit if we are not in a distributed setting"""
    if not dist.is_initialized():
        return model

    """Set this to false in init and then find out if we can use it"""
    need_hooks = False
    ddp_group = comm.get_group("data")

    """This is the trivial case"""
    if comm.get_size("model") == 1:
        ddp_group = None
    else:
        num_parameters_total = 0
        num_parameters_shared_model = 0
        for param in model.parameters():
            """
            if it does not have any annotation, we assume it is shared between all model ranks
            not needed here, sync_params annotates everything
            if not hasattr(param, "is_shared_mp"):
                param.is_shared_mp = ["model"]
            add the sharing type to the dict
            """
            num_parameters_total += 1
            if "model" in param.is_shared_mp:
                num_parameters_shared_model += 1

        """If all parameters are shared between all model ranks, then the situation is easy"""
        if (num_parameters_shared_model == num_parameters_total):
            ddp_group = None
            """Register some pre-multiply reduction hooks"""
            print("Setting up gradient hooks to account for shared parameter multiplicity")
            for param in model.parameters():
                param.register_hook(lambda grad: grad * float(comm.get_size("model")))
        else:
            ddp_group = comm.get_group("data")
            broadcast_buffers = False
            need_hooks = True

    model = DistributedDataParallel(
        model,
        device_ids=device_ids,
        output_device=output_device,
        bucket_cap_mb=bucket_cap_mb,
        broadcast_buffers=broadcast_buffers,
        find_unused_parameters=find_unused_parameters,
        gradient_as_bucket_view=gradient_as_bucket_view,
        static_graph=static_graph,
        process_group=ddp_group
    )
    if not need_hooks:
        return model

    """Define comm hook"""
    def reduction_comm_hook(state: object, bucket: dist.GradBucket) -> torch.futures.Future[torch.Tensor]:
        """Allreduce everything first"""
        buff = bucket.buffer()
        fut = dist.all_reduce(buff, op=dist.ReduceOp.AVG, group=comm.get_group("data"), async_op=True).get_future()
        params = bucket.parameters()

        def grad_reduction(fut, grads, group):
            """Reduce remaining gradients"""
            coalesced = _flatten_dense_tensors(grads)
            dist.all_reduce(coalesced, op=dist.ReduceOp.SUM, group=comm.get_group(group), async_op=False)
            for buf, synced in zip(grads, _unflatten_dense_tensors(coalesced, grads)):
                buf.copy_(synced)
            return bucket.buffer()

        for group in comm.get_names():
            if group == "data":
                continue
            grads = []
            for p in params:
                if group in p.is_shared_mp:
                    if p.grad is not None:
                        grads.append(p.grad.data)
            if not grads:
                continue
            """Append the new reduction functions"""
            fut = fut.then(lambda x: grad_reduction(x, grads=grads, group=group))

        return fut
    """Register model comm hook"""
    model.register_comm_hook(state=None, hook=reduction_comm_hook)
    return model
