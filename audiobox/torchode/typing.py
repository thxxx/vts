from jaxtyping import Bool, Float, Int64, Shaped
from torch import Tensor


def same_dtype(*tensors: Tensor):
    if len(tensors) <= 1:
        return True
    for a, b in zip(tensors[:-1], tensors[1:]):
        if a.dtype != b.dtype:
            return False
    return True


def same_device(*tensors: Tensor):
    if len(tensors) <= 1:
        return True
    for a, b in zip(tensors[:-1], tensors[1:]):
        if a.device != b.device:
            return False
    return True


def same_shape(*tensors: Tensor, dim: int | None = None):
    if len(tensors) <= 1:
        return True
    for a, b in zip(tensors[:-1], tensors[1:]):
        if dim is None:
            if a.shape != b.shape:
                return False
        else:
            if a.shape[dim] != b.shape[dim]:
                return False
    return True


################
# Tensor Types #
################

# These are subclasses of TensorType instead of using TensorType annotations directly,
# because TorchScript does not support custom type constructors. In this way, we can
# continue to document the shapes and types of tensors while being TorchScript
# compatible, see [1].
#
# [1] https://github.com/patrick-kidger/torchtyping/issues/13


DataTensor = Float[Tensor, "batch *feature"]
NormTensor = Float[Tensor, "batch"]
SolutionDataTensor = Float[Tensor, "time batch *feature"]
TimeTensor = Float[Tensor, "batch"]
EvaluationTimesTensor = Float[Tensor, "batch time"]
AcceptTensor = Bool[Tensor, "batch"]
StatusTensor = Int64[Tensor, "batch"]
InterpTimeTensor = Float[Tensor, "interp_points"]
InterpDataTensor = Float[Tensor, "interp_points *feature"]
SampleIndexTensor = Int64[Tensor, "interp_points"]
StepTensor = Float[Tensor, "weights batch *feature"]

CoefficientVector = Shaped[Tensor, "nodes"]
RungeKuttaMatrix = Shaped[Tensor, "nodes weights"]
WeightVector = Shaped[Tensor, "weights"]
WeightMatrix = Shaped[Tensor, "rows weights"]
