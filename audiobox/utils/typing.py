from jaxtyping import Bool, Float, Int
from torch import Tensor

AudioTensor = Float[Tensor, "batch audio audio_channel"]
AudioMaskTensor = Bool[Tensor, "batch audio"]
EncTensor = Float[Tensor, "batch codec channel"]
EncMaskTensor = Bool[Tensor, "batch codec"]
LengthTensor = Int[Tensor, "batch"]
LossTensor = Float[Tensor, ""]
TimeTensor = Float[Tensor, "batch"]
Batch = tuple[AudioTensor, AudioMaskTensor]
