class QuickGELU(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  def forward(self: __torch__.multimodal.model.multimodal_transformer.___torch_mangle_9501.QuickGELU,
    argument_1: Tensor) -> Tensor:
    _0 = torch.sigmoid(torch.mul(argument_1, CONSTANTS.c2))
    return torch.mul(argument_1, _0)
  def forward1(self: __torch__.multimodal.model.multimodal_transformer.___torch_mangle_9501.QuickGELU,
    argument_1: Tensor) -> Tensor:
    _1 = torch.sigmoid(torch.mul(argument_1, CONSTANTS.c2))
    return torch.mul(argument_1, _1)
