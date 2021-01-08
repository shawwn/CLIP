class LayerNorm(Module):
  __parameters__ = ["weight", "bias", ]
  __buffers__ = []
  weight : Tensor
  bias : Tensor
  training : bool
  def forward(self: __torch__.multimodal.model.multimodal_transformer.___torch_mangle_9397.LayerNorm,
    argument_1: Tensor) -> Tensor:
    _0 = self.bias
    _1 = self.weight
    input = torch.to(argument_1, torch.device("cpu"), 6, False, False, None)
    ret = torch.layer_norm(input, [768], _1, _0, 1.0000000000000001e-05, True)
    query = torch.to(ret, torch.device("cpu"), 5, False, False, None)
    return query
  def forward1(self: __torch__.multimodal.model.multimodal_transformer.___torch_mangle_9397.LayerNorm,
    argument_1: Tensor) -> Tensor:
    _2 = self.bias
    _3 = self.weight
    input = torch.to(argument_1, torch.device("cpu"), 6, False, False, None)
    ret = torch.layer_norm(input, [768], _3, _2, 1.0000000000000001e-05, True)
    query = torch.to(ret, torch.device("cpu"), 5, False, False, None)
    return query
