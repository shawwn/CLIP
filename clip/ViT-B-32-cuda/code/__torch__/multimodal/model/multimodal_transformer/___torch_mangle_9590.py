class LayerNorm(Module):
  __parameters__ = ["weight", "bias", ]
  __buffers__ = []
  weight : Tensor
  bias : Tensor
  training : bool
  def forward(self: __torch__.multimodal.model.multimodal_transformer.___torch_mangle_9590.LayerNorm,
    x: Tensor) -> Tensor:
    _0 = self.bias
    _1 = self.weight
    input = torch.to(x, torch.device("cuda:0"), 6, False, False, None)
    ret = torch.layer_norm(input, [512], _1, _0, 1.0000000000000001e-05, True)
    _2 = torch.to(ret, torch.device("cuda:0"), 5, False, False, None)
    return _2
  def forward1(self: __torch__.multimodal.model.multimodal_transformer.___torch_mangle_9590.LayerNorm,
    x: Tensor) -> Tensor:
    _3 = self.bias
    _4 = self.weight
    input = torch.to(x, torch.device("cuda:0"), 6, False, False, None)
    ret = torch.layer_norm(input, [512], _4, _3, 1.0000000000000001e-05, True)
    _5 = torch.to(ret, torch.device("cuda:0"), 5, False, False, None)
    return _5
