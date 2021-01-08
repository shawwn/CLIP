class LayerNorm(Module):
  __parameters__ = ["weight", "bias", ]
  __buffers__ = []
  weight : Tensor
  bias : Tensor
  training : bool
  def forward(self: __torch__.multimodal.model.multimodal_transformer.___torch_mangle_9367.LayerNorm,
    x: Tensor) -> Tensor:
    _0 = self.bias
    _1 = self.weight
    input = torch.to(x, torch.device("cuda:0"), 6, False, False, None)
    ret = torch.layer_norm(input, [768], _1, _0, 1.0000000000000001e-05, True)
    x0 = torch.to(ret, torch.device("cuda:0"), 5, False, False, None)
    return x0
  def forward1(self: __torch__.multimodal.model.multimodal_transformer.___torch_mangle_9367.LayerNorm,
    x: Tensor) -> Tensor:
    _2 = self.bias
    _3 = self.weight
    input = torch.to(x, torch.device("cuda:0"), 6, False, False, None)
    ret = torch.layer_norm(input, [768], _3, _2, 1.0000000000000001e-05, True)
    x1 = torch.to(ret, torch.device("cuda:0"), 5, False, False, None)
    return x1