class Linear(Module):
  __parameters__ = ["weight", "bias", ]
  __buffers__ = []
  weight : Tensor
  bias : Tensor
  training : bool
  def forward(self: __torch__.torch.nn.modules.linear.___torch_mangle_9574.Linear,
    argument_1: Tensor) -> Tensor:
    _0 = self.bias
    output = torch.matmul(argument_1, torch.t(self.weight))
    return torch.add_(output, _0, alpha=1)
  def forward1(self: __torch__.torch.nn.modules.linear.___torch_mangle_9574.Linear,
    argument_1: Tensor) -> Tensor:
    _1 = self.bias
    output = torch.matmul(argument_1, torch.t(self.weight))
    return torch.add_(output, _1, alpha=1)
