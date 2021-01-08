class Conv2d(Module):
  __parameters__ = ["weight", ]
  __buffers__ = []
  weight : Tensor
  training : bool
  def forward(self: __torch__.torch.nn.modules.conv.___torch_mangle_9366.Conv2d,
    input: Tensor) -> Tensor:
    x = torch._convolution(input, self.weight, None, [32, 32], [0, 0], [1, 1], False, [0, 0], 1, False, False, True, True)
    return x
  def forward1(self: __torch__.torch.nn.modules.conv.___torch_mangle_9366.Conv2d,
    input: Tensor) -> Tensor:
    x = torch._convolution(input, self.weight, None, [32, 32], [0, 0], [1, 1], False, [0, 0], 1, False, False, True, True)
    return x
