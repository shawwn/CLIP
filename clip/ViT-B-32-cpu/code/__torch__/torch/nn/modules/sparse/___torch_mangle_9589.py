class Embedding(Module):
  __parameters__ = ["weight", ]
  __buffers__ = []
  weight : Tensor
  training : bool
  def forward(self: __torch__.torch.nn.modules.sparse.___torch_mangle_9589.Embedding,
    input: Tensor) -> Tensor:
    _0 = torch.embedding(self.weight, input, -1, False, False)
    return _0
  def forward1(self: __torch__.torch.nn.modules.sparse.___torch_mangle_9589.Embedding,
    input: Tensor) -> Tensor:
    _1 = torch.embedding(self.weight, input, -1, False, False)
    return _1
