class Transformer(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  resblocks : __torch__.torch.nn.modules.container.___torch_mangle_9476.Sequential
  def forward(self: __torch__.multimodal.model.multimodal_transformer.___torch_mangle_9477.Transformer,
    x: Tensor) -> Tensor:
    return (self.resblocks).forward(x, )
  def forward1(self: __torch__.multimodal.model.multimodal_transformer.___torch_mangle_9477.Transformer,
    x: Tensor) -> Tensor:
    return (self.resblocks).forward1(x, )
