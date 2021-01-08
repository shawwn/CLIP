class Sequential(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  c_fc : __torch__.torch.nn.modules.linear.___torch_mangle_9452.Linear
  gelu : __torch__.multimodal.model.multimodal_transformer.___torch_mangle_9453.QuickGELU
  c_proj : __torch__.torch.nn.modules.linear.___torch_mangle_9454.Linear
  def forward(self: __torch__.torch.nn.modules.container.___torch_mangle_9455.Sequential,
    argument_1: Tensor) -> Tensor:
    _0 = self.c_proj
    _1 = (self.gelu).forward((self.c_fc).forward(argument_1, ), )
    return (_0).forward(_1, )
  def forward1(self: __torch__.torch.nn.modules.container.___torch_mangle_9455.Sequential,
    argument_1: Tensor) -> Tensor:
    _2 = self.c_proj
    _3 = (self.gelu).forward1((self.c_fc).forward1(argument_1, ), )
    return (_2).forward1(_3, )
