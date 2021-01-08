class ResidualAttentionBlock(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  attn : __torch__.torch.nn.modules.activation.___torch_mangle_9396.MultiheadAttention
  ln_1 : __torch__.multimodal.model.multimodal_transformer.___torch_mangle_9397.LayerNorm
  mlp : __torch__.torch.nn.modules.container.___torch_mangle_9401.Sequential
  ln_2 : __torch__.multimodal.model.multimodal_transformer.___torch_mangle_9402.LayerNorm
  def forward(self: __torch__.multimodal.model.multimodal_transformer.___torch_mangle_9403.ResidualAttentionBlock,
    argument_1: Tensor) -> Tensor:
    _0 = self.mlp
    _1 = self.ln_2
    _2 = (self.attn).forward((self.ln_1).forward(argument_1, ), )
    x = torch.add(argument_1, _2, alpha=1)
    x0 = torch.add(x, (_0).forward((_1).forward(x, ), ), alpha=1)
    return x0
  def forward1(self: __torch__.multimodal.model.multimodal_transformer.___torch_mangle_9403.ResidualAttentionBlock,
    argument_1: Tensor) -> Tensor:
    _3 = self.mlp
    _4 = self.ln_2
    _5 = (self.attn).forward1((self.ln_1).forward1(argument_1, ), )
    x = torch.add(argument_1, _5, alpha=1)
    x1 = torch.add(x, (_3).forward1((_4).forward1(x, ), ), alpha=1)
    return x1
