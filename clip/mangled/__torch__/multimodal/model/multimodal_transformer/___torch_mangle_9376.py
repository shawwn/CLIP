class ResidualAttentionBlock(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  attn : __torch__.torch.nn.modules.activation.___torch_mangle_9369.MultiheadAttention
  ln_1 : __torch__.multimodal.model.multimodal_transformer.___torch_mangle_9370.LayerNorm
  mlp : __torch__.torch.nn.modules.container.___torch_mangle_9374.Sequential
  ln_2 : __torch__.multimodal.model.multimodal_transformer.___torch_mangle_9375.LayerNorm
  def forward(self: __torch__.multimodal.model.multimodal_transformer.___torch_mangle_9376.ResidualAttentionBlock,
    x: Tensor) -> Tensor:
    _0 = self.mlp
    _1 = self.ln_2
    _2 = (self.attn).forward((self.ln_1).forward(x, ), )
    x0 = torch.add(x, _2, alpha=1)
    x1 = torch.add(x0, (_0).forward((_1).forward(x0, ), ), alpha=1)
    return x1
  def forward1(self: __torch__.multimodal.model.multimodal_transformer.___torch_mangle_9376.ResidualAttentionBlock,
    x: Tensor) -> Tensor:
    _3 = self.mlp
    _4 = self.ln_2
    _5 = (self.attn).forward1((self.ln_1).forward1(x, ), )
    x2 = torch.add(x, _5, alpha=1)
    _6 = (_3).forward1((_4).forward1(x2, ), )
    return torch.add(x2, _6, alpha=1)
