class ResidualAttentionBlock(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  attn : __torch__.torch.nn.modules.activation.___torch_mangle_9480.MultiheadAttention
  ln_1 : __torch__.multimodal.model.multimodal_transformer.___torch_mangle_9481.LayerNorm
  mlp : __torch__.torch.nn.modules.container.___torch_mangle_9485.Sequential
  ln_2 : __torch__.multimodal.model.multimodal_transformer.___torch_mangle_9486.LayerNorm
  def forward(self: __torch__.multimodal.model.multimodal_transformer.___torch_mangle_9487.ResidualAttentionBlock,
    x: Tensor) -> Tensor:
    _0 = self.mlp
    _1 = self.ln_2
    _2 = self.attn
    _3 = (self.ln_1).forward(x, )
    attn_mask = torch.to(CONSTANTS.c3, torch.device("cuda:0"), 5, False, False, None)
    x0 = torch.add(x, (_2).forward(_3, attn_mask, ), alpha=1)
    x1 = torch.add(x0, (_0).forward((_1).forward(x0, ), ), alpha=1)
    return x1
  def forward1(self: __torch__.multimodal.model.multimodal_transformer.___torch_mangle_9487.ResidualAttentionBlock,
    x: Tensor) -> Tensor:
    _4 = self.mlp
    _5 = self.ln_2
    _6 = self.attn
    _7 = (self.ln_1).forward1(x, )
    attn_mask = torch.to(CONSTANTS.c3, torch.device("cuda:0"), 5, False, False, None)
    x2 = torch.add(x, (_6).forward1(_7, attn_mask, ), alpha=1)
    _8 = (_4).forward1((_5).forward1(x2, ), )
    return torch.add(x2, _8, alpha=1)
