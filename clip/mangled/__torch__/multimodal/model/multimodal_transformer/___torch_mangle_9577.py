class ResidualAttentionBlock(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  attn : __torch__.torch.nn.modules.activation.___torch_mangle_9570.MultiheadAttention
  ln_1 : __torch__.multimodal.model.multimodal_transformer.___torch_mangle_9571.LayerNorm
  mlp : __torch__.torch.nn.modules.container.___torch_mangle_9575.Sequential
  ln_2 : __torch__.multimodal.model.multimodal_transformer.___torch_mangle_9576.LayerNorm
  def forward(self: __torch__.multimodal.model.multimodal_transformer.___torch_mangle_9577.ResidualAttentionBlock,
    argument_1: Tensor) -> Tensor:
    _0 = self.mlp
    _1 = self.ln_2
    _2 = self.attn
    _3 = (self.ln_1).forward(argument_1, )
    attn_mask = torch.to(CONSTANTS.c3, torch.device("cuda:0"), 5, False, False, None)
    x = torch.add(argument_1, (_2).forward(_3, attn_mask, ), alpha=1)
    x0 = torch.add(x, (_0).forward((_1).forward(x, ), ), alpha=1)
    return x0
  def forward1(self: __torch__.multimodal.model.multimodal_transformer.___torch_mangle_9577.ResidualAttentionBlock,
    argument_1: Tensor) -> Tensor:
    _4 = self.mlp
    _5 = self.ln_2
    _6 = self.attn
    _7 = (self.ln_1).forward1(argument_1, )
    attn_mask = torch.to(CONSTANTS.c3, torch.device("cuda:0"), 5, False, False, None)
    x = torch.add(argument_1, (_6).forward1(_7, attn_mask, ), alpha=1)
    x1 = torch.add(x, (_4).forward1((_5).forward1(x, ), ), alpha=1)
    return x1
