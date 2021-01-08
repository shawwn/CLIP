class Sequential(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  __annotations__["0"] = __torch__.multimodal.model.multimodal_transformer.___torch_mangle_9376.ResidualAttentionBlock
  __annotations__["1"] = __torch__.multimodal.model.multimodal_transformer.___torch_mangle_9385.ResidualAttentionBlock
  __annotations__["2"] = __torch__.multimodal.model.multimodal_transformer.___torch_mangle_9394.ResidualAttentionBlock
  __annotations__["3"] = __torch__.multimodal.model.multimodal_transformer.___torch_mangle_9403.ResidualAttentionBlock
  __annotations__["4"] = __torch__.multimodal.model.multimodal_transformer.___torch_mangle_9412.ResidualAttentionBlock
  __annotations__["5"] = __torch__.multimodal.model.multimodal_transformer.___torch_mangle_9421.ResidualAttentionBlock
  __annotations__["6"] = __torch__.multimodal.model.multimodal_transformer.___torch_mangle_9430.ResidualAttentionBlock
  __annotations__["7"] = __torch__.multimodal.model.multimodal_transformer.___torch_mangle_9439.ResidualAttentionBlock
  __annotations__["8"] = __torch__.multimodal.model.multimodal_transformer.___torch_mangle_9448.ResidualAttentionBlock
  __annotations__["9"] = __torch__.multimodal.model.multimodal_transformer.___torch_mangle_9457.ResidualAttentionBlock
  __annotations__["10"] = __torch__.multimodal.model.multimodal_transformer.___torch_mangle_9466.ResidualAttentionBlock
  __annotations__["11"] = __torch__.multimodal.model.multimodal_transformer.___torch_mangle_9475.ResidualAttentionBlock
  def forward(self: __torch__.torch.nn.modules.container.___torch_mangle_9476.Sequential,
    x: Tensor) -> Tensor:
    _0 = getattr(self, "11")
    _1 = getattr(self, "10")
    _2 = getattr(self, "9")
    _3 = getattr(self, "8")
    _4 = getattr(self, "7")
    _5 = getattr(self, "6")
    _6 = getattr(self, "5")
    _7 = getattr(self, "4")
    _8 = getattr(self, "3")
    _9 = getattr(self, "2")
    _10 = (getattr(self, "1")).forward((getattr(self, "0")).forward(x, ), )
    _11 = (_7).forward((_8).forward((_9).forward(_10, ), ), )
    _12 = (_4).forward((_5).forward((_6).forward(_11, ), ), )
    _13 = (_1).forward((_2).forward((_3).forward(_12, ), ), )
    return (_0).forward(_13, )
  def forward1(self: __torch__.torch.nn.modules.container.___torch_mangle_9476.Sequential,
    x: Tensor) -> Tensor:
    _14 = getattr(self, "11")
    _15 = getattr(self, "10")
    _16 = getattr(self, "9")
    _17 = getattr(self, "8")
    _18 = getattr(self, "7")
    _19 = getattr(self, "6")
    _20 = getattr(self, "5")
    _21 = getattr(self, "4")
    _22 = getattr(self, "3")
    _23 = getattr(self, "2")
    _24 = (getattr(self, "1")).forward1((getattr(self, "0")).forward1(x, ), )
    _25 = (_22).forward1((_23).forward1(_24, ), )
    _26 = (_20).forward1((_21).forward1(_25, ), )
    _27 = (_18).forward1((_19).forward1(_26, ), )
    _28 = (_16).forward1((_17).forward1(_27, ), )
    _29 = (_14).forward1((_15).forward1(_28, ), )
    return _29
