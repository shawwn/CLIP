class VisualTransformer(Module):
  __parameters__ = ["class_embedding", "positional_embedding", "proj", ]
  __buffers__ = []
  class_embedding : Tensor
  positional_embedding : Tensor
  proj : Tensor
  training : bool
  conv1 : __torch__.torch.nn.modules.conv.___torch_mangle_9366.Conv2d
  ln_pre : __torch__.multimodal.model.multimodal_transformer.___torch_mangle_9367.LayerNorm
  transformer : __torch__.multimodal.model.multimodal_transformer.___torch_mangle_9477.Transformer
  ln_post : __torch__.multimodal.model.multimodal_transformer.___torch_mangle_9478.LayerNorm
  def forward(self: __torch__.multimodal.model.multimodal_transformer.VisualTransformer,
    input: Tensor) -> Tensor:
    _0 = self.proj
    _1 = self.ln_post
    _2 = self.transformer
    _3 = self.ln_pre
    _4 = self.positional_embedding
    _5 = self.class_embedding
    _6 = (self.conv1).forward(input, )
    _7 = ops.prim.NumToTensor(torch.size(_6, 0))
    _8 = int(_7)
    _9 = ops.prim.NumToTensor(torch.size(_6, 1))
    x = torch.reshape(_6, [_8, int(_9), -1])
    x0 = torch.permute(x, [0, 2, 1])
    _10 = torch.to(_5, 5, False, False, None)
    _11 = ops.prim.NumToTensor(torch.size(x0, 0))
    _12 = int(_11)
    _13 = ops.prim.NumToTensor(torch.size(x0, 2))
    _14 = torch.zeros([_12, 1, int(_13)], dtype=5, layout=None, device=torch.device("cuda:0"), pin_memory=False)
    x1 = torch.cat([torch.add(_10, _14, alpha=1), x0], 1)
    x2 = torch.add(x1, torch.to(_4, 5, False, False, None), alpha=1)
    x3 = torch.permute((_3).forward(x2, ), [1, 0, 2])
    x4 = torch.permute((_2).forward(x3, ), [1, 0, 2])
    _15 = torch.slice(x4, 0, 0, 9223372036854775807, 1)
    x5 = torch.slice(torch.select(_15, 1, 0), 1, 0, 9223372036854775807, 1)
    _16 = torch.matmul((_1).forward(x5, ), _0)
    return _16
  def forward1(self: __torch__.multimodal.model.multimodal_transformer.VisualTransformer,
    input: Tensor) -> Tensor:
    _17 = self.proj
    _18 = self.ln_post
    _19 = self.transformer
    _20 = self.ln_pre
    _21 = self.positional_embedding
    _22 = self.class_embedding
    _23 = (self.conv1).forward1(input, )
    _24 = ops.prim.NumToTensor(torch.size(_23, 0))
    _25 = int(_24)
    _26 = ops.prim.NumToTensor(torch.size(_23, 1))
    x = torch.reshape(_23, [_25, int(_26), -1])
    x6 = torch.permute(x, [0, 2, 1])
    _27 = torch.to(_22, 5, False, False, None)
    _28 = ops.prim.NumToTensor(torch.size(x6, 0))
    _29 = int(_28)
    _30 = ops.prim.NumToTensor(torch.size(x6, 2))
    _31 = torch.zeros([_29, 1, int(_30)], dtype=5, layout=None, device=torch.device("cuda:0"), pin_memory=False)
    x7 = torch.cat([torch.add(_27, _31, alpha=1), x6], 1)
    _32 = torch.to(_21, 5, False, False, None)
    x8 = torch.add(x7, _32, alpha=1)
    x9 = torch.permute((_20).forward1(x8, ), [1, 0, 2])
    x10 = torch.permute((_19).forward1(x9, ), [1, 0, 2])
    _33 = torch.slice(x10, 0, 0, 9223372036854775807, 1)
    x11 = torch.slice(torch.select(_33, 1, 0), 1, 0, 9223372036854775807, 1)
    input0 = torch.matmul((_18).forward1(x11, ), _17)
    return input0
