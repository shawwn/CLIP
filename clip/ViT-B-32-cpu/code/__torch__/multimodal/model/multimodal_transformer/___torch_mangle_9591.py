class Multimodal(Module):
  __parameters__ = ["positional_embedding", "text_projection", "logit_scale", ]
  __buffers__ = ["input_resolution", "context_length", "vocab_size", ]
  positional_embedding : Tensor
  text_projection : Tensor
  logit_scale : Tensor
  input_resolution : Tensor
  context_length : Tensor
  vocab_size : Tensor
  training : bool
  visual : __torch__.multimodal.model.multimodal_transformer.VisualTransformer
  transformer : __torch__.multimodal.model.multimodal_transformer.___torch_mangle_9588.Transformer
  token_embedding : __torch__.torch.nn.modules.sparse.___torch_mangle_9589.Embedding
  ln_final : __torch__.multimodal.model.multimodal_transformer.___torch_mangle_9590.LayerNorm
  def encode_image(self: __torch__.multimodal.model.multimodal_transformer.___torch_mangle_9591.Multimodal,
    image: Tensor) -> Tensor:
    _0 = self.visual
    input = torch.to(image, torch.device("cpu"), 5, False, False, None)
    return (_0).forward(input, )
  def encode_text(self: __torch__.multimodal.model.multimodal_transformer.___torch_mangle_9591.Multimodal,
    input: Tensor) -> Tensor:
    _1 = self.text_projection
    _2 = self.ln_final
    _3 = self.transformer
    _4 = self.positional_embedding
    _5 = (self.token_embedding).forward(input, )
    x = torch.to(_5, torch.device("cpu"), 5, False, False, None)
    _6 = torch.to(_4, torch.device("cpu"), 5, False, False, None)
    x0 = torch.add(x, _6, alpha=1)
    x1 = torch.permute(x0, [1, 0, 2])
    x2 = torch.permute((_3).forward(x1, ), [1, 0, 2])
    x3 = torch.to((_2).forward(x2, ), torch.device("cpu"), 5, False, False, None)
    _7 = ops.prim.NumToTensor(torch.size(x3, 0))
    _8 = torch.arange(annotate(number, _7), dtype=None, layout=0, device=torch.device("cpu"), pin_memory=False)
    _9 = torch.argmax(input, -1, False)
    _10 = torch.to(_8, dtype=4, layout=0, device=torch.device("cpu"), pin_memory=False)
    _11 = torch.to(_9, dtype=4, layout=0, device=torch.device("cpu"), pin_memory=False)
    _12 = annotate(List[Optional[Tensor]], [_10, _11])
    _13 = torch.matmul(torch.index(x3, _12).float(), _1.float())
    return _13
  def forward(self: __torch__.multimodal.model.multimodal_transformer.___torch_mangle_9591.Multimodal,
    image: Tensor,
    input: Tensor) -> Tuple[Tensor, Tensor]:
    _14 = self.logit_scale
    _15 = self.text_projection
    _16 = self.ln_final
    _17 = self.transformer
    _18 = self.positional_embedding
    _19 = self.token_embedding
    _20 = self.visual
    input0 = torch.to(image, torch.device("cpu"), 5, False, False, None)
    _21 = (_20).forward1(input0, )
    x = torch.to((_19).forward1(input, ), torch.device("cpu"), 5, False, False, None)
    _22 = torch.to(_18, torch.device("cpu"), 5, False, False, None)
    x4 = torch.add(x, _22, alpha=1)
    x5 = torch.permute(x4, [1, 0, 2])
    x6 = torch.permute((_17).forward1(x5, ), [1, 0, 2])
    x7 = torch.to((_16).forward1(x6, ), torch.device("cpu"), 5, False, False, None)
    _23 = ops.prim.NumToTensor(torch.size(x7, 0))
    _24 = torch.arange(annotate(number, _23), dtype=None, layout=0, device=torch.device("cpu"), pin_memory=False)
    _25 = torch.argmax(input, -1, False)
    _26 = torch.to(_24, dtype=4, layout=0, device=torch.device("cpu"), pin_memory=False)
    _27 = torch.to(_25, dtype=4, layout=0, device=torch.device("cpu"), pin_memory=False)
    _28 = annotate(List[Optional[Tensor]], [_26, _27])
    input1 = torch.matmul(torch.index(x7, _28), _15)
    _29 = torch.frobenius_norm(_21, [-1], True)
    image_features = torch.div(_21, _29)
    _30 = torch.frobenius_norm(input1, [-1], True)
    text_features = torch.div(input1, _30)
    logit_scale = torch.exp(_14)
    _31 = torch.mul(logit_scale, image_features)
    _32 = torch.matmul(_31, torch.t(text_features))
    _33 = torch.mul(logit_scale, text_features)
    _34 = torch.matmul(_33, torch.t(image_features))
    return (_32, _34)
