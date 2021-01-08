class MultiheadAttention(Module):
  __parameters__ = ["in_proj_weight", "in_proj_bias", ]
  __buffers__ = []
  in_proj_weight : Tensor
  in_proj_bias : Tensor
  training : bool
  out_proj : __torch__.torch.nn.modules.linear.___torch_mangle_9431._LinearWithBias
  def forward(self: __torch__.torch.nn.modules.activation.___torch_mangle_9432.MultiheadAttention,
    argument_1: Tensor) -> Tensor:
    _0 = self.out_proj.bias
    _1 = self.out_proj.weight
    _2 = self.in_proj_bias
    _3 = self.in_proj_weight
    tgt_len = ops.prim.NumToTensor(torch.size(argument_1, 0))
    _4 = int(tgt_len)
    _5 = int(tgt_len)
    bsz = ops.prim.NumToTensor(torch.size(argument_1, 1))
    _6 = int(bsz)
    embed_dim = ops.prim.NumToTensor(torch.size(argument_1, 2))
    _7 = int(embed_dim)
    head_dim = torch.floor_divide(embed_dim, CONSTANTS.c0)
    _8 = int(head_dim)
    _9 = int(head_dim)
    _10 = int(head_dim)
    output = torch.matmul(argument_1.float(), torch.t(_3).float())
    _11 = torch.chunk(torch.add_(output, _2, alpha=1), 3, -1)
    q, k, v, = _11
    q0 = torch.mul(q, CONSTANTS.c1)
    q1 = torch.contiguous(q0, memory_format=0)
    _12 = [_5, int(torch.mul(bsz, CONSTANTS.c0)), _10]
    q2 = torch.transpose(torch.view(q1, _12), 0, 1)
    _13 = torch.contiguous(k, memory_format=0)
    _14 = [-1, int(torch.mul(bsz, CONSTANTS.c0)), _9]
    k0 = torch.transpose(torch.view(_13, _14), 0, 1)
    _15 = torch.contiguous(v, memory_format=0)
    _16 = [-1, int(torch.mul(bsz, CONSTANTS.c0)), _8]
    v0 = torch.transpose(torch.view(_15, _16), 0, 1)
    attn_output_weights = torch.bmm(q2, torch.transpose(k0, 1, 2))
    input = torch.softmax(attn_output_weights, -1, None)
    attn_output_weights0 = torch.dropout(input, 0., True)
    attn_output = torch.bmm(attn_output_weights0, v0)
    _17 = torch.contiguous(torch.transpose(attn_output, 0, 1), memory_format=0)
    input0 = torch.view(_17, [_4, _6, _7])
    output0 = torch.matmul(input0, torch.t(_1))
    return torch.add_(output0, _0, alpha=1)
  def forward1(self: __torch__.torch.nn.modules.activation.___torch_mangle_9432.MultiheadAttention,
    argument_1: Tensor) -> Tensor:
    _18 = self.out_proj.bias
    _19 = self.out_proj.weight
    _20 = self.in_proj_bias
    _21 = self.in_proj_weight
    tgt_len = ops.prim.NumToTensor(torch.size(argument_1, 0))
    _22 = int(tgt_len)
    _23 = int(tgt_len)
    bsz = ops.prim.NumToTensor(torch.size(argument_1, 1))
    _24 = int(bsz)
    embed_dim = ops.prim.NumToTensor(torch.size(argument_1, 2))
    _25 = int(embed_dim)
    head_dim = torch.floor_divide(embed_dim, CONSTANTS.c0)
    _26 = int(head_dim)
    _27 = int(head_dim)
    _28 = int(head_dim)
    output = torch.matmul(argument_1, torch.t(_21))
    _29 = torch.chunk(torch.add_(output, _20, alpha=1), 3, -1)
    q, k, v, = _29
    q3 = torch.mul(q, CONSTANTS.c1)
    q4 = torch.contiguous(q3, memory_format=0)
    _30 = [_23, int(torch.mul(bsz, CONSTANTS.c0)), _28]
    q5 = torch.transpose(torch.view(q4, _30), 0, 1)
    _31 = torch.contiguous(k, memory_format=0)
    _32 = [-1, int(torch.mul(bsz, CONSTANTS.c0)), _27]
    k1 = torch.transpose(torch.view(_31, _32), 0, 1)
    _33 = torch.contiguous(v, memory_format=0)
    _34 = [-1, int(torch.mul(bsz, CONSTANTS.c0)), _26]
    v1 = torch.transpose(torch.view(_33, _34), 0, 1)
    attn_output_weights = torch.bmm(q5, torch.transpose(k1, 1, 2))
    input = torch.softmax(attn_output_weights, -1, None)
    attn_output_weights1 = torch.dropout(input, 0., True)
    attn_output = torch.bmm(attn_output_weights1, v1)
    _35 = torch.contiguous(torch.transpose(attn_output, 0, 1), memory_format=0)
    input1 = torch.view(_35, [_22, _24, _25])
    output1 = torch.matmul(input1, torch.t(_19))
    return torch.add_(output1, _18, alpha=1)
