# SGLang-13044

[Bug] Flashinfer backend get wrong result on last layer of qwen2 model #13044
Open
@jianfei-wangg
Description
jianfei-wangg
opened last month · edited by jianfei-wangg
Contributor
Checklist

1. I have searched related issues but cannot get the expected help.

2. The bug has not been fixed in the latest version.

3. Please note that if the bug-related issue you submitted lacks corresponding environment info and a minimal reproducible demo, it will be challenging for us to reproduce and resolve the issue, reducing the likelihood of receiving feedback.

4. If the issue you raised is not a bug but a question, please raise a discussion at https://github.com/sgl-project/sglang/discussions/new/choose Otherwise, it will be closed.

5. Please use English, otherwise it will be closed.
Describe the bug
I have encountered qwen2model accuracy problem, and I found it can be reproduced by compare scaled_dot_product_attention with sglang's attention_backend, fa3 or triton has no problem.

Reproduction
here is the code snippet:
class Qwen2Attention(nn.Module):

    self.layer_id = layer_id

    self.qkv_proj = QKVParallelLinear(
        hidden_size,
        self.head_dim,
        self.total_num_heads,
        self.total_num_kv_heads,
        bias=True,
        quant_config=quant_config,
        prefix=add_prefix("qkv_proj", prefix),
    )
    self.o_proj = RowParallelLinear(
        self.total_num_heads * self.head_dim,
        hidden_size,
        bias=False,
        quant_config=quant_config,
        prefix=add_prefix("o_proj", prefix),
    )

    self.rotary_emb = get_rope(
        self.head_dim,
        rotary_dim=self.head_dim,
        max_position=max_position_embeddings,
        base=rope_theta,
        rope_scaling=rope_scaling,
        dual_chunk_attention_config=dual_chunk_attention_config,
    )
    self.attn = RadixAttention(
        self.num_heads,
        self.head_dim,
        self.scaling,
        num_kv_heads=self.num_kv_heads,
        layer_id=layer_id,
        quant_config=quant_config,
        prefix=add_prefix("attn", prefix),
    )

def _scaled_dot_product_attention(self, q, k, v):
    import torch
    seq_len = q.shape[0]
    q_reshaped = q.view(seq_len, self.num_heads, self.head_dim).unsqueeze(0).transpose(1, 2)
    k_reshaped = k.view(seq_len, self.num_kv_heads, self.head_dim).unsqueeze(0).transpose(1, 2)
    v_reshaped = v.view(seq_len, self.num_kv_heads, self.head_dim).unsqueeze(0).transpose(1, 2)
    if self.num_kv_heads != self.num_heads:
        k_reshaped = k_reshaped.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
        v_reshaped = v_reshaped.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
    attn_output_scaled = torch.nn.functional.scaled_dot_product_attention(q_reshaped, k_reshaped, v_reshaped, is_causal=True)
    return attn_output_scaled.transpose(1, 2).squeeze(0).reshape(seq_len, -1)

def _compare_two_tensors(self, tensor1, tensor2):
    import torch
    diff = tensor1 - tensor2
    max_diff = torch.max(torch.abs(diff))
    mean_diff = torch.mean(torch.abs(diff))
    cosine_sim = torch.nn.functional.cosine_similarity(tensor1, tensor2, dim=-1)
    print(f"layer_{self.layer_id} Max diff: {max_diff}, Mean diff: {mean_diff}, Cosine sim: {cosine_sim}")

def forward(
    self,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    forward_batch: ForwardBatch,
) -> torch.Tensor:
    qkv, _ = self.qkv_proj(hidden_states)
    q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
    q, k = self.rotary_emb(positions, q, k)
    attn_output = self.attn(q, k, v, forward_batch)
    if hidden_states.shape[0] > 1:
        scaled_attn_output = self._scaled_dot_product_attention(q, k, v)
        self._compare_two_tensors(attn_output, scaled_attn_output)
    output, _ = self.o_proj(attn_output)
    return output`
and then run:
python -m sglang.launch_server --model-path Qwen/Qwen2-7B-Instruct --context-length 32768 --max-running-requests 32 --tp-size 1 --port 8192 --cuda-graph-max-bs 32 --mem-fraction-static 0.85 --trust-remote-code --chunked-prefill-size -1 --log-level info --disable-cuda-graph --attention-backend flashinfer --host '0.0.0.0'

