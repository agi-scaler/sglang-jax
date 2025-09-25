import random
import unittest

import torch
import torch.nn.functional as F

from sglang.srt.layers.attention.triton_ops.decode_attention import (
    decode_attention_fwd,
    decode_attention_fwd_grouped,
    decode_attention_fwd_normal,
)
from sglang.srt.layers.attention.triton_ops.extend_attention import (
    extend_attention_fwd,
    redundant_attention,
)
from sglang.srt.layers.attention.triton_ops.prefill_attention import (
    context_attention_fwd,
)
from sglang.test.test_utils import CustomTestCase


class TestTritonAttention(CustomTestCase):

    def _set_all_seeds(self, seed):
        """Set all random seeds for reproducibility."""
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def setUp(self):
        # Set seeds before each test method
        self._set_all_seeds(42)

    def _test_extend_attention_once(self, B, N_CTX, H_Q, H_KV, D, b_seq_len_prefix=None, b_seq_len_extend=None, xai_temperature_len=-1):
        dtype = torch.bfloat16
        # By Lin 
        # Begin: use input seq_len if available 
        b_seq_len_prefix = torch.randint(
            1, N_CTX // 2, (B,), dtype=torch.int32, device="cuda"
        ) if b_seq_len_prefix is None else b_seq_len_prefix
        b_seq_len_extend = torch.randint(
            1, N_CTX // 2, (B,), dtype=torch.int32, device="cuda"
        ) if b_seq_len_extend is None else b_seq_len_extend
        # End

        b_seq_len = b_seq_len_prefix + b_seq_len_extend
        max_len_in_batch = torch.max(b_seq_len, 0)[0].item()

        # By Lin
        # Begin: load ref results from TPU
        mode = 'prefill'
        qkv_path = f'{mode}_{H_Q}_{D}_{H_KV}_{b_seq_len_extend.detach().cpu().item()}_{b_seq_len_prefix.detach().cpu().item()}_temp{None if xai_temperature_len == -1 else xai_temperature_len}.npy'
        print(f"Testing {qkv_path}", flush=True)
        import numpy as np
        qkv_np = np.load(f'/opt/tiger/open_verl/sglang-jax/{qkv_path}', allow_pickle=True).item()
        q_np, k_np, v_np, fa_np, naive_np = qkv_np['q'], qkv_np['k'], qkv_np['v'], qkv_np['fa'], qkv_np['naive']
        print("qkv_np.shape", k_np.shape, v_np.shape, q_np.shape, np.isnan(q_np).any(), np.isnan(k_np).any(), np.isnan(v_np).any(), np.isnan(fa_np).any(), np.isnan(naive_np).any())
        fa = torch.from_numpy(fa_np).to('cuda').to(dtype)
        naive = torch.from_numpy(naive_np).to('cuda').to(dtype)
        # End

        b_req_idx = torch.arange(B, dtype=torch.int32, device="cuda")
        b_start_loc = torch.zeros((B,), dtype=torch.int32, device="cuda")
        b_start_loc[1:] = torch.cumsum(b_seq_len[:-1], 0)
        b_start_loc_extend = torch.zeros((B,), dtype=torch.int32, device="cuda")
        b_start_loc_extend[1:] = torch.cumsum(b_seq_len_extend[:-1], 0)

        kv_indptr = torch.zeros((B + 1,), dtype=torch.int32, device="cuda")
        kv_indptr[1 : B + 1] = torch.cumsum(b_seq_len_prefix[:B], dim=0)
        kv_indices = torch.zeros(
            (b_seq_len_prefix.sum().item(),), dtype=torch.int32, device="cuda"
        )

        for i in range(B):
            kv_indices[kv_indptr[i] : kv_indptr[i + 1]] = torch.arange(
                b_start_loc[i], b_start_loc[i] + b_seq_len_prefix[i]
            )

        total_token_num = torch.sum(b_seq_len).item()
        extend_token_num = torch.sum(b_seq_len_extend).item()
        k_buffer = torch.empty(
            (total_token_num, H_KV, D), dtype=dtype, device="cuda"
        ).normal_(mean=0.1, std=0.2)
        v_buffer = torch.empty(
            (total_token_num, H_KV, D), dtype=dtype, device="cuda"
        ).normal_(mean=0.1, std=0.2)

        num_prefix_tokens = b_seq_len_prefix.sum().item()

        k_extend = torch.empty((extend_token_num, H_KV, D), dtype=dtype, device="cuda")
        v_extend = torch.empty((extend_token_num, H_KV, D), dtype=dtype, device="cuda")
        q_extend = torch.empty((extend_token_num, H_Q, D), dtype=dtype, device="cuda")

        for i in range(B):
            extend_start_in_buffer = b_start_loc[i] + b_seq_len_prefix[i]
            extend_end_in_buffer = b_start_loc[i] + b_seq_len[i]
            extend_start = b_start_loc_extend[i]
            extend_end = b_start_loc_extend[i] + b_seq_len_extend[i]
            k_extend[extend_start:extend_end] = k_buffer[
                extend_start_in_buffer:extend_end_in_buffer
            ]
            v_extend[extend_start:extend_end] = v_buffer[
                extend_start_in_buffer:extend_end_in_buffer
            ]
            q_extend[extend_start:extend_end] = torch.empty(
                (b_seq_len_extend[i], H_Q, D), dtype=dtype, device="cuda"
            ).normal_(mean=0.1, std=0.2)

        # By Lin
        # Begin
        k_buffer[:num_prefix_tokens] = torch.from_numpy(k_np).to('cuda').to(k_buffer.dtype)
        v_buffer[:num_prefix_tokens] = torch.from_numpy(v_np).to('cuda').to(v_buffer.dtype)
        q_extend[:] = torch.from_numpy(q_np).to('cuda').to(dtype)
        # End

        o_extend = torch.empty((extend_token_num, H_Q, D), dtype=dtype, device="cuda")
        o_extend_mask = torch.empty(
            (extend_token_num, H_Q, D), dtype=dtype, device="cuda"
        )
        o_redundant = torch.empty(
            (extend_token_num, H_Q, D), dtype=dtype, device="cuda"
        )

        b_seq_len_extend = b_seq_len - b_seq_len_prefix
        max_len_extend = torch.max(b_seq_len_extend, 0)[0].item()
        qo_indptr = torch.zeros((B + 1,), dtype=torch.int32, device="cuda")
        qo_indptr[1 : B + 1] = torch.cumsum(b_seq_len_extend[:B], dim=0)

        custom_mask = None
        mask_indptr = None

        extend_attention_fwd(
            q_extend,
            k_extend,
            v_extend,
            o_extend,
            k_buffer,
            v_buffer,
            qo_indptr,
            kv_indptr,
            kv_indices,
            custom_mask,
            True,
            mask_indptr,
            max_len_extend,
            xai_temperature_len=xai_temperature_len # pass temperature
        )

        b_seq_mask_len = b_seq_len_extend * b_seq_len
        custom_mask = torch.ones(
            (b_seq_mask_len.sum().item(),), dtype=torch.bool, device="cuda"
        )
        mask_indptr = torch.zeros((B + 1,), dtype=torch.int64, device="cuda")
        mask_indptr[1 : B + 1] = torch.cumsum(b_seq_mask_len[:B], dim=0)
        for i in range(B):
            causal_mask = (
                torch.tril(
                    torch.ones(b_seq_len_extend[i], b_seq_len_extend[i]), diagonal=0
                )
                == 1
            )
            prefix_mask = torch.ones(
                b_seq_len_extend[i], b_seq_len_prefix[i], dtype=torch.bool
            )
            mask_flatten = torch.cat([prefix_mask, causal_mask], dim=1).flatten()
            custom_mask[mask_indptr[i] : mask_indptr[i + 1]] = mask_flatten

        extend_attention_fwd(
            q_extend,
            k_extend,
            v_extend,
            o_extend_mask,
            k_buffer,
            v_buffer,
            qo_indptr,
            kv_indptr,
            kv_indices,
            custom_mask,
            True,
            mask_indptr,
            max_len_extend,
            xai_temperature_len=xai_temperature_len
        )

        redundant_attention(
            q_extend,
            o_redundant,
            k_buffer,
            v_buffer,
            b_req_idx,
            b_start_loc,
            b_seq_len,
            b_seq_len_prefix,
            max_len_in_batch,
        )

        # self.assertTrue(torch.allclose(o_extend, o_redundant, rtol=1e-2))
        print('Diff', o_extend.reshape(-1) - naive.reshape(-1), (o_extend.reshape(-1) - naive.reshape(-1)).abs().max(), flush=True)



        # self.assertTrue(torch.allclose(o_extend.reshape(-1), fa.reshape(-1), rtol=1e-2))
        # self.assertTrue(torch.allclose(o_extend.reshape(-1), naive.reshape(-1), rtol=1e-2))
        # self.assertTrue(torch.allclose(o_extend_mask, o_redundant, rtol=1e-2))

    def test_extend_attention_dump(self):

        # Define the varying parameter values
        attention_value = 128
        B, N_CTX, H_Q, H_KV, D = 1, 2048, 32, 8, attention_value
        for extend, prefix in zip([1, 3, 64, 20, 125, 123, 1], [128, 20, 64, 20, 125, 522, 511]):
            b_seq_len_prefix = torch.tensor([prefix], dtype=torch.int32, device='cuda')
            b_seq_len_extend = torch.tensor([extend], dtype=torch.int32, device='cuda')

            # Loop through the values and call the method
            self._test_extend_attention_once(B, N_CTX, H_Q, H_KV, D, b_seq_len_prefix=b_seq_len_prefix, b_seq_len_extend=b_seq_len_extend, xai_temperature_len=-1)

        # B = 7
        # b_seq_len_prefix = torch.tensor([1, 3, 64, 20, 125, 123, 1], dtype=torch.int32, device='cuda')
        # b_seq_len_extend = torch.tensor([128, 20, 64, 20, 125, 522, 511], dtype=torch.int32, device='cuda')
        # Loop through the values and call the method
        # self._test_extend_attention_once(B, N_CTX, H_Q, H_KV, D, b_seq_len_prefix=b_seq_len_prefix, b_seq_len_extend=b_seq_len_extend)


    def test_extend_attention(self):

        # Define the varying parameter values
        attention_values = [128, 96, 80, 13]

        # Loop through the values and call the method
        for value in attention_values:
            self._test_extend_attention_once(19, 12331, 12, 4, value)

    def test_decode_attention_temperature_len(self):
        # Here we just to ensure there is no error
        # TODO: correctnesss test
        # Test configurations
        configs = [
            (1, 32, 8, 128, 119),
            (1, 32, 8, 128, 127),
            (1, 32, 8, 128, 128),
            (1, 32, 8, 128, 129),
            (1, 32, 8, 128, 133),
            (1, 32, 8, 128, 1001),
            (1, 32, 8, 128, 1023),
            (1, 32, 8, 128, 1024),
            (1, 32, 8, 128, 1025),
        ]

        for B, H_Q, H_KV, D, seq_len in configs:
            self._test_decode_attention_once(B, H_Q, H_KV, D, xai_temperature_len=512, seq_len=seq_len)

    def _test_decode_attention_once(self, B, H_Q, H_KV, D, xai_temperature_len=-1, seq_len=128):
        dtype = torch.bfloat16
        total_tokens = B * seq_len
        sm_scale = 1.0 / (D**0.5)
        max_kv_splits = 8
        num_kv_splits = torch.full((B,), 4, dtype=torch.int32, device="cuda")

        # q represents the new token being generated, one per batch
        q = torch.randn(B, H_Q, D, dtype=dtype, device="cuda")

        # k_buffer and v_buffer represent all previous tokens
        k_buffer = torch.randn(total_tokens, H_KV, D, dtype=dtype, device="cuda")
        v_buffer = torch.randn(total_tokens, H_KV, D, dtype=dtype, device="cuda")

        # load_from_numpy
        import numpy as np
        mode = 'decode'
        qkv_path = f'{mode}_{H_Q}_{D}_{H_KV}_1_{seq_len}.npy'

        qkv_np = np.load(f'/opt/tiger/open_verl/sglang-jax/{qkv_path}', allow_pickle=True).item()
        # print("qkv_np", type(qkv_np), qkv_np)
        q_np, k_np, v_np, fa_np, naive_np = qkv_np['q'], qkv_np['k'], qkv_np['v'], qkv_np['fa'], qkv_np['naive']
        q = torch.from_numpy(q_np).to('cuda').to(dtype)
        k_buffer = torch.from_numpy(k_np).to('cuda').to(dtype)
        v_buffer = torch.from_numpy(v_np).to('cuda').to(dtype)

        # o will have the same shape as q
        o = torch.zeros(B, H_Q, D, dtype=dtype, device="cuda")

        b_seq_len = torch.full((B,), seq_len, device="cuda")

        kv_indptr = torch.zeros((B + 1,), dtype=torch.int32, device="cuda")
        kv_indptr[1 : B + 1] = torch.cumsum(b_seq_len[:B], dim=0)
        kv_indices = torch.arange(total_tokens, device="cuda")
        # print("kv_indptr", kv_indptr, flush=True)

        attn_logits = torch.empty(
            (B, H_Q, max_kv_splits, D),
            dtype=torch.float32,
            device="cuda",
        )
        attn_lse = torch.empty(
            (B, H_Q, max_kv_splits),
            dtype=torch.float32,
            device="cuda",
        )

        decode_attention_fwd(
            q,
            k_buffer,
            v_buffer,
            o,
            kv_indptr,
            kv_indices,
            attn_logits,
            attn_lse,
            num_kv_splits,
            max_kv_splits,
            sm_scale,
            xai_temperature_len=xai_temperature_len,
        )
        print("testing", qkv_path)
        o_1d = o.float().cpu().detach().numpy().reshape(-1)
        diff = o_1d - naive_np.reshape(-1)
        print(o_1d, naive_np.reshape(-1), diff, diff.max(), diff.min(), flush=True)
        self.assertTrue(np.allclose(o.float().cpu().detach().numpy().reshape(-1), naive_np.reshape(-1), atol=6e-2))
        self.assertTrue(np.allclose(o.float().cpu().detach().numpy().reshape(-1), fa_np.reshape(-1), atol=6e-2))
        print("passed", flush=True)

    def test_decode_attention(self):
        # Here we just to ensure there is no error
        # TODO: correctnesss test

        # Test configurations
        configs = [
            (2, 4, 4, 64),  # MHA
            (2, 4, 2, 64),  # GQA
            (2, 4, 4, 80),  # Non-standard head dim
            (2, 4, 4, 13),  # Prime number head dim
        ]

        for B, H_Q, H_KV, D in configs:
            self._test_decode_attention_once(B, H_Q, H_KV, D)


if __name__ == "__main__":
    unittest.main()

