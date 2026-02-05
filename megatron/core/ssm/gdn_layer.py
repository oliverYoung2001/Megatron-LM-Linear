# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, Tri Dao, Albert Gu.

# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.
# TODO
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Union

import torch
from torch import Tensor
import torch.nn.functional as F
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.dist_checkpointing.utils import apply_prefix_mapping
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.module import GraphableMegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import deprecate_inference_params
from torch import nn
from causal_conv1d import causal_conv1d_fn
from lacp.la_ops.gated_delta_rule.chunk import chunk_gated_delta_rule

@dataclass
class GDNLayerSubmodules:
    """
    Configuration class for specifying the submodules of a GDN layer.

    This class defines the structure and default implementations for various
    components of a GDN layer, allowing for flexible customization of the
    layer's architecture.

    Args:
        norm (Union[ModuleSpec, type]): Specification for the input layer normalization.
        mixer (Union[ModuleSpec, type]): Specification for the along-sequence mixing mechanism.
        gdn_bda (Union[ModuleSpec, type]): Specification for the bias-dropout-add operation
            after the mixer.
    """

    # norm: Union[ModuleSpec, type] = IdentityOp
    # mixer: Union[ModuleSpec, type] = IdentityOp # TODO
    # gdn_bda: Union[ModuleSpec, type] = IdentityOp

    # Mapping for sharded tensor keys to be applied in `sharded_state_dict` method
    sharded_state_dict_keys_map: Dict[str, str] = field(default_factory=dict)

    in_proj_qkvz: Union[ModuleSpec, type] = None
    in_proj_ba: Union[ModuleSpec, type] = None
    out_proj: Union[ModuleSpec, type] = None
    # causal_conv1d_fn: Union[ModuleSpec, type] = None    # TODO
    norm: Union[ModuleSpec, type] = None


class GDNLayer(GraphableMegatronModule):
    """
    A single GDN layer.

    GDN layer takes input with size [s, b, h] and returns an
    output of the same size.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: GDNLayerSubmodules,
        layer_number: int = 1,
        residual_in_fp32=False,
        pg_collection: ProcessGroupCollection = None,
        pp_layer_offset: int = 0,
    ):
        """Initialize GDN Layer."""
        super().__init__(config)
        assert pg_collection is not None, "pg_collection must be provided for GDNLayer"

        self.config = config
        self.submodules_config = submodules
        self.layer_number = layer_number
        self.residual_in_fp32 = residual_in_fp32
        self.hidden_dropout = config.hidden_dropout
        self.pg_collection = pg_collection
        self.hidden_size = self.config.hidden_size   # 2k

        self.dk = self.config.gdn_dk  # Dk=128
        self.dv = self.config.gdn_dv   # Dv=128
        self.hqk = self.config.gdn_hqk # Hqk=16
        self.hv = self.config.gdn_hv # Hv=32
        tp_size = self.pg_collection.tp.size()
        assert self.hqk % tp_size == 0, "hqk must be evenly divisble by tp_size"
        self.hqk_local_tp = self.hqk // tp_size
        assert self.hv % tp_size == 0, "hv must be evenly divisble by tp_size"
        self.hv_local_tp = self.hv // tp_size

        self.in_proj_qkvz = build_module(
            submodules.in_proj_qkvz,
            self.hidden_size,
            self.hqk * self.dk * 2 + self.hv * self.dv * 2,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=False,
            skip_bias_add=False,    # ?
            is_expert=False,
            tp_comm_buffer_name="fc1",
            tp_group=self.pg_collection.tp,
        )
        self.in_proj_ba = build_module(
            submodules.in_proj_ba,
            self.hidden_size,
            self.hv * 2,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=False,
            skip_bias_add=False,    # ?
            is_expert=False,
            tp_comm_buffer_name="fc1",
            tp_group=self.pg_collection.tp,
        )
        self.out_proj = build_module(
            submodules.out_proj,
            self.hv * self.dv,
            self.hidden_size,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=False,
            input_is_parallel=True,
            skip_bias_add=True, # ?
            is_expert=False,
            tp_comm_buffer_name="fc2",
            tp_group=self.pg_collection.tp,
        )
        self.norm = build_module(
            submodules.norm,
            self.dv,
            eps=1e-06,
            activation='silu',
            # device=torch.cuda.current_device(), # CPU?
            device='cpu' if config.use_cpu_initialization else torch.cuda.current_device(), # use_cpu_initialization=None
            dtype=config.params_dtype,    # For weights
        )
        self.conv_dim = self.hqk_local_tp * self.dk * 2 + self.hv_local_tp * self.dv  # [hqk/tp*dk*2+hv/tp*dv]
        self.conv_kernel_size = config.linear_conv_kernel_dim   # 4
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,  # 4
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,  # 3
        )
        # For gate
        self.dt_bias = nn.Parameter(torch.ones(self.hv_local_tp))   # [hv/tp]
        A = torch.empty(self.hv_local_tp).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A)) # [hv/tp]

    def qwen3_next_attn_fwd_no_memcpy(self, inpx, time_events: dict | None = None):
        # [NOTE]: Assume that B=1!
        hqk, hv, dk, dv = self.hqk, self.hv, self.dk, self.dv
        hqk_tp, hv_tp = self.hqk_local_tp, self.hv_local_tp
        # inpx [(B*S)/dcp/tp, H]
        mixed_ba, _ = self.in_proj_ba(inpx)    # [(B*S)/dcp, hv/tp*2] ---ag--> [(B*S)/dcp, hv/tp*2]
        b, a = torch.split(mixed_ba.unsqueeze(0), [hv_tp, hv_tp], dim=-1)    # [(B, S)/dcp, hv/tp]
        # if torch.distributed.get_rank() == 0:
        #     print(f'inpx: {inpx.shape}, {inpx.dtype}, {inpx.stride()}') # [(B*S)/dcp/tp, H]
        #     print(f'mixed_ba: {mixed_ba.shape}, {mixed_ba.dtype}, {mixed_ba.stride()}', flush=True) # [(B*S)/dcp, hv/tp*2]
        #     print(f'b: {b.shape}, {b.stride()}; a: {a.shape}, {a.stride()}', flush=True)    # [(B, S)/dcp, hv/tp]
        beta = b.sigmoid()  # [(B, S)/dcp, hv/tp], bf16
        # If the model is loaded in fp16, without the .float() here, A might be -inf
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)    # [(B, S)/dcp, hv/tp], fp32
        
        # Schedule start here !!!

        # projected_states_qkvz = torch.matmul(inpx, in_proj_qkvz_w)    # [B, S, hqk*dk*2+hv*dv*2]
        # query, key, value, z, b, a = fix_query_key_value_ordering(projected_states_qkvz, projected_states_ba, hqk, hv, dk, dv)
        # # query, key: [B, S, hqk, dk]
        # # value, z: [B, S, hv, dv]
        # # b, a: [B, S, hv]
        # query, key, value = (x.reshape(x.shape[0], x.shape[1], -1) for x in (query, key, value))
        # # query, key: [B, S, hqk*dk]
        # # value: [B, S, hv*dv]
        # mixed_qkv = torch.cat((query, key, value), dim=-1)  # Copy kernel [OPT]

        # mixed_qkvz = torch.matmul(inpx, in_proj_qkvz_w)  # [B, S, hqk*dk*2+hv*dv*2]
        mixed_qkvz, _ =  self.in_proj_qkvz(inpx)    # [(B*S)/dcp/tp, hqk/tp*dk*2+hv/tp*dv*2] ---ag--> [(B*S)/dcp, hqk/tp*dk*2+hv/tp*dv*2]
        mixed_qkv, z = torch.split(mixed_qkvz.unsqueeze(0), [hqk_tp*dk*2+hv_tp*dv, hv_tp*dv], dim=-1)   # [(B, S)/dcp, ...]
        # if torch.distributed.get_rank() == 0:
        #     print(f'mixed_qkvz: {mixed_qkvz.shape}, {mixed_qkvz.stride()}')
        #     print(f'mixed_qkv: {mixed_qkv.shape}, {mixed_qkv.stride()}')
        #     print(f'z: {z.shape}, {z.stride()}', flush=True)
        mixed_qkv = mixed_qkv.transpose(1, 2)   # [B/dp, hqk/tp*dk*2+hv/tp*dv, S/cp], either dim1 or dim2 is contiguous is OK
        # # causal_conv1d module
        if causal_conv1d_fn is not None:   # True [NOTE]: how to split it along h dim?
            # assert conv1d_mod.weight.squeeze(1).shape == conv1d_weight.shape    # [hqk*dk*2+hv*dv, 4]
            mixed_qkv = causal_conv1d_fn(   # [B, hqk/tp*dk*2+hv/tp*dv(contiguous), S], [hqk/tp*dk*2+hv/tp*dv, 4] -> [B, hqk/tp*dk*2+hv/tp*dv(contiguous), S]
                x=mixed_qkv,
                # weight=conv1d_weight,   # [hqk/tp*dk*2+hv/tp*dv, 4]
                # bias=None,
                # activation='silu',
                weight=self.conv1d.weight.squeeze(1),   # [hqk/tp*dk*2+hv/tp*dv, 4]
                bias=self.conv1d.bias,  # False
                activation='silu',
                seq_idx=None,
            )
        # else:
        #     mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len])

        mixed_qkv = mixed_qkv.transpose(1, 2)   # [(B, S)/dcp, hqk/tp*dk*2+hv/tp*dv]
        query, key, value = torch.split(mixed_qkv, [hqk_tp*dk, hqk_tp*dk, hv_tp*dv], dim=-1)
        # End
        query = query.reshape(query.shape[0], query.shape[1], -1, dk)  # [(B, S)/dcp, hqk/tp, dk]
        key = key.reshape(key.shape[0], key.shape[1], -1, dk)          # [(B, S)/dcp, hqk/tp, dk]
        value = value.reshape(value.shape[0], value.shape[1], -1, dv)  # [(B, S)/dcp, hv/tp, dv]

        if hv_tp // hqk_tp > 1:    # (32/tp)//(16/tp)=2
            query = query.repeat_interleave(hv_tp // hqk_tp, dim=2)    # [(B, S)/dcp, hv/tp, dk]   # [TODO]: how to optimize?
            key = key.repeat_interleave(hv_tp // hqk_tp, dim=2)        # [(B, S)/dcp, hv/tp, dk]

        core_attn_out, _ = chunk_gated_delta_rule(
            query,
            key,
            value,  # not contiguous
            g=g,
            beta=beta,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
        )
        z_shape_og = z.shape
        # reshape input data into 2D tensor
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])  # [(B*S)/dcp*(hv/tp), dv]
        z = z.reshape(-1, z.shape[-1])  # [(B*S)/dcp*(hv/tp), dv]
        # norm module
        # print(f'torch.get_default_dtype(): {torch.get_default_dtype()}', flush=True)  # torch.bfloat16
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)   # [(B, S)/dcp, hv/tp, dv]
        # core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1)   # [(B, S)/dcp, hv/tp*dv]
        # output = torch.matmul(core_attn_out, out_proj_w)
        core_attn_out = core_attn_out.reshape(core_attn_out.shape[0] * core_attn_out.shape[1], -1)   # [(B*S)/dcp, hv/tp*dv]
        output, _ = self.out_proj(core_attn_out)   # [(B*S)/dcp/tp, H]
        # if torch.distributed.get_rank() == 0:
        #     print(f'core_attn_out: {core_attn_out.shape}, {core_attn_out.stride()}')
        #     print(f'output: {output.shape}, {output.stride()}', flush=True)
        return output

    def gdn_state_shapes_per_request(self) -> Tuple[Tuple[int], Tuple[int]]:    # DONE
        """Returns the GDN conv and ssm states shapes per request."""
        conv_states_shape = (self.conv1d.weight.shape[0], self.conv_kernel_size)
        gdn_states_shape = (self.hv_local_tp, self.dv, self.dk)
        return (conv_states_shape, gdn_states_shape)

    def forward(    # DONE
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,  # Not used in GDNLayer
        inference_context: Optional[BaseInferenceContext] = None,
        rotary_pos_emb: Optional[Tensor] = None,  # Not used in GDNLayer
        *,
        inference_params: Optional[BaseInferenceContext] = None,
    ):
        """
        Perform a forward pass through the GDN layer.

        This method implements the core computation of a GDN layer, including
        the convolution and the selective SSM/SSD.

        Args:
            hidden_states (Tensor): Input tensor of shape [s, b, h] where s is sequence length,
                b is batch size, and h is hidden size.
            attention_mask (Tensor): Mask tensor for self-attention. Not used by this layer.
            inference_context (BaseInferenceContext, optional): Parameters for inference-time
                optimizations.
            rotary_pos_emb (Tensor, optional): Rotary positional embeddings.

        Returns:
            output (Tensor): Transformed hidden states of shape [s, b, h].
        """
        # [NOTE]: Assume that B=1 !
        hidden_states = hidden_states.transpose(0, 1).reshape(-1, hidden_states.shape[-1]) # [(S, B)/dcp/tp, H] ---> [(B*S)/dcp/tp, H], assume that B=1!
        hidden_states = self.qwen3_next_attn_fwd_no_memcpy(hidden_states)   # w/ tp w/o dcp
        # [TODO] Support dcp
        hidden_states = hidden_states.unsqueeze(1) # [(B*S)/dcp/tp, H] ---> [(S, B)/dcp/tp, H], assume that B=1!
        # inference_context = deprecate_inference_params(inference_context, inference_params)

        # residual = hidden_states
        # if self.residual_in_fp32:   # False
        #     residual = residual.to(torch.float32)

        # hidden_states = hidden_states.to(dtype=self.config.params_dtype)
        # hidden_states = self.norm(hidden_states)

        # mixer_out_with_bias = self.mixer(hidden_states, inference_context=inference_context)

        # with self.bias_dropout_add_exec_handler():
        #     hidden_states = self.gdn_bda(
        #         training=self.training, fused=self.config.bias_dropout_fusion
        #     )(mixer_out_with_bias, residual, self.hidden_dropout)

        return hidden_states

    def sharded_state_dict(
        self, prefix: str = '', sharded_offsets: tuple = (), metadata: Optional[dict] = None
    ) -> ShardedStateDict:
        """
        Generate a sharded state dictionary for the gdn layer.

        Args:
            prefix (str, optional): Prefix to be added to all keys in the state dict.
            sharded_offsets (tuple, optional): Tuple of sharding offsets.
            metadata (Optional[dict], optional): Additional metadata for sharding.

        Returns:
            ShardedStateDict: A dictionary containing the sharded state of the gdn layer.
        """
        sharded_state_dict = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        prefixed_map = {
            f'{prefix}{k}': f'{prefix}{v}'
            for k, v in self.submodules_config.sharded_state_dict_keys_map.items()
        }
        if prefixed_map:
            apply_prefix_mapping(sharded_state_dict, prefixed_map)
        return sharded_state_dict

    def _te_cuda_graph_replay(self, *args, **kwargs):
        """
        CUDA graph replay for this layer and microbatch `self.current_microbatch` using TE
        interface. TransformerEngine versions>=1.10 allow keyword arguments with CUDA graph.
        However, CUDA graph accepts only Tensor inputs.
        Hence, `inference_context` is excluded from input list.
        """
        assert kwargs.get('inference_context') is None, (
            "CUDA graph accepts only Tensor inputs. inference_context is excluded from input list. "
            "For inference cuda graph, please use cuda_graph_impl=local instead."
        )
        return super()._te_cuda_graph_replay(*args, **kwargs)

    def _should_call_local_cudagraph(self, *args, **kwargs):
        """
        Check if we should call the local cudagraph path.
        """
        # Training and validation mode CUDA graphs
        if hasattr(self, 'cudagraph_manager') and kwargs.get('inference_context') is None:
            return True
        # Inference mode. CUDA graphs are used in the decode phase only, when attn mask is None
        elif not self.training and (
            hasattr(self, 'cudagraph_manager')
            and kwargs.get('attention_mask') is None
            and kwargs['inference_context'].is_decode_only()
        ):
            return True
        return False
