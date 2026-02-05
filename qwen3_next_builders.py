# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

from model_provider import count_parameters_in_layer
from megatron.core.models.qwen3_next import Qwen3NextModel
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.spec_utils import import_module
from megatron.training import print_rank_0
from megatron.training.arguments import core_transformer_config_from_args

def qwen3_next_builder(args, pre_process, post_process, vp_stage=None, config=None, pg_collection=None):    # TODO
    '''
    Output: model
    '''
    print_rank_0('building Qwen3-Next model ...')
    if config is None:  # True
        config = core_transformer_config_from_args(args, TransformerConfig)
    assert args.use_legacy_models is False, "Qwen3-Next only supported in Mcore!"

    if config.transformer_impl == "inference_optimized":    # False; 'transformer_engine'
        from megatron.core.models.qwen3_next.qwen3_next_layer_specs import qwen3_next_inference_stack_spec
        qwen3_next_stack_spec = qwen3_next_inference_stack_spec 
    elif args.spec is not None: # True
        qwen3_next_stack_spec = import_module(args.spec)    # [TODO] modify args.spec to adapt for qwen3_next
    else:
        raise ValueError("You must provide a valid Qwen3-Next layer spec via --spec")

    model = Qwen3NextModel(
        config=config,
        qwen3_next_stack_spec=qwen3_next_stack_spec,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        hybrid_attention_ratio=args.hybrid_attention_ratio,     # Useless
        hybrid_mlp_ratio=args.hybrid_mlp_ratio,                 # Useless
        hybrid_override_pattern=args.hybrid_override_pattern,   # Useful
        post_process=post_process,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=True,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type,
        rotary_percent=args.rotary_percent,
        rotary_base=args.rotary_base,
        pg_collection=pg_collection,
    )

    for l in range(model.decoder.num_layers_per_pipeline_rank):
        layer_params = count_parameters_in_layer(model, f'decoder.layers.{l}.')
        print_rank_0(f" == params layer {l}: {layer_params}")

    return model
