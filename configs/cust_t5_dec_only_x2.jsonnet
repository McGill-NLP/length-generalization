(import 'base.jsonnet')
+ (import 'models/custom_t5_base_dec_only.jsonnet')
+ (import 'tokenizers/whitespace.jsonnet')
+ (import 'trainer/small_model.jsonnet')
+ (import 'data/seq2seq.jsonnet')
+ {
    global_vars+: {
        debug_mode: false,
    },
    dataset+: {
        is_decoder_only: true,
        decoder_only_block_size: 128,
        decoder_only_group_samples: false,
        decoder_only_mask_inputs: true,
        decoder_only_padding_side: 'right',
        decoder_only_include_position_ids: false,
    },
    trainer+: {
        type: 'decoder_only',
    },
}
