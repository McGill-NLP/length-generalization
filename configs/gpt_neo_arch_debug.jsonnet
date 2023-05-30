(import 'base.jsonnet')
+ (import 'models/gpt_neo_arch_debug.jsonnet')
+ (import 'tokenizers/whitespace.jsonnet')
+ (import 'trainer/debug.jsonnet')
+ (import 'data/seq2seq.jsonnet')
+ {
    global_vars+: {
        debug_mode: true,
    },
    dataset+: {
        num_proc: 1,
        is_decoder_only: true,
        decoder_only_block_size: 128,
        decoder_only_group_samples: false,
        decoder_only_mask_inputs: true,
    },
    trainer+: {
        type: 'decoder_only',
        learning_rate: 0.01,
        per_device_eval_batch_size: 4,
    },
}
