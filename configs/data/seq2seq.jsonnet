local base = (import 'base.jsonnet');

base + {
    dataset+: {
        type: 'seq2seq',
        source_seq_key: 'source',
        target_seq_key: 'target',
        append_vocab: 'no',
        max_source_length: 256,
        max_target_length: 256,

        is_decoder_only: false,
        decoder_only_input_output_sep_token:  "<sep>",
        decoder_only_block_size: 1024,
        decoder_only_group_samples: false,
        decoder_only_mask_inputs: true,
    },
}
