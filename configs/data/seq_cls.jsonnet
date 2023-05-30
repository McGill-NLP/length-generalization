local base = (import 'base.jsonnet');

base + {
    dataset+: {
        type: 'seq_classification',
        append_vocab: 'no',
        target_seq_key: 'label',
        max_source_length: 512,

        decoder_only_group_samples: false,
        decoder_only_mask_inputs: false,

        decoder_only_input_output_sep_token: '\n',
    },
}
