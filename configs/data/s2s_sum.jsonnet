{
    dataset+: {
        name: 's2s_sum',
        split: std.extVar('APP_DS_SPLIT'),

        source_seq_key: 'source',
        target_seq_key: 'target',

        train_filename: 'train.jsonl',
        validation_filename: 'validation.jsonl',
        test_filename: 'test.jsonl',

        decoder_only_input_output_sep_token: '[SEP] ',

        instance_processor+: {
            type: 's2s_sum',
            modulo_factor: 10,
        } + (import 'minimal_template.jsonnet'),
    },
}
