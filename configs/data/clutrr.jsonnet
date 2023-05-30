{
    dataset+: {
        name: 'clutrr',
        split: std.extVar('APP_DS_SPLIT'),
        train_filename: 'train.jsonl',
        validation_filename: 'valid.jsonl',
        test_filename: 'test.jsonl',

        instance_processor+: {
            type: 'clutrr',
        } + (import 'minimal_template.jsonnet'),

        decoder_only_input_output_sep_token: '[SEP] '
    },
}
