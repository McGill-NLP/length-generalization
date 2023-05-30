{
    dataset+: {
        name: 's2s_addition',
        split: std.extVar('APP_DS_SPLIT'),
        train_filename: 'train.jsonl',
        validation_filename: 'validation.jsonl',
        test_filename: 'test.jsonl',
        decoder_only_input_output_sep_token: '',
        instance_processor+: {
            type: 's2s_addition',
        } + (import 'minimal_template.jsonnet'),
        answer_index_in_label: -1,
    },
}
