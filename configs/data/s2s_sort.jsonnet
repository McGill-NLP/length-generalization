{
    dataset+: {
        name: 's2s_sort',
        split: std.extVar('APP_DS_SPLIT'),
        train_filename: 'train.jsonl',
        validation_filename: 'validation.jsonl',
        test_filename: 'test.jsonl',
        decoder_only_input_output_sep_token: '',
        instance_processor+: {
            type: if std.startsWith(std.extVar('APP_DS_SPLIT'), 'len_sngd_') then 's2s_sort_single_digit' else 's2s_sort_multi_digit',
        } + (import 'minimal_template.jsonnet'),
        answer_index_in_label: -1,
    },
}
