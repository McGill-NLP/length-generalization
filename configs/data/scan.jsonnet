{
    dataset+: {
        name: 'scan',
        split: std.extVar('APP_DS_SPLIT'),
        train_filename: 'train.jsonl',
        validation_filename: 'validation.jsonl',
        test_filename: 'test.jsonl',

        decoder_only_input_output_sep_token: "[SEP] ",

        instance_processor+: {
            type: if std.startsWith(std.extVar('APP_DS_SPLIT'), 'mdlen_') then 'scan_bos' else 'scan',
        },
    },
}
