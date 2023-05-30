{
    dataset+: {
        name: 'sum',
        split: std.extVar('APP_DS_SPLIT'),
        source_seq_key: 'source',
        target_seq_key: 'target',

        train_filename: 'train.jsonl',
        validation_filename: 'validation.jsonl',
        test_filename: 'test.jsonl',

        input_prompt: 'source',
        is_regression: true,
        truncate_source: false,

        instance_processor+: {
            type: 'sum',
        },
    },
}
