{
    dataset+: {
        name: 'sc_count_mod',
        split: std.extVar('APP_DS_SPLIT'),
        source_seq_key: 'source',
        target_seq_key: 'target',

        train_filename: 'train.jsonl',
        validation_filename: 'validation.jsonl',
        test_filename: 'test.jsonl',

        input_prompt: "source",
        is_regression: false,
        truncate_source: false,

        label_list: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],

        instance_processor+: {
            type: 'count_mod',
        },
    },
}

