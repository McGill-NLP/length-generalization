{
    dataset+: {
        name: 'pcfg',
        split: std.extVar('APP_DS_SPLIT'),
        train_filename: 'train.jsonl',
        validation_filename: 'validation.jsonl',
        test_filename: 'test.jsonl',

        decoder_only_input_output_sep_token: "[SEP] ",

        instance_processor+: {
            type: if std.startsWith(std.extVar('APP_DS_SPLIT'), 'md_productivity') then 'pcfg_bos' else 'identity',
        },

        append_vocab: 'src',
    },


    trainer+: {
        max_steps: 100000,
    }
}
