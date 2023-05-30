{
    model+: {
        type: 'custom_decoder_only_t5_for_seq_cls',
        config+: {
            type: 'seq2seq_t5',
        },
        from_pretrained: false,
    },
}
