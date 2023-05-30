{
    model+: {
        type: 'custom_decoder_only_t5',
        config+: {
            type: 'seq2seq_t5',
        },
        from_pretrained: false,
    },
}
