(import 'cbs.jsonnet')
+ {
    trainer+: {
        type: 'decoder_only',
        eval_steps: 2000,
        save_total_limit: 2,
    },
}
