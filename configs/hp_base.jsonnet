(import 'base.jsonnet')
+ {
    trainer+: {
        seed: $.global_vars.seed,
    },
    config_filenames: [],
    sweep_run: false,
}
