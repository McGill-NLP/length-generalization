local gvars = {
    seed: std.parseInt(std.extVar('APP_SEED')),
    debug_mode: false,
    dirs: {
        experiments: 'experiments',
        data: 'data',
    },
} + (import '../entity_name.json');


{
    global_vars+: gvars,
}
