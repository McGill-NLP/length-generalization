local base = (import 'base.jsonnet');


local hp_num_layers = {
    values: [3, 4, 5]
};
local hp_lr = {
    distribution: "uniform",
    min: 1,
    max: 2
};

base + {
    method: 'random',
    parameters+: {
        model+: {
            arch_config+: {
                num_layers: std.manifestJsonMinified(hp_num_layers),
            }
        },

        trainer+: {
            learning_rate: std.manifestJsonMinified(hp_lr),
        }
    }
}
