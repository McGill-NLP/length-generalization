
local base = (import 'base.jsonnet');


local hp_lr = {
    values: [0.00001, 0.00003],
};

local hp_weight_decay = {
    values: [0.1],
};

base + {
    method: 'grid',
    metric: {
        goal: 'maximize',
        name: 'pred/valid_seq_acc',
    },
    parameters+: {
        trainer+: {
            learning_rate: std.manifestJsonMinified(hp_lr),
            weight_decay: std.manifestJsonMinified(hp_weight_decay),
        }
    },
}
