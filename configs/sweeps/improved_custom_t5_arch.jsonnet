
local base = (import 'base.jsonnet');


local hp_lr = {
    values: [0.00001, 0.00003, 0.00005, 0.0001],
};

local hp_weight_decay = {
    values: [0.1, 0.15],
};

local hp_warmup_ratio = {
    values: [0.06, 0.15],
};

local hp_num_layers = {
    values: [9, 12],
};

local hp_train_batch_size = {
    values: [128],
};

base + {
    method: 'random',
    metric: {
        goal: 'maximize',
        name: 'pred/valid_seq_acc',
    },
    parameters+: {
        trainer+: {
            learning_rate: std.manifestJsonMinified(hp_lr),
            weight_decay: std.manifestJsonMinified(hp_weight_decay),
            warmup_ratio: std.manifestJsonMinified(hp_warmup_ratio),
            per_device_train_batch_size: std.manifestJsonMinified(hp_train_batch_size),
        },
        model+: {
            config+: {
                num_layers: std.manifestJsonMinified(hp_num_layers),
            }
        }
    },
}
