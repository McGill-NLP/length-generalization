local base = (import 'default.jsonnet');

base + {
    trainer+: {
        max_steps: 50000,
        eval_steps: 1000,

        save_total_limit: 5,

        per_device_train_batch_size: 128,
        per_device_eval_batch_size: 128,

        learning_rate: 0.003,
        weight_decay: 0.05,
    },
}
