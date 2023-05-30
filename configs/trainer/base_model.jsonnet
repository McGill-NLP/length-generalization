local base = (import 'default.jsonnet');

base + {
    trainer+: {
        max_steps: 40000,
        eval_steps: 2000,

        save_total_limit: 2,

        target_batch_size: 64,
        target_eval_batch_size: 32,
        auto_compute_batch_size: true,

        per_device_train_batch_size: null,
        per_device_eval_batch_size: null,

        generation_max_length: 256,

        learning_rate: 3e-5,
        weight_decay: 0.05,
    },
}
