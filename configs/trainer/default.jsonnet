local base = (import 'base.jsonnet');

base + {
    trainer+: {
        max_steps: 50000,
        eval_steps: 5000,
        logging_steps: 20,

        metric_for_best_model: "seq_acc",

        save_total_limit: 5,

        per_device_train_batch_size: 256,
        per_device_eval_batch_size: 256,

        generation_max_length: 256,

        warmup_ratio: 0.06,

        lr_scheduler_type: 'polynomial',

        learning_rate: 0.0003,
        weight_decay: 0,

        dataloader_num_workers: 4,
        dataloader_pin_memory: true,
    },
}
