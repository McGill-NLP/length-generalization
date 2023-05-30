local base = (import 'base.jsonnet');

base + {
    trainer+: {
        max_steps: 200,
        eval_steps: 100,
        logging_steps: 2,

        metric_for_best_model: "seq_acc",

        save_steps: self.eval_steps,
        save_total_limit: 1,

        per_device_train_batch_size: 10,
        per_device_eval_batch_size: 128,

        warmup_steps: 2,

        dataloader_num_workers: 4,
        dataloader_pin_memory: true,
    },
}
