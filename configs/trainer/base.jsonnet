{
    trainer+: {
        evaluation_strategy: 'steps',
        logging_strategy: 'steps',
        save_strategy: 'steps',

        max_steps: 100,
        eval_steps: 10,
        logging_steps: 10,

        save_steps: self.eval_steps,
        save_total_limit: 5,

        seed: $.global_vars.seed,

        per_device_train_batch_size: 8,
        per_device_eval_batch_size: 8,

        generation_max_length: 100,
        generation_num_beams: 1,

        predict_with_generate: true,

        learning_rate: 2,
        weight_decay: 0,

        lr_scheduler_type: 'constant',
    },
}
