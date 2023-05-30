(import 'base.jsonnet')
+ (import 'models/gpt2_arch_debug.jsonnet')
+ (import 'tokenizers/pretrained.jsonnet')
+ (import 'trainer/debug.jsonnet')
+ (import 'data/seq_cls.jsonnet')
+ (import 'seq_classification.jsonnet')
+ {
    global_vars+: {
        debug_mode: true,
    },
    dataset+: {
        num_proc: 1,
        is_decoder_only: true,
    },
    trainer+: {
        type: 'trainer_with_metrics',
        learning_rate: 0.01,
        per_device_eval_batch_size: 4,
        metric_for_best_model: 'accuracy',
    },
    model+: {
        type: 'gpt2_seq_classifier',
        hf_model_name: 'gpt2',
    },
}
