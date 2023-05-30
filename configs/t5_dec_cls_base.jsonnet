(import 'base.jsonnet')
+ (import 'models/custom_t5_base_dec_only_for_seq_cls.jsonnet')
+ (import 'models/t5_base.jsonnet')
+ (import 'tokenizers/pretrained_fast.jsonnet')
+ (import 'trainer/base_model.jsonnet')
+ (import 'data/seq_cls.jsonnet')
+ (import 'seq_classification.jsonnet')
+ {
    global_vars+: {
        debug_mode: false,
    },
    dataset+: {
        is_encoder_only: false,
        is_decoder_only: true,
    },
    model+: {
        from_pretrained: false,
    },
    trainer+: {
        type: 'trainer_with_metrics',
        metric_for_best_model: if std.endsWith($.dataset.name, '_mod') then 'accuracy' else 'mse',
    },
    analyzers: [
        (import 'analyzers/seq_cls_analyzer.jsonnet'),
    ],
}
