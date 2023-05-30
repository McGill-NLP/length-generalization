(import 'base.jsonnet')
+ (import 'models/custom_t5_base_dec_only.jsonnet')
+ (import 'models/t5_small.jsonnet')
+ (import 'tokenizers/pretrained.jsonnet')
+ (import 'trainer/small_model.jsonnet')
+ (import 'data/seq2seq.jsonnet')
+ {
    global_vars+: {
        debug_mode: false,
    },
    dataset+: {
        is_decoder_only: true,
        decoder_only_block_size: 128,
        decoder_only_group_samples: false,
        decoder_only_mask_inputs: true,
        decoder_only_padding_side: 'right',
        decoder_only_include_position_ids: false,
    },
    trainer+: {
        type: 'decoder_only',
        evaluation_strategy: 'steps',
        eval_steps: 1,
        per_device_train_batch_size: 4,
        per_device_eval_batch_size: 4,
        target_batch_size: 4,
        target_eval_batch_size: 4,
        callbacks: [
            //            (import 'trainer/callbacks/save_predictions.jsonnet'),
            (import 'trainer/callbacks/cartography_measure_valid.jsonnet') + { label_index: $.dataset.answer_index_in_label },
        ],
    },

    analyzers: [
        (import 'analyzers/s2s_analyzer.jsonnet'),
    ],
}
