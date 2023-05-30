local base = (import 'base.jsonnet');

base + {
    model+: {
        type: "seq2seq_t5",
        config+: {
            type: "seq2seq_t5",
            d_ff: 1048,
            d_kv: std.floor(self.d_model / self.num_heads),
            d_model: 256,
            decoder_start_token_id: 0,
            dropout_rate: 0.1,
            eos_token_id: 1,
            feed_forward_proj: 'relu',
            initializer_factor: 1.0,
            is_encoder_decoder: true,
            layer_norm_epsilon: 1e-06,
            model_type: 't5',
            n_positions: 512,
            num_decoder_layers: self.num_layers,
            num_heads: 8,
            num_layers: 3,
            output_past: true,
            pad_token_id: 0,
            relative_attention_num_buckets: 32,
            use_cache: true,
        },
    },
}