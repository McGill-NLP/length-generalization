local base = (import 't5a.jsonnet');

base + {
    model+: {
        config+: {
            d_ff: 256,
            d_model: 64,
            num_heads: 8,
            num_layers: 2,
        },
    },
}
