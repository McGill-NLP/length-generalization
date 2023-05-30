local base = (import 'gpt2_arch.jsonnet');

base + {
    model+: {
        type: 'gpt2',
        config+: {
            type: 'gpt2',
            n_embd: 64,
            n_head: 8,
            n_inner: null,
            n_layer: 3,
        },
    },
}
