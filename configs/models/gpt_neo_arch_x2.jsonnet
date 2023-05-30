local base = (import 'gpt_neo_arch.jsonnet');

base + {
    model+: {
        type: 'gpt_neo',
        config: {
            type: 'gpt_neo',
            attention_layers: [
                'global',
                'local',
                'global',
                'local',
                'global',
                'local',
            ],
            attention_types: [[['global', 'local'], 3]],
            hidden_size: 64,
            num_heads: 8,
            num_layers: 6,
        },
    },
}
