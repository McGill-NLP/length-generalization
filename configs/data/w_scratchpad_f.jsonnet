{
    dataset+: {
        instance_processor+: {
            include_scratchpad: true,
        },
        max_source_length: 512,
        max_target_length: 10000,

        enable_hf_datasets_cache: true,
    },

    trainer+: {
        auto_compute_batch_size: true,
        generation_max_length: 1500,
    },
}
