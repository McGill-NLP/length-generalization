import math
from typing import Optional


import torch
from torch.nn import Embedding

from common import ConfigurationError
from modules.span_extractors.span_extractor import SpanExtractor


def bucket_values(
    distances: torch.Tensor, num_identity_buckets: int = 4, num_total_buckets: int = 10
) -> torch.Tensor:
    """
    Places the given values (designed for distances) into `num_total_buckets`semi-logscale
    buckets, with `num_identity_buckets` of these capturing single values.
    The default settings will bucket values into the following buckets:
    [0, 1, 2, 3, 4, 5-7, 8-15, 16-31, 32-63, 64+].
    # Parameters
    distances : `torch.Tensor`, required.
        A Tensor of any size, to be bucketed.
    num_identity_buckets: `int`, optional (default = `4`).
        The number of identity buckets (those only holding a single value).
    num_total_buckets : `int`, (default = `10`)
        The total number of buckets to bucket values into.
    # Returns
    `torch.Tensor`
        A tensor of the same shape as the input, containing the indices of the buckets
        the values were placed in.
    """
    # Chunk the values into semi-logscale buckets using .floor().
    # This is a semi-logscale bucketing because we divide by log(2) after taking the log.
    # We do this to make the buckets more granular in the initial range, where we expect
    # most values to fall. We then add (num_identity_buckets - 1) because we want these indices
    # to start _after_ the fixed number of buckets which we specified would only hold single values.
    logspace_index = (distances.float().log() / math.log(2)).floor().long() + (
        num_identity_buckets - 1
    )
    # create a mask for values which will go into single number buckets (i.e not a range).
    use_identity_mask = (distances <= num_identity_buckets).long()
    use_buckets_mask = 1 + (-1 * use_identity_mask)
    # Use the original values if they are less than num_identity_buckets, otherwise
    # use the logspace indices.
    combined_index = use_identity_mask * distances + use_buckets_mask * logspace_index
    # Clamp to put anything > num_total_buckets into the final bucket.
    return combined_index.clamp(0, num_total_buckets - 1)


class SpanExtractorWithSpanWidthEmbedding(SpanExtractor):
    """
    `SpanExtractorWithSpanWidthEmbedding` implements some common code for span
    extractors which will need to embed span width.
    Specifically, we initiate the span width embedding matrix and other
    attributes in `__init__`, leave an `_embed_spans` method that can be
    implemented to compute span embeddings in different ways, and in `forward`
    we concatenate span embeddings returned by `_embed_spans` with span width
    embeddings to form the final span representations.
    We keep SpanExtractor as a purely abstract base class, just in case someone
    wants to build a totally different span extractor.
    # Parameters
    input_dim : `int`, required.
        The final dimension of the `sequence_tensor`.
    num_width_embeddings : `int`, optional (default = `None`).
        Specifies the number of buckets to use when representing
        span width features.
    span_width_embedding_dim : `int`, optional (default = `None`).
        The embedding size for the span_width features.
    bucket_widths : `bool`, optional (default = `False`).
        Whether to bucket the span widths into log-space buckets. If `False`,
        the raw span widths are used.
    # Returns
    span_embeddings : `torch.FloatTensor`.
        A tensor of shape `(batch_size, num_spans, embedded_span_size)`,
        where `embedded_span_size` depends on the way spans are represented.
    """

    def __init__(
        self,
        input_dim: int,
        num_width_embeddings: int = None,
        span_width_embedding_dim: int = None,
        bucket_widths: bool = False,
    ) -> None:
        super().__init__()
        self._input_dim = input_dim
        self._num_width_embeddings = num_width_embeddings
        self._bucket_widths = bucket_widths

        self._span_width_embedding: Optional[Embedding] = None
        if num_width_embeddings is not None and span_width_embedding_dim is not None:
            self._span_width_embedding = Embedding(
                num_embeddings=num_width_embeddings,
                embedding_dim=span_width_embedding_dim,
            )
        elif num_width_embeddings is not None or span_width_embedding_dim is not None:
            raise ConfigurationError(
                "To use a span width embedding representation, you must"
                "specify both num_width_embeddings and span_width_embedding_dim."
            )

    def forward(
        self,
        sequence_tensor: torch.FloatTensor,
        span_indices: torch.LongTensor,
        sequence_mask: torch.BoolTensor = None,
        span_indices_mask: torch.BoolTensor = None,
    ):
        """
        Given a sequence tensor, extract spans, concatenate width embeddings
        when need and return representations of them.
        # Parameters
        sequence_tensor : `torch.FloatTensor`, required.
            A tensor of shape (batch_size, sequence_length, embedding_size)
            representing an embedded sequence of words.
        span_indices : `torch.LongTensor`, required.
            A tensor of shape `(batch_size, num_spans, 2)`, where the last
            dimension represents the inclusive start and end indices of the
            span to be extracted from the `sequence_tensor`.
        sequence_mask : `torch.BoolTensor`, optional (default = `None`).
            A tensor of shape (batch_size, sequence_length) representing padded
            elements of the sequence.
        span_indices_mask : `torch.BoolTensor`, optional (default = `None`).
            A tensor of shape (batch_size, num_spans) representing the valid
            spans in the `indices` tensor. This mask is optional because
            sometimes it's easier to worry about masking after calling this
            function, rather than passing a mask directly.
        # Returns
        A tensor of shape `(batch_size, num_spans, embedded_span_size)`,
        where `embedded_span_size` depends on the way spans are represented.
        """
        # shape (batch_size, num_spans, embedding_dim)
        span_embeddings = self._embed_spans(
            sequence_tensor, span_indices, sequence_mask, span_indices_mask
        )
        if self._span_width_embedding is not None:
            # width = end_index - start_index + 1 since `SpanField` use inclusive indices.
            # But here we do not add 1 beacuse we often initiate the span width
            # embedding matrix with `num_width_embeddings = max_span_width`
            # shape (batch_size, num_spans)
            widths_minus_one = span_indices[..., 1] - span_indices[..., 0]

            if self._bucket_widths:
                widths_minus_one = bucket_values(
                    widths_minus_one, num_total_buckets=self._num_width_embeddings  # type: ignore
                )

            # Embed the span widths and concatenate to the rest of the representations.
            span_width_embeddings = self._span_width_embedding(widths_minus_one)
            span_embeddings = torch.cat([span_embeddings, span_width_embeddings], -1)

        if span_indices_mask is not None:
            # Here we are masking the spans which were originally passed in as padding.
            return span_embeddings * span_indices_mask.unsqueeze(-1)

        return span_embeddings

    def get_input_dim(self) -> int:
        return self._input_dim

    def _embed_spans(
        self,
        sequence_tensor: torch.FloatTensor,
        span_indices: torch.LongTensor,
        sequence_mask: torch.BoolTensor = None,
        span_indices_mask: torch.BoolTensor = None,
    ) -> torch.Tensor:
        """
        Returns the span embeddings computed in many different ways.
        """
        raise NotImplementedError
