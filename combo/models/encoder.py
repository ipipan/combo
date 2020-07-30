"""Encoder."""
from typing import Optional, Tuple

import torch
import torch.nn.utils.rnn as rnn
from allennlp import common, modules
from allennlp.modules import input_variational_dropout, stacked_bidirectional_lstm, seq2seq_encoders
from overrides import overrides


class StackedBiLSTM(stacked_bidirectional_lstm.StackedBidirectionalLstm, common.FromParams):

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, recurrent_dropout_probability: float,
                 layer_dropout_probability: float, use_highway: bool = False):
        super().__init__(input_size=input_size,
                         hidden_size=hidden_size,
                         num_layers=num_layers,
                         recurrent_dropout_probability=recurrent_dropout_probability,
                         layer_dropout_probability=layer_dropout_probability,
                         use_highway=use_highway)

    @overrides
    def forward(self,
                inputs: rnn.PackedSequence,
                initial_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                ) -> Tuple[rnn.PackedSequence, Tuple[torch.Tensor, torch.Tensor]]:
        """Changes when compared to stacked_bidirectional_lstm.StackedBidirectionalLstm
        * dropout also on last layer
        * accepts BxTxD tensor
        * state from n-1 layer used as n layer initial state

        :param inputs:
        :param initial_state:
        :return:
        """
        output_sequence = inputs
        state_fwd = None
        state_bwd = None
        for i in range(self.num_layers):
            forward_layer = getattr(self, f"forward_layer_{i}")
            backward_layer = getattr(self, f"backward_layer_{i}")

            forward_output, state_fwd = forward_layer(output_sequence, state_fwd)
            backward_output, state_bwd = backward_layer(output_sequence, state_bwd)

            forward_output, lengths = rnn.pad_packed_sequence(forward_output, batch_first=True)
            backward_output, _ = rnn.pad_packed_sequence(backward_output, batch_first=True)

            output_sequence = torch.cat([forward_output, backward_output], -1)

            output_sequence = self.layer_dropout(output_sequence)
            output_sequence = rnn.pack_padded_sequence(output_sequence, lengths, batch_first=True)

        return output_sequence, (state_fwd, state_bwd)


@modules.Seq2SeqEncoder.register("combo_encoder")
class ComboEncoder(seq2seq_encoders.PytorchSeq2SeqWrapper):
    """COMBO encoder (https://www.aclweb.org/anthology/K18-2004.pdf).

    This implementation uses Variational Dropout on the input and then outputs of each BiLSTM layer
    (instead of used Gaussian Dropout and Gaussian Noise).
    """

    def __init__(self, stacked_bilstm: StackedBiLSTM, layer_dropout_probability: float):
        super().__init__(stacked_bilstm, stateful=False)
        self.layer_dropout = input_variational_dropout.InputVariationalDropout(p=layer_dropout_probability)

    @overrides
    def forward(self,
                inputs: torch.Tensor,
                mask: torch.BoolTensor,
                hidden_state: torch.Tensor = None) -> torch.Tensor:
        x = self.layer_dropout(inputs)
        x = super().forward(x, mask)
        return self.layer_dropout(x)
