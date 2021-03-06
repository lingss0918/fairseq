# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn.functional as F

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion


@register_criterion('decoar')
class DeCoArCriterion(FairseqCriterion):

    def __init__(self, task, infonce=False, loss_weights=None, log_keys=None):
        super().__init__(task)
        self.infonce = infonce
        self.loss_weights = eval(loss_weights)
        self.log_keys = [] if log_keys is None else eval(log_keys)
        self.count = 0

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--infonce', action='store_true',
                            help='if set, uses cross entropy instead of binary cross entropy (i.e. InfoNCE loss)')
        parser.add_argument('--loss-weights', type=str, default=None,
                            help='weights for additional loss terms (not first one)')
        parser.add_argument('--log-keys', type=str, default=None,
                            help='output keys to log')

    def forward(self, model, sample, reduce=True, log_pred=False):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        logits, recon = model.get_logits(net_output)

        losses = []

        loss = F.l1_loss(logits, recon, reduction="none").sum(axis=(0,1)).mean()
        sample_size = logits.shape[0] * logits.shape[1]
        losses.append(loss)

        if self.loss_weights != 0:
            extra_losses = model.get_extra_losses(net_output)
            extra_losses = self.loss_weights * extra_losses.float() * sample_size
            loss += extra_losses
            losses.append(extra_losses)
            self.loss_weights = self.loss_weights
            

        logging_output = {
            'loss': loss.item() if reduce else loss,
            'ntokens': sample_size,
            'nsentences': sample['id'].numel(),
            'sample_size': sample_size,
            'loss_weights': self.loss_weights,
        }

        for lk in self.log_keys:
            if lk in net_output:
                logging_output[lk] = float((net_output[lk]))

        if len(losses) > 1:
            for i, l in enumerate(losses):
                logging_output[f'loss_{i}'] = l.item()

        if log_pred:
            logging_output['logits'] = logits.cpu().numpy()
            logging_output['target'] = target.cpu().numpy()
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = utils.item(sum(log.get('loss', 0) for log in logging_outputs))
        ntokens = utils.item(sum(log.get('ntokens', 0) for log in logging_outputs))
        nsentences = utils.item(sum(log.get('nsentences', 0) for log in logging_outputs))
        sample_size = utils.item(sum(log.get('sample_size', 0) for log in logging_outputs))

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('ntokens', ntokens)
        metrics.log_scalar('nsentences', nsentences)

        correct = sum(log.get("correct", 0) for log in logging_outputs)
        metrics.log_scalar("_correct", correct)

        total = sum(log.get("count", 0) for log in logging_outputs)
        metrics.log_scalar("_total", total)


        if total > 0:
            metrics.log_derived(
                "accuracy",
                lambda meters: round(meters["_correct"].sum / meters["_total"].sum, 5)
                if meters["_total"].sum > 0
                else float("nan"),
            )

        builtin_keys = {'loss', 'ntokens', 'nsentences', 'sample_size', 'correct', 'count'}

        for k in logging_outputs[0]:
            if k not in builtin_keys:
                val = sum(log.get(k, 0) for log in logging_outputs) / len(logging_outputs)
                if k.startswith('loss'):
                    metrics.log_scalar(k, val / sample_size / math.log(2), sample_size)
                else:
                    metrics.log_scalar(k, val, round=3)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return False
