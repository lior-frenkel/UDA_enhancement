"""Gradient Reversal Layer for Domain-Adversarial Neural Networks (DANN).

Implements the gradient reversal trick from:
    Ganin et al., "Domain-Adversarial Training of Neural Networks", JMLR 2016.

Forward pass: identity function.
Backward pass: negates and scales the gradient by alpha.
"""

from torch.autograd import Function


class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x, alpha)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        _, alpha = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = -alpha * grad_output
        return grad_input, None


revgrad = GradientReversalFunction.apply
