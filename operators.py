#
# operators.py
# ------------
# Copyright (c) 2017 Chandranath Gunjal. Available under the MIT License
#
import bisect
import random


class GAError(Exception):
    """Base class for Genetic Algorithm exceptions."""

    def __init__(self, message=''):
        self.message = message
        super().__init__(self, message)

    def __repr__(self):
        return self.message

    __str__ = __repr__


class GeneticOperator:
    """Common interface for genetic operators implemented in an Individual."""

    def __init__(self, name, func, num_parents, num_children,
                 prob_initial, prob_final, params=None):
        self.name = name
        self._func = func
        self.num_parents = num_parents
        self.num_children = num_children
        self.prob_initial = prob_initial
        self.prob_final = prob_final
        self.parameters = params

    def procreate(self, parents, children, args):
        # combine operator specific parameters with GA specific parameters
        combined = args.copy()
        if self.parameters is not None:
            combined.update(self.parameters)

        # ... and call the reproduction function
        self._func(parents, children, combined)


class OperatorManager:
    """Manage the random choosing of genetic operators for reproduction.

    A cumulative distribution of the probabilities of choosing different
    operators is updated every few generations based on adapt_interval and
    adapt_scale. This allows certain operators to be more prevalent (or
    absent) at different stages of the search.
    """

    def __init__(self, adapt_interval, adapt_scale):
        self._operators = []
        self._cdf = None

        self._next_interval = 0.0
        self._interval_step = adapt_interval

        self._curr_scale = 0.0
        self._scale_step = adapt_scale

    def register(self, op):
        self._operators.append(op)

    def rescale_probabilities(self, epoch):
        """Update the probabilities of choosing a particular operator."""
        if epoch < self._next_interval:
            return

        if (self._cdf is None) or (len(self._cdf) != len(self._operators)):
            self._cdf = [0.0] * len(self._operators)
        if self._curr_scale > 1.0:
            self._curr_scale = 1.0

        # rescale probabilities for the current interval
        p = 1.0 - self._curr_scale
        q = self._curr_scale
        for i, op in enumerate(self._operators):
            self._cdf[i] = (p * op.prob_initial) + (q * op.prob_final)

        # normalize
        total = sum(self._cdf)
        self._cdf[0] /= total
        for i in range(1, len(self._cdf)):
            self._cdf[i] = self._cdf[i - 1] + self._cdf[i] / total

        self._next_interval += self._interval_step
        self._curr_scale += self._scale_step

    def choose(self):
        """Choose an operator based on current probabilities."""
        return self._operators[bisect.bisect_left(self._cdf, random.random())]
