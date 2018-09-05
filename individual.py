#
# individual.py
# -------------
# Copyright (c) 2017 Chandranath Gunjal. Available under the MIT License
#
import random
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from configparser import ConfigParser

from operators import GAError, GeneticOperator

# Sections in the configuration file
SECTION_FI = 'FloatIndividual'
SECTION_OSI = 'OrderedIndividual'

# String representation
NO_OPERATOR = 'RAmu'
START_OF_GENE = 'C:'
FI_GENES_PER_LINE = 5
OSI_GENES_PER_LINE = 20


class Individual(metaclass=ABCMeta):
    """Abstract Base Class for Genetic Algorithm individuals."""

    def __init__(self, details):
        self.operator = NO_OPERATOR
        self.changed = True
        self.fitness = self.adj_fitness = 0.0
        self.f1 = self.f2 = self.f3 = self.f4 = self.f5 = 0.0

        # Individuals are created from a prototype instance.
        # The prototype is initially configured by user-defined parameters
        # specified in a config file or as parameters passed in a dict
        if not isinstance(details, (ConfigParser, Individual, dict)):
            raise GAError('Individual init: incompatible blueprint')

    def reset(self):
        self.changed = True
        self.fitness = 0.0

    def copyto(self, other):
        if other is self:
            return
        if not isinstance(other, Individual):
            raise GAError('Individual copy: incompatible class')

        other.changed = self.changed
        other.fitness = self.fitness
        other.adj_fitness = self.adj_fitness
        other.operator = self.operator
        other.f1 = self.f1
        other.f2 = self.f2
        other.f3 = self.f3
        other.f4 = self.f4
        other.f5 = self.f5

    def read(self, fh):
        # return False on EOF
        s = fh.readline().strip()
        if not s:
            return False

        flds = s.split()
        if len(flds) != 9:
            raise GAError('Individual read: insufficient fields; expected 8')
        if flds[0] != START_OF_GENE:
            raise GAError('Individual read: start of gene marker not found')

        self.fitness = float(flds[1])
        self.operator = flds[2]
        self.adj_fitness = float(flds[3])
        self.f1 = float(flds[4])
        self.f2 = float(flds[5])
        self.f3 = float(flds[6])
        self.f4 = float(flds[7])
        self.f5 = float(flds[8])

        return True

    def write(self, fh):
        print(self, file=fh)
        print(file=fh)

    def __repr__(self):
        s1 = '{:s} {:0.4f}  {:4s} {:0.4f}'. \
            format(START_OF_GENE, self.fitness, self.operator,
                   self.adj_fitness)
        s2 = '   {:0.4f}  {:0.4f}  {:0.4f}  {:0.4f}  {:0.4f}'. \
            format(self.f1, self.f2, self.f3, self.f4, self.f5)
        return s1 + s2

    @abstractmethod
    def randomize(self):
        pass

    @abstractmethod
    def equal(self, other):
        pass

    @abstractmethod
    def valid(self):
        pass

    @abstractmethod
    def chromosome(self):
        pass

    @classmethod
    def get_attributes(cls):
        """Return a dictionary with required creation parameters."""
        raise GAError('Individual: get_attributes() undefined in {0}'
                      .format(cls))

    @classmethod
    def load_population(cls, filename, plist, attributes=None):
        """Load individuals from a given file.

        plist - a list of Individual instances compatible with the file being
        read. The file is read into these instances. If None, new instances
        are created using 'attributes'
        attributes - dict (as returned by get_attributes() with updated values)
        """

        # read into the given list
        n = 0
        if plist is not None:
            if not (isinstance(plist, list)
                    and len(plist) > 0
                    and all(isinstance(x, cls) for x in plist)):
                raise GAError('Individual: load_population() bad plist')

            with open(filename, 'rt') as fh:
                while n < len(plist):
                    if not plist[n].read(fh):
                        break
                    n += 1

        else:
            # create a new list and return it
            plist = []
            with open(filename, "rt") as fh:
                while True:
                    i = cls(attributes)
                    if not i.read(fh):
                        break
                    n += 1
                    plist.append(i)

        return n, plist

    @classmethod
    def save_population(cls, filename, population):
        """Save the population (list of individuals) to the given file."""
        with open(filename, "wt") as fh:
            for x in population:
                x.write(fh)

    @classmethod
    def register_operators(cls, op_manager, configparser):
        raise GAError('Individual: register_operators() undefined in {0}'
                      .format(cls))

    @staticmethod
    def _parse_probabilities(s):
        if s is None or s.strip == '':
            return 0.0, 0.0

        flds = s.split()
        if len(flds) == 1:
            a = float(flds[0])
            return a, a
        return float(flds[0]), float(flds[1])


class FloatIndividual(Individual):
    """Individual whose genes are represented as floats"""

    def __init__(self, details):
        super().__init__(details)
        self.num_genes = 0
        self.genes = None
        self.minval = None
        self.maxval = None

        # configure based on prototype
        if isinstance(details, FloatIndividual):
            self.num_genes = details.num_genes

            # share the prototype's ranges - or make a copy?
            self.minval = details.minval
            self.maxval = details.maxval

        # fetch parameters from the config file and build prototype
        elif isinstance(details, ConfigParser):
            self.num_genes = details.getint(SECTION_FI, 'fi.num_genes',
                                            fallback=0)
            if self.num_genes <= 2:
                raise GAError('FloatIndividual init: num genes <= 2')

            # range for each gene
            common_ranges = details.getboolean(SECTION_FI, 'fi.common_ranges',
                                               fallback=True)
            rlow = details.getfloat(SECTION_FI, 'fi.range_low',
                                    fallback=0.0)
            rhigh = details.getfloat(SECTION_FI, 'fi.range_high',
                                     fallback=0.0)
            rfilename = details.get(SECTION_FI, 'fi.range_filename',
                                    fallback=None)

            self.minval = [rlow] * self.num_genes
            self.maxval = [rhigh] * self.num_genes
            if common_ranges:
                # check common range for each gene
                if rlow >= rhigh:
                    raise GAError('FloatIndividual init: range low >= high')
            else:
                # specific range for each gene defined in given file
                with open(rfilename) as fh:
                    for i in range(self.num_genes):
                        # format assumed "low high"
                        flds = fh.readline().strip().split()
                        if len(flds) != 2:
                            raise GAError(
                                'FloatIndividual init: bad range file')
                        if flds[0] >= flds[1]:
                            raise GAError(
                                'FloatIndividual init: range low >= high')

                        self.minval[i] = float(flds[0])
                        self.maxval[i] = float(flds[1])

        # dictionary with parameters
        elif isinstance(details, dict):
            self.num_genes = details['fi.num_genes']
            if self.num_genes <= 2:
                raise GAError('FloatIndividual init: num genes <= 2')

            # range for each gene
            rlow = details['fi.range_low']
            rhigh = details['fi.range_high']
            if rlow >= rhigh:
                raise GAError('FloatIndividual init: range low >= high')

            self.minval = [rlow] * self.num_genes
            self.maxval = [rhigh] * self.num_genes

        # ooops!
        else:
            raise GAError('FloatIndividual init: cannot initialise')

        # create a null genotype
        self.genes = [0.0] * self.num_genes

    def copyto(self, other):
        if other is self:
            return
        if not isinstance(other, FloatIndividual):
            raise GAError('FloatIndividual copy: incompatible object class')

        super().copyto(other)
        other.num_genes = self.num_genes
        other.genes = self.genes.copy()

    def read(self, fh):
        if not super().read(fh):
            return False

        # parse genes - same as str representation
        flds = []
        for i in range(self.num_genes):
            if (i % FI_GENES_PER_LINE) == 0:
                s = fh.readline().strip()
                if not s:
                    raise GAError('FloatIndividual read: unexpected EOF')

                flds = s.split()
            self.genes[i] = float(flds.pop(0))

        # read a blank line separating each individual
        s = fh.readline().strip()
        if s != '':
            raise GAError('FloatIndividual read: expected blank line missing')
        return True

    def __repr__(self):
        s1 = super().__repr__()

        s2 = ''
        i = 0
        while i < self.num_genes:
            s2 += '  {:13.6E}'.format(self.genes[i])
            i += 1

            if (i < self.num_genes) and (i % FI_GENES_PER_LINE) == 0:
                s2 += '\n'
        return s1 + '\n' + s2

    def randomize(self):
        for i in range(self.num_genes):
            self.genes[i] = random.uniform(self.minval[i], self.maxval[i])

    def equal(self, other):
        return self.genes == other.genes

    def valid(self):
        # all genes in valid range? otherwise dump to stderr?
        for i in range(self.num_genes):
            if not (self.minval[i] <= self.genes[i] <= self.maxval[i]):
                return False
        return True

    def chromosome(self):
        # return gene (or a copy?)
        return self.genes

    @classmethod
    def get_attributes(cls):
        return {'fi.num_genes': 0, 'fi.range_low': 0.0, 'fi.range_high': 0.0}

    # Genetic operators
    @classmethod
    def register_operators(cls, op_manager, cfp):
        # Mutation operators
        p1, p2 = cls._parse_probabilities(
            cfp.get(SECTION_FI, 'fi.randomize_mutation', fallback=None))
        op_manager.register(GeneticOperator('RAmu', cls._randomize_mutation,
                                            1, 1, p1, p2))

        p1, p2 = cls._parse_probabilities(
            cfp.get(SECTION_FI, 'fi.uniform_mutation', fallback=None))
        op_manager.register(GeneticOperator('UFmu', cls._uniform_mutation,
                                            1, 1, p1, p2))

        p1, p2 = cls._parse_probabilities(
            cfp.get(SECTION_FI, 'fi.nonuniform_mutation', fallback=None))
        beta = cfp.getfloat(SECTION_FI, 'fi.nonuniform_mutation_beta',
                            fallback=3.0)
        op_manager.register(GeneticOperator('NUmu', cls._nonuniform_mutation,
                                            1, 1, p1, p2, dict(beta=beta)))

        p1, p2 = cls._parse_probabilities(
            cfp.get(SECTION_FI, 'fi.boundary_mutation', fallback=None))
        op_manager.register(GeneticOperator('BOmu', cls._boundary_mutation,
                                            1, 1, p1, p2))

        p1, p2 = cls._parse_probabilities(
            cfp.get(SECTION_FI, 'fi.creep_mutation', fallback=None))
        beta = cfp.getfloat(SECTION_FI, 'fi.creep_mutation_beta',
                            fallback=0.01)
        op_manager.register(GeneticOperator('CRmu', cls._creep_mutation,
                                            1, 1, p1, p2, dict(beta=beta)))
        # Crossover operators
        p1, p2 = cls._parse_probabilities(
            cfp.get(SECTION_FI, 'fi.single_xover', fallback=None))
        op_manager.register(GeneticOperator('SIxo', cls._single_xover,
                                            2, 2, p1, p2))

        p1, p2 = cls._parse_probabilities(
            cfp.get(SECTION_FI, 'fi.double_xover', fallback=None))
        op_manager.register(GeneticOperator('DOxo', cls._double_xover,
                                            2, 2, p1, p2))

        p1, p2 = cls._parse_probabilities(
            cfp.get(SECTION_FI, 'fi.npoint_xover', fallback=None))
        op_manager.register(GeneticOperator('NPxo', cls._npoint_xover,
                                            2, 2, p1, p2))

        # Arithmetic crossover operators
        p1, p2 = cls._parse_probabilities(
            cfp.get(SECTION_FI, 'fi.single_arith_xover', fallback=None))
        op_manager.register(GeneticOperator('SIax', cls._single_arith_xover,
                                            2, 2, p1, p2))

        p1, p2 = cls._parse_probabilities(
            cfp.get(SECTION_FI, 'fi.npoint_arith_xover', fallback=None))
        op_manager.register(GeneticOperator('NPax', cls._npoint_arith_xover,
                                            2, 2, p1, p2))

        p1, p2 = cls._parse_probabilities(
            cfp.get(SECTION_FI, 'fi.whole_arith_xover', fallback=None))
        op_manager.register(GeneticOperator('WHax', cls._whole_arith_xover,
                                            2, 2, p1, p2))

    @staticmethod
    def _randomize_mutation(parents, children, _args):
        """Randomize all genes - useful as a null hypothesis."""
        p1 = parents[0]
        c1 = children[0]
        c1.genes = p1.genes.copy()

        c1.randomize()

    @staticmethod
    def _uniform_mutation(parents, children, _args):
        """Mutate a random gene to a random value."""
        p1 = parents[0]
        c1 = children[0]
        c1.genes = p1.genes.copy()

        w = random.randrange(c1.num_genes)
        c1.genes[w] = random.uniform(c1.minval[w], c1.maxval[w])

    @staticmethod
    def _nonuniform_mutation(parents, children, args):
        """Mutate a random gene to a non-uniform random value."""
        p1 = parents[0]
        c1 = children[0]
        c1.genes = p1.genes.copy()

        w = random.randrange(c1.num_genes)
        epoch = args['epoch']
        beta = args['beta']
        delta = random.random() ** ((1.0 - epoch) ** beta)
        delta = (c1.maxval[w] - c1.minval[w]) * (1.0 - delta)
        c1.genes[w] += delta if random.random() < 0.5 else -delta

        # fix if gene has breached the valid range
        c1.genes[w] = min(max(c1.minval[w], c1.genes[w]), c1.maxval[w])

    @staticmethod
    def _boundary_mutation(parents, children, _args):
        """Mutate a random gene to within a decile of a bondary value"""
        p1 = parents[0]
        c1 = children[0]
        c1.genes = p1.genes.copy()

        w = random.randrange(c1.num_genes)
        rng = (c1.maxval[w] - c1.minval[w]) / 10.0
        if random.random() < 0.5:
            c1.genes[w] = random.uniform(c1.minval[w], c1.minval[w] + rng)
        else:
            c1.genes[w] = random.uniform(c1.maxval[w] - rng, c1.maxval[w])

    @staticmethod
    def _creep_mutation(parents, children, args):
        """Mutate a random number of genes by perturbing them slightly."""
        p1 = parents[0]
        c1 = children[0]
        c1.genes = p1.genes.copy()

        # perturb genes by a creep factor
        gamma = 1.0 + args['beta']
        prob = random.uniform(0.25, 0.5)
        for i in range(c1.num_genes):
            if random.random() > prob:
                continue

            if random.random() < 0.5:
                c1.genes[i] *= gamma
            else:
                c1.genes[i] /= gamma

            # fix if gene has breached the valid range
            c1.genes[i] = min(max(c1.minval[i], c1.genes[i]), c1.maxval[i])

    @staticmethod
    def _single_xover(parents, children, _args):
        """Crossover at a single point."""
        p1 = parents[0]
        p2 = parents[1]
        c1 = children[0]
        c2 = children[1]
        c1.genes = p1.genes.copy()
        c2.genes = p2.genes.copy()

        # potential error if num genes is less than 2
        if p1.num_genes < 2:
            return

        # crossover at an arbitrary point
        w = random.randrange(1, c1.num_genes)
        for i in range(w, c1.num_genes):
            c1.genes[i] = p2.genes[i]
            c2.genes[i] = p1.genes[i]

    @staticmethod
    def _double_xover(parents, children, _args):
        """Crossover of a random section"""
        p1 = parents[0]
        p2 = parents[1]
        c1 = children[0]
        c2 = children[1]
        c1.genes = p1.genes.copy()
        c2.genes = p2.genes.copy()

        # potential error if num genes is less than 3
        if p1.num_genes < 3:
            return

        # crossover of a random section
        w1 = random.randrange(1, c1.num_genes - 1)
        w2 = random.randrange(w1 + 1, c1.num_genes)
        for i in range(w1, w2):
            c1.genes[i] = p2.genes[i]
            c2.genes[i] = p1.genes[i]

    @staticmethod
    def _npoint_xover(parents, children, _args):
        """Crossover with each gene coming from either parent."""
        p1 = parents[0]
        p2 = parents[1]
        c1 = children[0]
        c2 = children[1]
        c1.genes = p1.genes.copy()
        c2.genes = p2.genes.copy()

        # crossover of multiple random sections
        for i in range(1, c1.num_genes):
            if random.random() >= 0.5:
                c1.genes[i] = p2.genes[i]
                c2.genes[i] = p1.genes[i]

    @staticmethod
    def _single_arith_xover(parents, children, _args):
        """Arithmetic crossover (mix) of a random gene from two parents."""
        p1 = parents[0]
        p2 = parents[1]
        c1 = children[0]
        c2 = children[1]
        c1.genes = p1.genes.copy()
        c2.genes = p2.genes.copy()

        # mix a random gene... how is this different from a mutation?
        w = random.randrange(c1.num_genes)
        if c1.genes[w] == c2.genes[w]:
            return

        x = random.random()
        y = 1.0 - x
        c1.genes[w] = x * p1.genes[w] + y * p2.genes[w]
        c2.genes[w] = x * p2.genes[w] + y * p1.genes[w]

    @staticmethod
    def _npoint_arith_xover(parents, children, _args):
        """Arithmetic crossover (mix) of multiple genes in different
        proportions.
        """
        p1 = parents[0]
        p2 = parents[1]
        c1 = children[0]
        c2 = children[1]
        c1.genes = p1.genes.copy()
        c2.genes = p2.genes.copy()

        # some genes mixed in random ratios
        for i in range(c1.num_genes):
            if random.random() >= 0.5:
                x = random.random()
                y = 1.0 - x
                c1.genes[i] = x * p1.genes[i] + y * p2.genes[i]
                c2.genes[i] = x * p2.genes[i] + y * p1.genes[i]

    @staticmethod
    def _whole_arith_xover(parents, children, _args):
        """Arithmetic crossover (mix) of all genes in the same proportion."""
        p1 = parents[0]
        p2 = parents[1]
        c1 = children[0]
        c2 = children[1]
        c1.genes = p1.genes.copy()
        c2.genes = p2.genes.copy()

        # all genes mixed in the same random ratio
        x = random.random()
        y = 1.0 - x
        for i in range(c1.num_genes):
            c1.genes[i] = x * p1.genes[i] + y * p2.genes[i]
            c2.genes[i] = x * p2.genes[i] + y * p1.genes[i]


class OrderedIndividual(Individual):
    """Individual whose genes form an ordered set"""

    def __init__(self, details):
        super().__init__(details)
        self.num_genes = 0
        self.genes = None
        self.range_low = 0

        # configure based on prototype
        if isinstance(details, OrderedIndividual):
            self.num_genes = details.num_genes
            self.range_low = details.range_low

        # fetch parameters from the config file and build prototype
        elif isinstance(details, ConfigParser):
            self.num_genes = details.getint(SECTION_OSI, 'osi.num_genes',
                                            fallback=0)
            if self.num_genes <= 2:
                raise GAError('OrderedIndividual init: num genes <= 2')

            self.range_low = details.getint(SECTION_OSI, 'osi.range_low',
                                            fallback=0)

        # dictionary with parameters
        elif isinstance(details, dict):
            self.num_genes = details['osi.num_genes']
            if self.num_genes <= 2:
                raise GAError('OrderedIndividual init: num genes <= 2')
            self.range_low = details.get('osi.range_low', 0)

        # ooops!
        else:
            raise GAError('OrderedIndividual init: cannot initialise')

        # create a null genotype
        self.genes = list(range(self.range_low,
                                self.range_low + self.num_genes))

    def copyto(self, other):
        if other is self:
            return
        if not isinstance(other, OrderedIndividual):
            raise GAError('OrderedIndividual copy: incompatible object class')

        super().copyto(other)
        other.num_genes = self.num_genes
        other.genes = self.genes.copy()

    def read(self, fh):
        if not super().read(fh):
            return False

        # parse genes - same as str representation
        flds = []
        for i in range(self.num_genes):
            if (i % OSI_GENES_PER_LINE) == 0:
                s = fh.readline().strip()
                if not s:
                    raise GAError('OrderedIndividual read: unexpected EOF')

                flds = s.split()
            self.genes[i] = int(flds.pop(0))

        # read a blank line separating each individual
        s = fh.readline().strip()
        if s != '':
            raise GAError(
                'OrderedIndividual read: expected blank line missing')

        return True

    def __repr__(self):
        s1 = super().__repr__()

        s2 = ''
        i = 0
        while i < self.num_genes:
            s2 += ' {:3d}'.format(self.genes[i])
            i += 1

            if (i < self.num_genes) and (i % OSI_GENES_PER_LINE) == 0:
                s2 += '\n'

        return s1 + '\n' + s2

    def randomize(self):
        random.shuffle(self.genes)

    def equal(self, other):
        return self.genes == other.genes

    def valid(self):
        s = set(self.genes)
        if (min(s) == self.range_low) and (len(s) == self.num_genes):
            return True
        return False

    def chromosome(self):
        # return gene (or a copy?)
        return self.genes

    @classmethod
    def get_attributes(cls):
        return {'osi.num_genes': 0, 'osi.range_low': 0}

    # Genetic operators
    @classmethod
    def register_operators(cls, op_manager, cfp):
        # Mutation operators
        p1, p2 = cls._parse_probabilities(
            cfp.get(SECTION_OSI, 'osi.randomize_mutation', fallback=None))
        op_manager.register(GeneticOperator('RAmu', cls._randomize_mutation,
                                            1, 1, p1, p2))

        p1, p2 = cls._parse_probabilities(
            cfp.get(SECTION_OSI, 'osi.position_mutation', fallback=None))
        op_manager.register(GeneticOperator('POmu', cls._position_mutation,
                                            1, 1, p1, p2))

        p1, p2 = cls._parse_probabilities(
            cfp.get(SECTION_OSI, 'osi.order_mutation', fallback=None))
        op_manager.register(GeneticOperator('ORmu', cls._order_mutation,
                                            1, 1, p1, p2))

        p1, p2 = cls._parse_probabilities(
            cfp.get(SECTION_OSI, 'osi.scramble_mutation', fallback=None))
        op_manager.register(GeneticOperator('SCmu', cls._scramble_mutation,
                                            1, 1, p1, p2))

        p1, p2 = cls._parse_probabilities(
            cfp.get(SECTION_OSI, 'osi.reverse_mutation', fallback=None))
        op_manager.register(GeneticOperator('RVmu', cls._reverse_mutation,
                                            1, 1, p1, p2))

        p1, p2 = cls._parse_probabilities(
            cfp.get(SECTION_OSI, 'osi.position_xover', fallback=None))
        op_manager.register(GeneticOperator('POxo', cls._position_xover,
                                            2, 2, p1, p2))

        p1, p2 = cls._parse_probabilities(
            cfp.get(SECTION_OSI, 'osi.order_xover', fallback=None))
        op_manager.register(GeneticOperator('ORxo', cls._order_xover,
                                            2, 2, p1, p2))

        p1, p2 = cls._parse_probabilities(
            cfp.get(SECTION_OSI, 'osi.edge_recombination_xover',
                    fallback=None))
        cyclic = cfp.getboolean(SECTION_OSI, 'osi.cyclic_path',
                                fallback=True)
        op_manager.register(GeneticOperator('ERxo',
                                            cls._edge_recombination_xover,
                                            2, 2, p1, p2, dict(cyclic=cyclic)))

    @staticmethod
    def _randomize_mutation(parents, children, _args):
        """Randomize (shuffle) all genes - useful as a null hypothesis."""
        p1 = parents[0]
        c1 = children[0]
        c1.genes = p1.genes.copy()

        c1.randomize()

    @staticmethod
    def _position_mutation(parents, children, _args):
        """Mutate by moving a random item to a random position.

        The order is maintained:
            123x456y789 => 123xy456789 or 123456xy789
        """
        p1 = parents[0]
        c1 = children[0]
        c1.genes = p1.genes.copy()

        # select two items at random...
        w1 = random.randrange(0, c1.num_genes - 1)
        w2 = random.randrange(w1 + 1, c1.num_genes)
        if random.random() < 0.5:
            # ... place the second after the first
            # 123x456y789 => 123xy456789
            c1.genes.insert(w1 + 1, c1.genes[w2])
            del c1.genes[w2 + 1]
        else:
            # ... or the first before the second
            # 123x456y789 => 123456xy789
            c1.genes.insert(w2, c1.genes[w1])
            del c1.genes[w1]

    @staticmethod
    def _order_mutation(parents, children, _args):
        """Mutate by switching the order of two items.

        The positions are maintained:
            123x456y789 => 123y456x789
        """
        p1 = parents[0]
        c1 = children[0]
        c1.genes = p1.genes.copy()

        # select two items at random - switch them
        w1 = random.randrange(0, c1.num_genes - 1)
        w2 = random.randrange(w1 + 1, c1.num_genes)
        c1.genes[w1], c1.genes[w2] = c1.genes[w2], c1.genes[w1]

    @staticmethod
    def _scramble_mutation(parents, children, _args):
        """Mutate by scrambling a section of the chromosome.

            123x456y789 => 123x645y789
        """
        p1 = parents[0]
        c1 = children[0]
        c1.genes = p1.genes.copy()

        # select two indices at random - scramble the items between them
        w1 = random.randrange(0, c1.num_genes - 1)
        w2 = random.randrange(w1 + 1, c1.num_genes)
        snippet = c1.genes[w1:w2]
        random.shuffle(snippet)
        c1.genes = c1.genes[:w1] + snippet + c1.genes[w2:]

    @staticmethod
    def _reverse_mutation(parents, children, _args):
        """Mutate by reversing a section of the chromosome.

            123x456y789 => 123x654y789
        """
        p1 = parents[0]
        c1 = children[0]
        c1.genes = p1.genes.copy()

        # select two indices at random - reverse the items between them
        w1 = random.randrange(0, c1.num_genes - 1)
        w2 = random.randrange(w1 + 1, c1.num_genes)
        c1.genes = (c1.genes[:w1] + list(reversed(c1.genes[w1:w2]))
                    + c1.genes[w2:])

    @staticmethod
    def _position_xover(parents, children, _args):
        """Crossover maintaining position from each parent

        Given
            p1 = beagfdc
            p2 = agdbfec
            m  = 1000110

        "afe" from p2 is inserted in p1 in the same position. All other items
        in p1 are shifted down in original order,
            c1 = abgdfec
            c2 = bagefdc
        """
        p1 = parents[0]
        p2 = parents[1]
        c1 = children[0]
        c2 = children[1]

        # create a mask of genes to splice
        prob = random.uniform(0.25, 0.5)
        mask = [random.random() < prob for _ in range(c1.num_genes)]
        c1.genes = _crossover_by_posn(p1.genes, p2.genes, mask)
        c2.genes = _crossover_by_posn(p2.genes, p1.genes, mask)

    @staticmethod
    def _order_xover(parents, children, _args):
        """Crossover maintaining order in each parent.

        Given
            p1 = beagfdc
            p2 = agdbfec
            m  = 1000110

        "afe" from p2 replace "eaf" in p1. All other items maintain their
        position,
            c1 = bafgedc    (afe from p2 inserted in p1)
            c2 = agbfdec    (bfd from p1 inserted in p2)
        """
        p1 = parents[0]
        p2 = parents[1]
        c1 = children[0]
        c2 = children[1]

        # create a mask of genes to splice
        prob = random.uniform(0.25, 0.5)
        mask = [random.random() < prob for _ in range(c1.num_genes)]
        c1.genes = _crossover_by_posn(p1.genes, p2.genes, mask)
        c2.genes = _crossover_by_posn(p2.genes, p1.genes, mask)

    @staticmethod
    def _edge_recombination_xover(parents, children, args):
        """Crossover based on edges i.e. pairs of vertices.

        See Wikipedia for more details.
        The 'cyclic' flag indicates if there is an edge between the first and
        last item.
        """
        p1 = parents[0]
        p2 = parents[1]
        c1 = children[0]
        c2 = children[1]

        cyclic = args['cyclic']
        c1.genes = _edge_recombine(p1.genes, p2.genes, cyclic)
        c2.genes = _edge_recombine(p2.genes, p1.genes, cyclic)


#
# helper routines for OrderedIndividual operators
#
def _crossover_by_posn(p1, p2, mask):
    g2 = [x for m, x in zip(mask, p2) if m]
    g1 = [x for x in p1 if x not in g2]

    c = [g2.pop(0) if m else g1.pop(0) for m in mask]
    return c


def _crossover_by_order(p1, p2, mask):
    g2 = [x for m, x in zip(mask, p2) if m]
    gc = g2.copy()

    c = [x if x not in g2 else gc.pop(0) for x in p1]
    return c


def _insert_in_edgemap(emap, r, cyclic):
    n = len(r)

    node = r[0]
    emap[node].add(r[1])
    if cyclic:
        emap[node].add(r[-1])

    for i in range(1, n - 1):
        node = r[i]
        emap[node].add(r[i - 1])
        emap[node].add(r[i + 1])

    node = r[n - 1]
    emap[node].add(r[n - 2])
    if cyclic:
        emap[node].add(r[0])


def _edge_recombine(p1, p2, cyclic):
    # cyclic - assume a Hamiltonian cycle with an edge between the
    # first and last elements.

    # Build the edge map.
    emap = defaultdict(set)
    _insert_in_edgemap(emap, p1, cyclic)
    _insert_in_edgemap(emap, p2, cyclic)

    # build new path
    v = p1[0]
    child = [v]
    while len(child) < len(p1):
        # remove v from the edge map
        for edge in emap.values():
            if v in edge:
                edge.discard(v)

        # order nodes from v by fewest edges
        ecount = defaultdict(list)
        for x in emap[v]:
            ecount[len(emap[x])].append(x)
        if len(ecount) > 0:
            from_nodes = ecount[sorted(ecount)[0]]
        else:
            from_nodes = list(set(p1) - set(child))
        v = random.choice(from_nodes)
        child.append(v)
    return child
