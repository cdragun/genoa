#
# genoa.py
# --------
# Copyright (c) 2017 Chandranath Gunjal. Available under the MIT License
#
# Implementation of a generic genetic algorithm
#
import csv
import datetime
import random
import statistics
import bisect
import sys
from configparser import ConfigParser, ExtendedInterpolation

from individual import Individual
from operators import GAError, OperatorManager

# Sections in the configuration file
SECTION_GA = 'GA'

# GA Parameters
INIT_RANDOM = 'R'
INIT_LOAD = 'L'
INIT_SEEDED = 'S'
SELECT_CROWD = 'C'
SELECT_ROULETTE = 'R'
SCALE_EVALUATED = 'E'
SCALE_LINEAR = 'L'


class GAParameters:
    def __init__(self, cfp):
        # population initialisation - Random / Seeded
        self.init_method = cfp.get(SECTION_GA, 'init.method',
                                   fallback=INIT_RANDOM).upper()
        self.init_filename = cfp.get(SECTION_GA, 'init.filename',
                                     fallback='indiv.txt')

        # run specific parameters / flags
        self.search = cfp.getboolean(SECTION_GA, 'run.search',
                                     fallback=True)
        self.max_generations = cfp.getint(SECTION_GA, 'run.max_generations',
                                          fallback=300)
        self.pop_size = cfp.getint(SECTION_GA, 'run.population_size',
                                   fallback=50)
        self.validate = cfp.getboolean(SECTION_GA, 'run.validate',
                                       fallback=True)

        self.nreplace_percent = cfp.getfloat(SECTION_GA,
                                             'run.nreplace_percent',
                                             fallback=-1.0)
        if 0 < self.nreplace_percent < 100:
            self.nreplace = int(self.pop_size * self.nreplace_percent / 100.0)
        else:
            self.nreplace = cfp.getint(SECTION_GA, 'run.nreplace',
                                       fallback=10)

        self.adapt_interval = cfp.getfloat(SECTION_GA, 'run.adapt_interval',
                                           fallback=0.1)
        self.adapt_scale = cfp.getfloat(SECTION_GA, 'run.adapt_scale',
                                        fallback=0.1)
        self.random_seed = cfp.get(SECTION_GA, 'run.random_seed',
                                   fallback=None)

        # parent selection method - Crowd / Roulette
        self.select_method = cfp.get(SECTION_GA, 'select.method',
                                     fallback=SELECT_CROWD).upper()
        self.crowd_factor = cfp.getint(SECTION_GA, 'select.crowd_factor',
                                       fallback=4)

        # fitness scaling method - Evaluated / Linear
        self.scale_method = cfp.get(SECTION_GA, 'scale.method',
                                    fallback=SCALE_EVALUATED).upper()
        self.eval_min = cfp.getfloat(SECTION_GA, 'scale.eval_min',
                                     fallback=1.0)
        self.linear_min = cfp.getfloat(SECTION_GA, 'scale.linear_min',
                                       fallback=10.0)
        self.linear_max = cfp.getfloat(SECTION_GA, 'scale.linear_max',
                                       fallback=100.0)
        self.linear_decr = cfp.getfloat(SECTION_GA, 'scale.linear_decr',
                                        fallback=1.0)

        # individual logging method - All / Best
        self.log_all_interval = cfp.getint(SECTION_GA, 'log.all.interval',
                                           fallback=100)
        self.log_all_prefix = cfp.get(SECTION_GA, 'log.all.filename_prefix',
                                      fallback='pop')
        self.log_best_filename = cfp.get(SECTION_GA, 'log.best.filename',
                                         fallback='best.log')
        self.show_progress = cfp.getboolean(SECTION_GA, 'log.progress.show',
                                            fallback=True)
        self.progress_filename = cfp.get(SECTION_GA, 'log.progress.filename',
                                         fallback='progress.log')

        # check parameters
        if self.init_method not in [INIT_RANDOM, INIT_LOAD, INIT_SEEDED]:
            raise GAError('INI file: Valid init.method = [R, L, S]')

        if self.select_method not in [SELECT_CROWD, SELECT_ROULETTE]:
            raise GAError('INI file: Valid select.method = [C, R]')
        if self.crowd_factor <= 0:
            self.crowd_factor = 4

        if self.scale_method not in [SCALE_EVALUATED, SCALE_LINEAR]:
            raise GAError('INI file: Valid scale.method = [E, L]')

        if self.log_all_interval <= 0:
            self.log_all_interval = 100

        if self.pop_size <= 0:
            raise GAError('INI file: Population size is <= 0')


class GALogger:
    def __init__(self, dumpfile_prefix, best_filename, progress_filename,
                 show_progress):
        self.tstamp = datetime.datetime.now()
        self.dumpfile_prefix = dumpfile_prefix

        # open file to log the best individual
        self.fh_best = open(best_filename, 'wt')

        # log performance statistics
        self.show_progress = show_progress
        self.fh_progress = open(progress_filename, 'w', newline='')
        self.csv = csv.writer(self.fh_progress)
        self.csv.writerow(['generation', 'timetaken',
                           'fmax', 'fmin', 'favg', 'fstd',
                           'operator', 'f1', 'f2', 'f3', 'f4', 'f5'])

    def log_best(self, goat):
        goat.write(self.fh_best)

    def log_all(self, generation, population):
        fname = '{:s}_{:04d}.log'.format(self.dumpfile_prefix, generation)
        Individual.save_population(fname, population)

    def log_progress(self, generation, fitlist, goat):
        fmin = min(fitlist)
        fmax = max(fitlist)
        favg = statistics.mean(fitlist)
        fstd = statistics.stdev(fitlist)

        # time taken per generation in millisec
        tnow = datetime.datetime.now()
        diff = (tnow - self.tstamp).total_seconds() * 1000.0
        self.tstamp = tnow

        self.csv.writerow(round(x, 2) if type(x) == float else x for x in
                          [generation, diff, fmax, fmin, favg, fstd,
                           goat.operator, goat.f1, goat.f2, goat.f3, goat.f4,
                           goat.f5])

        # show progress on stderr?
        if not self.show_progress:
            return

        s1 = 'Gen {:d}  Best: {:s}  F1-5: {:.3f} {:.3f} {:.3f} {:.3f}' \
             '{:.3f}\n'.format(generation, goat.operator, goat.f1,
                               goat.f2, goat.f3, goat.f4, goat.f5)
        s2 = 'Fitness: Max/Min: {:.3f} / {:.3f}  ' \
             'Avg/SD: {:.3f} / {:.3f}\n'.format(fmax, fmin, favg, fstd)

        print(s1, s2, file=sys.stderr)

    def shutdown(self):
        self.fh_best.close()
        self.fh_progress.close()


class GeneticAlgorithm:
    def __init__(self, configparser, indivclass, objective_func,
                 eval_mode=False):

        # load parameters from configuration file
        self._params = GAParameters(configparser)
        random.seed(self._params.random_seed)

        # Individual class - initialise prototype
        if not issubclass(indivclass, Individual):
            raise GAError('GA init: incompatible class for Individual')
        self._indivclass = indivclass
        self._prototype = indivclass(configparser)

        # override INI file search mode?
        if eval_mode:
            self._search = False
            self._max_generations = 1
        else:
            self._search = self._params.search
            self._max_generations = self._params.max_generations

        # population
        self._generation = 0
        self._population = []
        self._progeny = []
        self._objective_func = objective_func

        # reproduction - parent selection
        self._op_manager = OperatorManager(self._params.adapt_interval,
                                           self._params.adapt_scale)
        indivclass.register_operators(self._op_manager, configparser)
        self._roulette = [0.0] * self._params.pop_size

        # set up scaling/selection functions to be called
        if self._params.scale_method == SCALE_EVALUATED:
            self._scale_fitness = self._scale_evaluated
        else:
            self._scale_fitness = self._scale_linear

        if self._params.select_method == SELECT_CROWD:
            self._select_parent = self._select_from_crowd
        else:
            self._select_parent = self._select_by_roulette

        # variables for statistics
        self._sorted_order = []
        self._goat = indivclass(self._prototype)  # Greatest of all time

        # set up logger
        self._logger = GALogger(self._params.log_all_prefix,
                                self._params.log_best_filename,
                                self._params.progress_filename,
                                self._params.show_progress)

    def initialise(self):
        # create the population from the defined prototype
        self._population = [self._indivclass(self._prototype)
                            for _ in range(self._params.pop_size)]
        self._progeny = [self._indivclass(self._prototype)
                         for _ in range(self._params.nreplace + 1)]

        # initialise the population
        if self._params.init_method == INIT_RANDOM:
            for x in self._population:
                x.randomize()
        else:
            # read from file
            n, _ = self._indivclass.load_population(self._params.init_filename,
                                                    self._population)

            # seeded population? randomize the rest
            if n < self._params.pop_size:
                if self._params.init_method == INIT_LOAD:
                    raise GAError('GA load: insufficent individuals in file')
                elif self._params.init_method == INIT_SEEDED:
                    for i in range(n, self._params.pop_size):
                        self._population[i].randomize()

        self.reset_population()

    def iterate(self):
        if self._generation > self._max_generations:
            raise GAError('GA iterate: max generations exceeded')

        # evaluate new individuals from previous generation
        self._evaluate_new()

        # generate and log statistics
        self._calculate_stats()
        self._scale_fitness()

        # reproduce?
        if self._params.search:
            self._reproduce()

        # finished the run?
        self._generation += 1
        finished = (self._generation > self._max_generations)
        if finished:
            # all clean up procedures
            self._logger.shutdown()

        return not finished

    def reset_population(self):
        for x in self._population:
            x.reset()

    def best_individual(self):
        return self._goat

    def current_generation(self):
        return self._generation

    def _update_roulette(self):
        # Normalize & create a cumulative distribution on the population
        # based on the adjusted fitness.
        total = sum(x.adj_fitness for x in self._population)
        self._roulette[0] = self._population[0].adj_fitness / total
        for i in range(1, self._params.pop_size):
            self._roulette[i] = self._roulette[i - 1] \
                                + (self._population[i].adj_fitness / total)

    # Parent selection methods
    def _select_by_roulette(self):
        # probability of selection = f(adj_fitness)
        i = bisect.bisect_left(self._roulette, random.random())
        return self._population[i]

    def _select_from_crowd(self):
        # select the best adj_fitness in a random crowd
        b = self._population[random.randrange(self._params.pop_size)]
        for _ in range(1, self._params.crowd_factor):
            x = self._population[random.randrange(self._params.pop_size)]
            if x.adj_fitness > b.adj_fitness:
                b = x
        return b

    # produce the next generation and replace worst in current
    def _reproduce(self):
        # get sorted order & scale fitness
        if self._params.select_method == SELECT_ROULETTE:
            self._update_roulette()

        # normalise operator picking probablities based on generation no.
        epoch = self._generation / self._max_generations
        self._op_manager.rescale_probabilities(epoch)

        # parameters (if) needed by genetic operators
        args = dict(epoch=epoch)

        # produce progeny
        n = 0
        while n < self._params.nreplace:
            op = self._op_manager.choose()

            # choose parents
            parents = [self._select_parent() for _ in range(op.num_parents)]

            # next free slots of progeny
            children = []
            for i in range(op.num_children):
                children.append(self._progeny[n])
                children[i].operator = op.name
                n += 1

            # apply the genetic operator
            op.procreate(parents, children, args)

        # replace worst
        for i in range(self._params.nreplace):
            w = self._sorted_order[i]
            self._progeny[i].copyto(self._population[w])
            self._population[w].reset()

    # Utility functions
    def _evaluate_new(self):
        # call objective function for new individuals
        for x in self._population:
            if x.changed:
                if self._params.validate and not x.valid():
                    raise GAError('GA validate: validation failed')
                self._objective_func(x)
                x.changed = False

    # Population statistics & logging
    def _calculate_stats(self):
        # collect (fitness, index) for population
        ranking = [(x.fitness, i) for i, x in enumerate(self._population)]

        # rank in ascending fitness
        self._sorted_order = [t[1] for t in sorted(ranking)]
        flist = [t[0] for t in ranking]
        best_idx = self._sorted_order[-1]

        # keep a copy of the best individual across all generations
        if not self._population[best_idx].equal(self._goat):
            self._population[best_idx].copyto(self._goat)
            self._logger.log_best(self._goat)

        # log entire population
        if (self._generation % self._params.log_all_interval == 0) \
                or (self._generation == self._max_generations):
            self._logger.log_all(self._generation, self._population)

        # show stats for this generation
        self._logger.log_progress(self._generation, flist, self._goat)

    # Methods to scale fitness to adj_fitness
    def _scale_evaluated(self):
        min_fitness = self._population[self._sorted_order[0]].fitness
        for x in self._population:
            x.adj_fitness = x.fitness - min_fitness + self._params.eval_min

    def _scale_linear(self):
        wt = 0
        for i in reversed(self._sorted_order):
            self._population[i].adj_fitness = \
                max(self._params.linear_max - (wt * self._params.linear_decr),
                    self._params.linear_min)
            wt += 1


#
# Utility functions
# -----------------
def load_config_parser(filename):
    cfg = ConfigParser(interpolation=ExtendedInterpolation())
    cfg.read(filename)
    return cfg
