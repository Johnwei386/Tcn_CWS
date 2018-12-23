import sys
import time
import numpy as np
from collections import defaultdict, Counter, OrderedDict


class ConfusionMatrix(object):
    """
    A confusion matrix stores counts of (true, guessed) labels, used to
    compute several evaluation metrics like accuracy, precision, recall
    and F1.
    """

    def __init__(self, labels, default_label=None):
        self.labels = labels
        self.default_label = default_label if default_label is not None else len(labels) -1
        self.counts = defaultdict(Counter)

    def update(self, gold, guess):
        """Update counts"""
        # defaultdict(<class 'collections.Counter'>, {4: Counter({1: 3, 2: 3}), 6: Counter({1: 4})})
        self.counts[gold][guess] += 1

    def to_table(self, data, row_labels, column_labels, precision=2, digits=4):
        """Pretty print tables.
        Assumes @data is a 2D array and uses @row_labels and @column_labels
        to display table.
        """
        # Convert data to strings
        line = "%0" + str(digits) + "." + str(precision) + "f"
        data = [[line % v for v in row] for row in data]
        cell_width = max(
            max(map(len, row_labels)),
            max(map(len, column_labels)),
            max(max(map(len, row)) for row in data))

        def c(s):
            """adjust cell output"""
            return s + " " * (cell_width - len(s))

        ret = ""
        ret += "\t".join(map(c, column_labels)) + "\n"
        for l, row in zip(row_labels, data):
            ret += "\t".join(map(c, [l] + row)) + "\n"
        return ret

    def as_table(self):
        """Print tables"""
        # Header
        data = [[self.counts[l][l_] for l_,_ in enumerate(self.labels)] for l,_ in enumerate(self.labels)]
        return self.to_table(data, self.labels, ["go\\gu"] + self.labels, precision=0, digits=0)

    def summary(self, quiet=False):
        """Summarize counts"""
        keys = range(len(self.labels))
        data = []
        macro = np.array([0., 0., 0., 0.])
        micro = np.array([0., 0., 0., 0.])
        default = np.array([0., 0., 0., 0.])
        for l in keys:
            tp = self.counts[l][l]
            fp = sum(self.counts[l_][l] for l_ in keys if l_ != l)
            tn = sum(self.counts[l_][l__] for l_ in keys if l_ != l for l__ in keys if l__ != l)
            fn = sum(self.counts[l][l_] for l_ in keys if l_ != l)

            acc = (tp + tn)/(tp + tn + fp + fn) if tp > 0 else 0
            prec = (tp)/(tp + fp) if tp > 0 else 0
            rec = (tp)/(tp + fn) if tp > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if tp > 0 else 0

            # update micro/macro averages
            micro += np.array([tp, fp, tn, fn])
            macro += np.array([acc, prec, rec, f1])
            if l != self.default_label:  # Count count for everything that is not the default label!
                default += np.array([tp, fp, tn, fn])

            data.append([acc, prec, rec, f1])

        # micro average
        tp, fp, tn, fn = micro
        acc = (tp + tn)/(tp + tn + fp + fn) if tp > 0 else 0
        prec = (tp)/(tp + fp) if tp > 0 else 0
        rec = (tp)/(tp + fn) if tp > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if tp > 0 else 0
        data.append([acc, prec, rec, f1])
        # Macro average
        data.append(macro / len(keys))

        # default average
        tp, fp, tn, fn = default
        acc = (tp + tn)/(tp + tn + fp + fn) if tp > 0 else 0
        prec = (tp)/(tp + fp) if tp > 0 else 0
        rec = (tp)/(tp + fn) if tp > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if tp > 0 else 0
        data.append([acc, prec, rec, f1])

        # Macro and micro average.
        return self.to_table(data, self.labels + ["micro", "macro", "not-O"], ["label", "acc", "prec", "rec", "f1"])


class Progbar(object):
    """
    Progbar class copied from keras (https://github.com/fchollet/keras/)
    Displays a progress bar.
    # Arguments
        target: Total number of steps expected.
        interval: Minimum visual progress update interval (in seconds).
    """

    def __init__(self, target, width=30, verbose=1):
        self.width = width
        self.target = target
        self.sum_values = {}
        self.unique_values = []
        self.start = time.time()
        self.total_width = 0
        self.seen_so_far = 0
        self.verbose = verbose

    def update(self, current, values=None, exact=None):
        """
        Updates the progress bar.
        # Arguments
            current: Index of current step.
            values: List of tuples (name, value_for_last_step).
                The progress bar will display averages for these values.
            exact: List of tuples (name, value_for_last_step).
                The progress bar will display these values directly.
        """
        values = values or []
        exact = exact or []

        for k, v in values:
            if k not in self.sum_values:
                self.sum_values[k] = [v * (current - self.seen_so_far), current - self.seen_so_far]
                self.unique_values.append(k)
            else:
                self.sum_values[k][0] += v * (current - self.seen_so_far)
                self.sum_values[k][1] += (current - self.seen_so_far)
        for k, v in exact:
            if k not in self.sum_values:
                self.unique_values.append(k)
            self.sum_values[k] = [v, 1]
        self.seen_so_far = current

        now = time.time()
        if self.verbose == 1:
            prev_total_width = self.total_width
            sys.stdout.write("\b" * prev_total_width)
            sys.stdout.write("\r")

            numdigits = int(np.floor(np.log10(self.target))) + 1
            barstr = '%%%dd/%%%dd [' % (numdigits, numdigits)
            bar = barstr % (current, self.target)
            prog = float(current)/self.target
            prog_width = int(self.width*prog)
            if prog_width > 0:
                bar += ('='*(prog_width-1))
                if current < self.target:
                    bar += '>'
                else:
                    bar += '='
            bar += ('.'*(self.width-prog_width))
            bar += ']'
            sys.stdout.write(bar)
            self.total_width = len(bar)

            if current:
                time_per_unit = (now - self.start) / current
            else:
                time_per_unit = 0
            eta = time_per_unit*(self.target - current)
            info = ''
            if current < self.target:
                info += ' - ETA: %ds' % eta
            else:
                info += ' - %ds' % (now - self.start)
            for k in self.unique_values:
                if isinstance(self.sum_values[k], list):
                    info += ' - %s: %.4f' % (k, self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                else:
                    info += ' - %s: %s' % (k, self.sum_values[k])

            self.total_width += len(info)
            if prev_total_width > self.total_width:
                info += ((prev_total_width-self.total_width) * " ")

            sys.stdout.write(info)
            sys.stdout.flush()

            if current >= self.target:
                sys.stdout.write("\n")

        if self.verbose == 2:
            if current >= self.target:
                info = '%ds' % (now - self.start)
                for k in self.unique_values:
                    info += ' - %s: %.4f' % (k, self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                sys.stdout.write(info + "\n")

    def add(self, n, values=None):
        self.update(self.seen_so_far+n, values)

