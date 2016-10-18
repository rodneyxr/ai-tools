# -*- coding: utf-8 -*-
"""Entropy for Learning/Decision Trees

This module calculates entropy and gain for learning or decision trees.
It takes, as input, Samples of attributes and results as demonstrated
below. It can calculate entropy of certain values of attributes, entropy
of the entire data set, and gain of an attribute.

Example:
    # Define attributes
    outlook = Attribute('outlook', ('sunny', 'overcast', 'rain'))
    temperature = Attribute('temp', ('hot', 'mild', 'cool'))
    humidity = Attribute('humidity', ('high', 'normal'))
    wind = Attribute('wind', ('weak', 'strong'))

    # Initialize the sample set
    sample_set = SampleSet(
        (outlook, temperature, humidity, wind),  # attributes
        ('yes', 'no'),  # classes
        [
            Sample('no', ('sunny', 'hot', 'high', 'weak')),  # D1
            Sample('no', ('sunny', 'hot', 'high', 'strong')),  # D2
            Sample('yes', ('overcast', 'hot', 'high', 'weak')),  # D3
            Sample('yes', ('rain', 'mild', 'high', 'weak')),  # D4
            Sample('yes', ('rain', 'cool', 'normal', 'weak')),  # D5
            Sample('no', ('rain', 'cool', 'normal', 'strong')),  # D6
            Sample('yes', ('overcast', 'cool', 'normal', 'strong')),  # D7
            Sample('no', ('sunny', 'mild', 'high', 'weak')),  # D8
            Sample('yes', ('sunny', 'cool', 'normal', 'weak')),  # D9
            Sample('yes', ('rain', 'mild', 'normal', 'weak')),  # D10
            Sample('yes', ('sunny', 'mild', 'normal', 'strong')),  # D11
            Sample('yes', ('overcast', 'mild', 'high', 'strong')),  # D12
            Sample('yes', ('overcast', 'hot', 'normal', 'weak')),  # D13
            Sample('no', ('rain', 'mild', 'high', 'strong')),  # D14
        ])

    print(sample_set)
    print()

    gains = []
    for attr in sample_set.attributes:
        gains.append([attr.name, sample_set.gain(attr.name)])
    print(tabulate(gains, headers=['attribute', 'gain']))

Output:
      #  outlook    temp    humidity    wind    class
    ---  ---------  ------  ----------  ------  -------
      1  sunny      hot     high        weak    no
      2  sunny      hot     high        strong  no
      3  overcast   hot     high        weak    yes
      4  rain       mild    high        weak    yes
      5  rain       cool    normal      weak    yes
      6  rain       cool    normal      strong  no
      7  overcast   cool    normal      strong  yes
      8  sunny      mild    high        weak    no
      9  sunny      cool    normal      weak    yes
     10  rain       mild    normal      weak    yes
     11  sunny      mild    normal      strong  yes
     12  overcast   mild    high        strong  yes
     13  overcast   hot     normal      weak    yes
     14  rain       mild    high        strong  no

    attribute         gain
    -----------  ---------
    outlook      0.24675
    temp         0.0292226
    humidity     0.151836
    wind         0.048127

"""
from math import log2

from tabulate import tabulate


def info_function(values):
    """
    Calculates the information function. Result will be between 0 and 1.
    Ex: info_function([x, y]) =
        I(x,y) = -(x/x+y)*log2(x/x+y) - (y/x+y)*log2(y/x+y)

    :param values: List of occurrences of each value.
    :return: The result of the information function.
    """
    total = sum(values)
    fracs = [value / total for value in values]
    i = 0
    for frac in fracs:
        if frac != 0:
            i -= frac * log2(frac)
    return i


class Attribute:
    """
    Represents a possible feature in a SampleSet.
    """

    def __init__(self, name, values):
        self.name = name
        self.values = values

    def __str__(self):
        return '%s: %s' % (self.name, self.values)


class Sample:
    """
    A sample for a SampleSet.

    :result: The result of the sample
    :values: The input that yields the result
    """

    def __init__(self, result, values):
        self.result = result
        self.values = values


class SampleSet:
    """
    :attributes: Tuple of Attributes
    :classes: Tuple of possible classes as sample could be a part of. ex: (yes, no)
    :samples: List of Samples
    """

    def __init__(self, attributes, classes, samples=list()):
        self.attributes = attributes
        self.classes = classes
        self.samples = samples

    def add_sample(self, sample):
        self.samples.append(sample)

    def entropy(self) -> float:
        """
        Entropy(S) = S - p(I) * log2 p(I)

        :return: The entropy of the data
        """
        s = dict.fromkeys(self.classes, 0)
        for x in self.samples:
            s[x.result] += 1

        entropy = 0
        c = len(self.samples)  # cardinality of S
        for _, v in s.items():
            frac = v / c
            if frac != 0:
                entropy -= frac * log2(frac)
        return entropy

    def attribute_entropy(self, attr, value) -> float:
        """
        Entropy(Sweak) = - (6/8)*log2(6/8) - (2/8)*log2(2/8) = 0.811

        :param attr_index: Index of the attribute (ex: 'wind' => 3)
        :param value: Value of the attribute to be calculated (ex: 'weak')
        :return: The entropy of the value
        """
        attr_index = attr
        if isinstance(attr, str):
            attr_index = self.index_of_attribute(attr)

        c = 0  # number of samples that match feature
        s = dict.fromkeys(self.classes, 0)  # result counter for feature
        for x in self.samples:
            if x.values[attr_index] == value:
                c += 1
                s[x.result] += 1

        entropy = 0
        for _, v in s.items():
            frac = v / c
            if frac != 0:
                entropy -= frac * log2(frac)
        return entropy

    def attribute_gain(self, attr1, value, attr2):
        """
        Gain(attr1.value, attr2)
        :param attr1: attribute for the value
        :param value: value of attr1
        :param attr2: attribute to calculate gain for
        :return: the gain of attr2 given attr1.value
        """
        attr1_index = self.index_of_attribute(attr1)
        attr2_index = self.index_of_attribute(attr2)
        # initialize gain with entropy of attr1
        gain = self.attribute_entropy(attr1_index, value)

        # get samples that match attr1.value
        samples = [x for x in self.samples if x.values[attr1_index] == value]

        # split samples into categories by their value for attr2
        cats = dict.fromkeys(self.attributes[attr2_index].values, [])
        for s in samples:
            cats[s.values[attr2_index]].append(s)

        # split samples by results
        results = {}
        for k, l in cats.items():
            s = dict.fromkeys(self.classes, 0)
            for sample in l:
                if sample.values[attr2_index] == k:
                    s[sample.result] += 1
            results[k] = s

        n = len(samples)
        for _, x in results.items():
            l = []
            for _, c in x.items():
                l.append(c)
            gain -= (sum(l) / n) * info_function(l)
        return gain

    def gain(self, attribute) -> float:
        """
        Gain(S, A) = Entropy(S) - S ((|Sv| / |S|) * Entropy(Sv))
        = 0.940 - (8/14)*0.811 - (6/14)*1.00

        :param attribute: The attribute to compute the gain
        :return: The gain the for the given attribute
        """
        gain = self.entropy()
        i = self.index_of_attribute(attribute)

        # count number of occurrences for each feature
        s = dict.fromkeys(self.attributes[i].values, 0)
        for x in self.samples:
            s[x.values[i]] += 1

        c = len(self.samples)
        for k, v in s.items():
            frac = v / c
            gain -= frac * self.attribute_entropy(i, k)
        return gain

    def index_of_attribute(self, attribute):
        """
        Finds the index of the attribute since it is a tuple and not a dict
        :param name: The name of the attribute to look for
        :return: The index of the attribute.
        """
        for i, x in enumerate(self.attributes):
            if x.name == attribute:
                return i
        raise Exception("'%s' is not an attribute." % attribute)

    def __str__(self):
        # generate headers
        headers = ['#'] + [x.name for x in self.attributes] + ['class']

        # generate table
        table = []
        i = 0
        for s in self.samples:
            i += 1
            row = [i] + list(s.values) + [s.result]
            table.append(row)
        return tabulate(table, headers=headers)
