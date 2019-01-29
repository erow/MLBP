from math import log
from typing import Any, Union
import numpy as np

from pandas import DataFrame, Index, Series
from pandas.core.arrays import ExtensionArray

from pandas.core.frame import DataFrame
from pandas.core.generic import NDFrame


class DecisionTree(dict):
    classification: Union[None, str]

    def __init__(self, data: DataFrame, label: str, continues_attributes=None):
        dict.__init__(self)
        self.sub_trees = {}
        if continues_attributes is None:
            continues_attributes = []
        self.sub_attribute = None
        self.attributes = generate_attributes(data, continues_attributes)
        self.attributes.pop(label)
        self.data = data
        self.label = label
        self.continues_attributes = continues_attributes
        self.size = data.index.size
        self.dump = dict()

        self.calculate_purity()
        self.generate_subtree()

    def generate_subtree(self):
        data = self.data
        label = self.label
        continues_attributes = self.continues_attributes
        if isinstance(self.sub_attribute, tuple):
            (attribute, value) = self.sub_attribute
            indice = data[attribute] < value
            data = data.drop(attribute, axis=1)
            self.sub_trees = \
                {
                    'yes': gen_tree(data.loc[indice], label, continues_attributes),
                    'no': gen_tree(data.loc[~indice], label, continues_attributes)
                }
        else:
            col = self.data[self.sub_attribute]
            data=data.drop(self.sub_attribute, axis=1)
            for e in self.attributes[self.sub_attribute]:
                indice = col == e
                t = gen_tree(data.loc[indice],
                             label, continues_attributes)
                if t!=None:
                    self.sub_trees[e] = t

    def calculate_purity(self):
        maximal = 0
        D = ent(self.data[self.label])
        if D == 0:
            raise Exception

        for attribute, enum in self.attributes.items():
            if attribute in self.continues_attributes:
                d = 0
                for e in enum:
                    l = self.data[self.label]
                    indice = self.data[attribute] < e
                    p = indice.value_counts() / self.size
                    d1 = D - (p * [ent(l.loc[indice]), ent(l.loc[~indice])]).sum()
                    if d1 > maximal:
                        self.sub_attribute = (attribute, e)
                        maximal = d1
                    d = max(d, d1)
                self.dump[attribute] = d
            else:
                d = 0
                for e in enum:
                    indice = self.data[attribute] == e
                    selected = self.data[self.label].loc[indice]
                    d += (selected.size / self.size) * ent(selected)
                if D - d > maximal:
                    self.sub_attribute = attribute
                    maximal = D - d
                self.dump[attribute] = D - d

    def __getitem__(self, key):
        return self.sub_trees[key]

    def predict(self, x : Series):
        condition = x[self.sub_attribute]
        if condition in self.sub_trees.keys():
            if isinstance(self.sub_trees[condition],str):
                return self.sub_trees[condition]
            else:
                return self.sub_trees[condition].predict(x.drop(self.sub_attribute))
        else:
            return self.data[self.label].value_counts().keys()[0]


def generate_attributes(data: DataFrame, continues_attributes):
    keys = data.keys()
    result = dict()
    for attribute in keys:
        enum = list(set(data[attribute]))
        if attribute in continues_attributes:
            bi_partition = []
            for i in range(1, len(enum)):
                bi_partition.append((enum[i - 1] + enum[i]) / 2)
            result[attribute] = bi_partition
        else:
            result[attribute] = enum
    return result


def ent(data: Series):
    p = (data.value_counts().to_numpy() / data.size)
    return -np.sum(p * np.log2(p))


def ent_continue(data: DataFrame, sep):
    p = (data < sep).mean()
    return -log(p, 2)


def gen_tree(data: DataFrame, label: str, continues_attributes):
    if data.size == 0:  # 样本为空
        return None
    # 返回分类
    D = ent(data[label])
    if D == 0:  # 只有一类样本
        return data[label].iloc[0]

    if len(data.keys()) == 1:  # 无可用标签
        counts = data[label].value_counts()
        return counts.keys()[0]
    # 返回树
    return DecisionTree(data, label, continues_attributes)
