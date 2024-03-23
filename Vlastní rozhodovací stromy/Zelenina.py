from typing import List, Any, Union
class DecisionTreeNode:
    def __init__(
        self,
        attribute: int,
        thresholds: List[Any],
        descendants: List[Union["DecisionTreeNode", bool]],
    ):
        self.attribute = attribute
        self.thresholds = thresholds
        self.descendants = descendants

    def evaluate(self, data: List[Any]) -> bool:
        value = data[self.attribute]
        for i in range(len(self.thresholds)):
            if value < self.thresholds[i]:
                if self.descendants[i] == True or self.descendants[i] == False:
                    return self.descendants[i]
                return self.descendants[i].evaluate(data)
        if self.descendants[-1] == True or self.descendants[-1] == False:
            return self.descendants[-1]
        return self.descendants[-1].evaluate(data)


class DecisionTree:
    def __init__(self, first_node):
        self.first_node = first_node

    def evaluate(self, data: List[Any]) -> bool:
        return self.first_node.evaluate(data)

data = [
([6.11, False, 2], True),
([0.19, False, 1], True),
([0.19, True, 0], False),
([2.30, True, 1], False),
([7.29, True, 1], False),
([2.58, False, 2], True),
([0.87, True, 1], True),
([5.58, True, 1], False),
([5.58, False, 1], True),
([9.46, True, 2], False),
([0.82, False, 2], True),
([8.92, True, 0], False),
([9.07, True, 2], False),
([5.13, True, 0], False),
([6.54, True, 0], False),
([6.54, True, 2], False),
([6.28, False, 1], True),
([5.13, True, 1], False),
([9.40, False, 0], False),
([6.32, True, 1], False),
([0.24, False, 1], True),
([9.22, True, 1], False),
([0.92, True, 2], True),
([4.15, False, 2], True),
([4.19, True, 1], False),
([9.68, False, 0], False),
([1.53, False, 2], True),
([0.73, False, 1], True)
]

# constants
COST = 0
HAS_AT_HOME = 1
MONEY = 2

# Tuto funkci implementuj.
def make_decision_tree() -> 'DecisionTree':
    node4 = DecisionTreeNode(0, [1], [True, False])
    node3 = DecisionTreeNode(1, [True], [True, node4])
    node2 = DecisionTreeNode(0, [6.5], [True, False])
    node1 = DecisionTreeNode(2, [1, 2], [False, node3, node2])
    return DecisionTree(node1)


# Príklad stromu, ktorý kúpi Zeleninu, ak má karlík veľa peňazí, alebo doma Zeleninu nemá
def make_example_tree() -> DecisionTree:
    # Uzol pozerá vlastnosť 2 - Karlíkove peniaze, hranicu má na hodnote 2, pod
    # ktorou vráti False, a na/nad ňou vráti True
    node1 = DecisionTreeNode(2, [2], [False, True])

    # Uzol pozerá vlastnosť 1 - či má Karlík doma Zeleninu, hranica je na hodnote
    # True, teda pre False sa vráti True a pre True sa ide do node1
    node2 = DecisionTreeNode(1, [True], [True, node1])

    return DecisionTree(node2)


tree = make_decision_tree()
print(tree.evaluate([1.99, False, 0]))
print(tree.evaluate([1.99, True, 2]))
print(tree.evaluate([1.99, True, 0]))

for datum in data:
    print(datum)
    print(tree.evaluate(datum[0]))
    assert datum[1] == tree.evaluate(datum[0])
