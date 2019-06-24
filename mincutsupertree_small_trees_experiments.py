from mincutsupertree import *
from small_trees import *
# from min_cuts import *


# Experiment 1

X = "ABCDE"
ambig = 2
restrictions = full_cover_restrictions(X, ambig)
restriction = next(restrictions)
# restriction = ("ABC", "ABD")

result = visualize_restricted_rooted_trees(X, restriction)
result_keys = list(result.keys())

for i, key in enumerate(result_keys): print(f"{i}: {len(result[key].treelist)}")

i=8
print(result[result_keys[i]].proj1)
print(result[result_keys[i]].proj2)
for tree in result[result_keys[i]].treelist: print(tree)

trees = [result[result_keys[i]].proj1, result[result_keys[i]].proj2]
supertree = mincutsupertree(trees)
print(supertree)
