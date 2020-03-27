from importlib import reload
from collections import OrderedDict
from more_itertools import powerset

from small_trees import *

import classes
import derivatives

reload(classes)
from classes import *

reload(derivatives)
from derivatives import *

# Restriction experiment for unit tests

reload(classes)
from classes import *

X = Clade("ABCDEF")
restriction = Clade("ABCDE")

ccd = CCDSet.random_with_sparsity(X, sparsity=0.7)
dist = ccd.tree_distribution()
dist_res = dist.restrict(restriction)

res_ccd = ccd.restrict(restriction)
res_ccd_dist = res_ccd.tree_distribution()

dist_res.kl_divergence(res_ccd_dist)
res_ccd_dist.kl_divergence(dist_res)

res_ccd_dist_ccd = CCDSet.from_tree_distribution(res_ccd_dist)

res_ccd.kl_divergence(res_ccd_dist_ccd)
res_ccd_dist_ccd.kl_divergence(res_ccd)

dist_res_ccd = CCDSet.from_tree_distribution(dist_res)

dist_res_ccd.kl_divergence(res_ccd)
res_ccd.kl_divergence(dist_res_ccd)

dist_res_ccd_dist = dist_res_ccd.tree_distribution()

res_ccd_dist.kl_divergence(dist_res_ccd_dist)
dist_res_ccd_dist.kl_divergence(res_ccd_dist)


# Double check CCD probability functions
# Result: iter_ functions have multiple entries per each subsplit

reload(classes)
from classes import *

ccd = CCDSet.random("ABCDEF")

uncond = ccd.unconditional_probabilities()
for subsplit, joint_prob in ccd.iter_unconditional_probabilities():
    print(f"Subsplit {subsplit}: {uncond[subsplit]:8.4g} {joint_prob:8.4g}")

len(uncond)
len(list(ccd.iter_unconditional_probabilities()))


# subsplit probability derivative

from importlib import reload
import classes
import random

reload(classes)
from classes import *


scd = SCDSet.random("ABCDE")
delta = 0.00001

transit = scd.transit_probabilities()

prob_of = random.choice(list(transit.keys()))
# wrt_parent = random.choice(list(transit[prob_of].keys()))
# wrt_child = random.choice(list(scd[wrt_parent].compatible_distribution(prob_of).keys()))
# wrt = PCSS(wrt_parent, wrt_child)
wrt = random.choice(list(scd.iter_pcss()))

scd2 = scd.copy()
scd2.add_log(wrt, delta)
scd2.normalize()
uncond = scd.subsplit_probabilities()
uncond2 = scd2.subsplit_probabilities()
by_hand = (uncond2[prob_of] - uncond[prob_of]) / delta

est_deriv = scd_estimate_subsplit_derivative(prob_of=prob_of, wrt=wrt, scd=scd, delta=delta)
theo_deriv = scd_subsplit_derivative(prob_of=prob_of, wrt=wrt, scd=scd, transit=transit)
print(abs(theo_deriv - est_deriv))
print(abs(theo_deriv - est_deriv)/abs(est_deriv))

est_deriv2 = scd_estimate_subsplit_derivative(prob_of=prob_of, wrt=wrt, scd=scd, delta=delta, uncond=uncond, uncond2=uncond2)
print(f"{by_hand}\n{est_deriv}\n{theo_deriv}\n{est_deriv2}")

tree_dist = scd.tree_distribution()
tree_dist2 = scd2.tree_distribution()
tree_num = tree_dist.feature_prob(prob_of)
tree_num2 = tree_dist2.feature_prob(prob_of)
by_hand2 = (tree_num2 - tree_num) / delta
print(f"{by_hand}\n{est_deriv}\n{theo_deriv}\n{by_hand2}")

#### size 2 subsplit test

root_subsplit = scd.root_subsplit()
prob_of = Subsplit("A", "C")
wrt = PCSS(Subsplit("AC", "BE"), Subsplit("A", "C"))
wrt_parent = wrt.parent
wrt_parent_clade = wrt.parent_clade()
wrt_child = wrt.child

uncond_wrt = transit.get(wrt_parent, dict()).get(root_subsplit, 0.0) * scd[wrt]
child_to_prob_of = transit.get(prob_of, dict()).get(wrt_child, 0.0)
parent_to_prob_of = transit.get(prob_of, dict()).get(wrt_parent_clade, 0.0)

## All x all experiment

# theo_res_simple = dict()
# est_res_simple = dict()
# for prob_of_for in scd.iter_subsplits():
#     if prob_of_for not in theo_res_simple:
#         theo_res_simple[prob_of_for] = dict()
#     if prob_of_for not in est_res_simple:
#         est_res_simple[prob_of_for] = dict()
#     for wrt_for in scd.iter_pcss():
#         theo_res_simple[prob_of_for][wrt_for] = scd_subsplit_derivative(prob_of=prob_of_for, wrt=wrt_for, scd=scd, transit=transit)
#         est_res_simple[prob_of_for][wrt_for] = scd_estimate_subsplit_derivative(prob_of=prob_of_for, wrt=wrt_for, scd=scd, delta=delta)
#         print(f"{prob_of_for} wrt {wrt_for}")
#         print(f"Theo: {theo_res_simple[prob_of_for][wrt_for]:8.4g}")
#         print(f"Est:  {est_res_simple[prob_of_for][wrt_for]:8.4g}")

def brute1(scd, delta):
    theo_res_simple = dict()
    est_res_simple = dict()
    uncond_for = scd.subsplit_probabilities()
    transit_for = scd.transit_probabilities()
    for wrt_for in scd.iter_pcss():
        if wrt_for not in theo_res_simple:
            theo_res_simple[wrt_for] = dict()
        if wrt_for not in est_res_simple:
            est_res_simple[wrt_for] = dict()
        scd2_for = scd.copy()
        scd2_for.add_log(wrt_for, delta)
        scd2_for.normalize(wrt_for.parent_clade())
        uncond2_for = scd2_for.subsplit_probabilities()
        for prob_of_for in scd.iter_subsplits(include_root=True):
            theo = scd_subsplit_derivative(
                prob_of=prob_of_for, wrt=wrt_for, scd=scd, transit=transit_for
            )
            theo_res_simple[wrt_for][prob_of_for] = theo
            est = scd_estimate_subsplit_derivative(
                prob_of=prob_of_for, wrt=wrt_for, scd=scd, delta=delta,
                uncond=uncond_for, uncond2=uncond2_for
            )
            est_res_simple[wrt_for][prob_of_for] = est
            if not theo == est == 0.0:
                print(f"{prob_of_for} wrt {wrt_for}")
                print(f"Theo: {theo:8.4g}")
                print(f"Est:  {est:8.4g}")
    return theo_res_simple, est_res_simple

theo_res_simple, est_res_simple = brute1(scd, delta)

for wrt_for in theo_res_simple:
    for prob_of_for in theo_res_simple[wrt_for]:
        theo = theo_res_simple[wrt_for][prob_of_for]
        est = est_res_simple[wrt_for][prob_of_for]
        if abs(est) < 1e-12:
            abs_diff = abs(theo - est)
            if abs_diff > 1e-8:
                print(f"{prob_of_for} wrt {wrt_for}: est zero, theo={theo:10.6g}")
        elif abs(theo) < 1e-12:
            abs_diff = abs(theo - est)
            if abs_diff > 1e-8:
                print(f"{prob_of_for} wrt {wrt_for}: theo zero, est={est:10.6g}")
        else:
            rel_diff = abs(theo - est)/abs(est)
            if rel_diff > 5e-6:
                print(f"{prob_of_for} wrt {wrt_for}: rel diff={rel_diff:10.6g}")


### double check

wrt_test = PCSS(Subsplit("BDE", "C"), Subsplit("B", "DE"))
prob_of_test = Subsplit("B", "DE")
scd_estimate_subsplit_derivative(prob_of=prob_of_test, wrt=wrt_test, scd=scd, delta=delta)
scd_subsplit_derivative(prob_of=prob_of_test, wrt=wrt_test, scd=scd, transit=transit)
theo_res_simple[wrt_test][prob_of_test]
est_res_simple[wrt_test][prob_of_test]

scd2_test = scd.copy()
scd2_test.add_log(wrt_test, delta)
scd2_test.normalize(wrt_test.parent)
uncond_test = scd.subsplit_probabilities()
uncond2_test = scd2_test.subsplit_probabilities()
scd_estimate_subsplit_derivative(prob_of=prob_of_test, wrt=wrt_test, scd=scd, delta=delta, uncond=uncond_test, uncond2=uncond2_test)


#### stealing from below

theo_res_brute, est_res_brute = efficient_brute_force_gradient(scd=scd, wrt=wrt_test, delta=delta)

### efficient brute force


def efficient_brute_force_gradient(scd: SCDSet, wrt: PCSS, delta: float=0.00001):
    theo_res = dict()
    est_res = dict()
    uncond = scd.subsplit_probabilities()
    transit = scd.transit_probabilities()
    scd2 = scd.copy()
    scd2.add_log(wrt, delta)
    scd2.normalize(wrt.parent)
    uncond2 = scd2.subsplit_probabilities()
    for prob_of in scd.iter_subsplits():
        theo = scd_subsplit_derivative(prob_of=prob_of, wrt=wrt, scd=scd, transit=transit)
        est = scd_estimate_subsplit_derivative(prob_of=prob_of, wrt=wrt, scd=scd, delta=delta, uncond=uncond, uncond2=uncond2)
        theo_res[prob_of] = theo
        est_res[prob_of] = est
        if not theo == est == 0.0:
            print(f"{prob_of} wrt {wrt}")
            print(f"Theo: {theo:8.4g}")
            print(f"Est:  {est:8.4g}")
    return theo_res, est_res


wrt_brute = random.choice(list(scd.iter_pcss()))

theo_res_brute, est_res_brute = efficient_brute_force_gradient(scd=scd, wrt=wrt_brute, delta=delta)
# theo_res_brute
# est_res_brute

for prob_of_for in theo_res_brute:
    theo = theo_res_brute[prob_of_for]
    est = est_res_brute[prob_of_for]
    if abs(est) < 1e-12:
        abs_diff = abs(theo - est)
        if abs_diff > 1e-8:
            print(f"{prob_of_for} wrt {wrt_brute}: abs diff={abs_diff:10.6g}")
    else:
        rel_diff = abs(theo - est) / abs(est)
        if rel_diff > 1e-4:
            print(f"{prob_of_for} wrt {wrt_brute}: rel diff={rel_diff:10.6g}")

### Looped

theo_res_brute2 = dict()
est_res_brute2 = dict()
for wrt_brute in scd.iter_pcss():
    theo_res_brute2[wrt_brute], est_res_brute2[wrt_brute] = efficient_brute_force_gradient(scd=scd, wrt=wrt_brute, delta=delta)

for wrt_brute in theo_res_brute2:
    for prob_of_for in theo_res_brute2[wrt_brute]:
        theo = theo_res_brute2[wrt_brute][prob_of_for]
        est = est_res_brute2[wrt_brute][prob_of_for]
        if abs(est) < 1e-12:
            abs_diff = abs(theo - est)
            if abs_diff > 1e-12:
                print(f"{prob_of_for} wrt {wrt_brute}: abs diff={abs_diff:10.6g}")
        else:
            rel_diff = abs(theo - est) / abs(est)
            if rel_diff > 5e-6:
                print(f"{prob_of_for} wrt {wrt_brute}: rel diff={rel_diff:10.6g}")

# subsplit to subsplit probability derivative

reload(classes)
from classes import *


scd = SCDSet.random("ABCDE")
delta = 0.00001

s2s = scd.subsplit_to_subsplit_probabilities()

prob_of = random.choice(list(s2s.keys()))
wrt_parent = random.choice(list(s2s[prob_of].keys()))
wrt_child = random.choice(list(scd[wrt_parent].compatible_distribution(prob_of).keys()))
wrt = PCSS(wrt_parent, wrt_child)
cond_on = random.choice(list(s2s[wrt_parent].keys()))

scd2 = scd.copy()
scd2.add_log(wrt, delta)
scd2.normalize()
s2s = scd.subsplit_to_subsplit_probabilities()
s2s2 = scd2.subsplit_to_subsplit_probabilities()
by_hand = (s2s2[prob_of][cond_on] - s2s[prob_of][cond_on]) / delta

est_deriv = scd_estimate_subsplit_to_subsplit_cond_derivative(prob_of=prob_of, cond_on=cond_on, wrt=wrt, scd=scd, delta=delta)
theo_deriv = scd_subsplit_to_subsplit_cond_derivative(prob_of=prob_of, cond_on=cond_on, wrt=wrt, scd=scd, transit=s2s)
print(abs(theo_deriv - est_deriv))
print(abs(theo_deriv - est_deriv)/abs(est_deriv))
print(f"{by_hand}\n{est_deriv}\n{theo_deriv}")

tree_dist = scd.tree_distribution()
tree_dist2 = scd2.tree_distribution()
tree_num = tree_dist.prob_all([prob_of, cond_on])
tree_num2 = tree_dist2.prob_all([prob_of, cond_on])
tree_den = tree_dist.feature_prob(cond_on)
tree_den2 = tree_dist2.feature_prob(cond_on)
tree_cond = tree_num/tree_den
tree_cond2 = tree_num2/tree_den2
by_hand2 = (tree_cond2 - tree_cond) / delta
print(f"{by_hand}\n{est_deriv}\n{theo_deriv}\n{by_hand2}")

## all x all experiment


def brute2(scd, delta):
    theo_res_simple = dict()
    est_res_simple = dict()
    # uncond_for = scd.subsplit_probabilities()
    transit_for = scd.transit_probabilities()
    for wrt_for in scd.iter_pcss():
        if wrt_for not in theo_res_simple:
            theo_res_simple[wrt_for] = dict()
        if wrt_for not in est_res_simple:
            est_res_simple[wrt_for] = dict()
        scd2_for = scd.copy()
        scd2_for.add_log(wrt_for, delta)
        scd2_for.normalize(wrt_for.parent_clade())
        # uncond2_for = scd2_for.subsplit_probabilities()
        transit2_for = scd2_for.transit_probabilities()
        for cond_on_for, prob_of_for in product(scd.iter_subsplits(include_root=True), scd.iter_subsplits(include_root=True)):
            theo = scd_subsplit_to_subsplit_cond_derivative(
                prob_of=prob_of_for, cond_on=cond_on_for, wrt=wrt_for, scd=scd, transit=transit_for
            )
            theo_res_simple[wrt_for][(cond_on_for, prob_of_for)] = theo
            est = scd_estimate_subsplit_to_subsplit_cond_derivative(
                prob_of=prob_of_for, cond_on=cond_on_for, wrt=wrt_for, scd=scd, delta=delta,
                transit=transit_for, transit2=transit2_for
            )
            est_res_simple[wrt_for][(cond_on_for, prob_of_for)] = est
            if not theo == est == 0.0:
                print(f"{prob_of_for}|{cond_on_for} wrt {wrt_for}")
                print(f"Theo: {theo:8.4g}")
                print(f"Est:  {est:8.4g}")
    return theo_res_simple, est_res_simple


theo_res_simple, est_res_simple = brute2(scd, delta)

for wrt_for in theo_res_simple:
    for cond_on_for, prob_of_for in theo_res_simple[wrt_for]:
        theo = theo_res_simple[wrt_for][(cond_on_for, prob_of_for)]
        est = est_res_simple[wrt_for][(cond_on_for, prob_of_for)]
        if abs(est) < 1e-12:
            abs_diff = abs(theo - est)
            if abs_diff > 1e-8:
                print(f"{prob_of_for}|{cond_on_for} wrt {wrt_for}: abs diff={abs_diff:10.6g}")
        else:
            rel_diff = abs(theo - est)/abs(est)
            if rel_diff > 5e-6:
                print(f"{prob_of_for}|{cond_on_for} wrt {wrt_for}: rel diff={rel_diff:10.6g}")



# via experiment

reload(classes)
from classes import *


scd = SCDSet.random("ABCDE")
root_subsplit = scd.root_subsplit()
delta = 0.00001

transit = scd.transit_probabilities()

# prob_of = random.choice(list(transit.keys()))
# wrt_parent = random.choice(list(transit[prob_of].keys()))
# wrt_child = random.choice(list(scd[wrt_parent].compatible_distribution(prob_of).keys()))
# wrt = PCSS(wrt_parent, wrt_child)
# via = random.choice(list(transit[prob_of].keys()))
prob_of = Subsplit("A", "B")
via = Subsplit("AB", "DE")
wrt = PCSS(Subsplit("AB", "DE"), Subsplit("A", "B"))

scd2 = scd.copy()
scd2.add_log(wrt, delta)
scd2.normalize()

transit2 = scd2.transit_probabilities()
by_hand = (transit2[prob_of][via]*transit2[via][root_subsplit] - transit[prob_of][via]*transit[via][root_subsplit]) / delta

est_deriv = scd_estimate_subsplit_via_subsplit_derivative(prob_of=prob_of, via=via, wrt=wrt, scd=scd, delta=delta)
theo_deriv = scd_subsplit_via_subsplit_derivative(prob_of=prob_of, via=via, wrt=wrt, scd=scd, transit=transit)
print(abs(theo_deriv - est_deriv))
print(abs(theo_deriv - est_deriv)/abs(est_deriv))
print(f"{by_hand}\n{est_deriv}\n{theo_deriv}")

## all x all experiment


def brute3(scd, delta):
    theo_res_simple = dict()
    est_res_simple = dict()
    # uncond_for = scd.subsplit_probabilities()
    transit_for = scd.transit_probabilities()
    for wrt_for in scd.iter_pcss():
        if wrt_for not in theo_res_simple:
            theo_res_simple[wrt_for] = dict()
        if wrt_for not in est_res_simple:
            est_res_simple[wrt_for] = dict()
        scd2_for = scd.copy()
        scd2_for.add_log(wrt_for, delta)
        scd2_for.normalize(wrt_for.parent_clade())
        # uncond2_for = scd2_for.subsplit_probabilities()
        transit2_for = scd2_for.transit_probabilities()
        for via_for, prob_of_for in product(scd.iter_subsplits(include_root=True), scd.iter_subsplits(include_root=True)):
            theo = scd_subsplit_via_subsplit_derivative(
                prob_of=prob_of_for, via=via_for, wrt=wrt_for, scd=scd, transit=transit_for
            )
            theo_res_simple[wrt_for][(via_for, prob_of_for)] = theo
            est = scd_estimate_subsplit_via_subsplit_derivative(
                prob_of=prob_of_for, via=via_for, wrt=wrt_for, scd=scd, delta=delta,
                transit=transit_for, transit2=transit2_for
            )
            est_res_simple[wrt_for][(via_for, prob_of_for)] = est
            if not theo == est == 0.0:
                print(f"{prob_of_for}|{via_for} wrt {wrt_for}")
                print(f"Theo: {theo:8.4g}")
                print(f"Est:  {est:8.4g}")
    return theo_res_simple, est_res_simple


theo_res_simple, est_res_simple = brute3(scd, delta)

for wrt_for in theo_res_simple:
    for via_for, prob_of_for in theo_res_simple[wrt_for]:
        theo = theo_res_simple[wrt_for][(via_for, prob_of_for)]
        est = est_res_simple[wrt_for][(via_for, prob_of_for)]
        if abs(est) < 1e-12:
            abs_diff = abs(theo - est)
            if abs_diff > 1e-8:
                print(f"{prob_of_for}|{via_for} wrt {wrt_for}: abs diff={abs_diff:10.6g}")
        else:
            rel_diff = abs(theo - est)/abs(est)
            if rel_diff > 5e-6:
                print(f"{prob_of_for}|{via_for} wrt {wrt_for}: rel diff={rel_diff:10.6g}")


# restricted unconditional derivative

reload(classes)
from classes import *


scd = SCDSet.random("ABCDE")
restriction = Clade("ABCD")
delta = 0.00001

transit = scd.transit_probabilities()

scd_res = scd.restrict(restriction)

# prob_of = random.choice(list(scd_res.iter_subsplits()))
# possible = [subsplit for subsplit in scd.iter_subsplits() if subsplit.restrict(restriction) == prob_of]
# prob_of_full = random.choice(possible)
# wrt_parent = random.choice(list(s2s[prob_of_full].keys()))
# wrt_child = random.choice(list(scd[wrt_parent].compatible_distribution(prob_of_full).keys()))
# wrt = PCSS(wrt_parent, wrt_child)

prob_of = Subsplit("ABCD")
wrt = PCSS(Subsplit("ABCDE"), Subsplit("A", "BCDE"))
# wrt = wrt_brute

scd2 = scd.copy()
scd2.add_log(wrt, delta)
scd2.normalize(wrt.parent)
scd2_res = scd2.restrict(restriction)

uncond = scd_res.subsplit_probabilities()
uncond2 = scd2_res.subsplit_probabilities()
by_hand = (uncond2[prob_of] - uncond[prob_of]) / delta

est_deriv = scd_estimate_restricted_subsplit_derivative(restriction=restriction, prob_of=prob_of, wrt=wrt, scd=scd, delta=delta)
theo_deriv = scd_restricted_subsplit_derivative(restriction=restriction, prob_of=prob_of, wrt=wrt, scd=scd, transit=transit)
print(abs(theo_deriv - est_deriv))
print(abs(theo_deriv - est_deriv)/abs(est_deriv))

est_deriv2 = scd_estimate_restricted_subsplit_derivative(restriction=restriction, prob_of=prob_of, wrt=wrt, scd=scd, delta=delta, uncond=uncond, uncond2=uncond2)
print(abs(est_deriv2 - est_deriv))

tree_dist = scd_res.tree_distribution()
tree_dist2 = scd2_res.tree_distribution()
tree_num = tree_dist.feature_prob(prob_of)
tree_num2 = tree_dist2.feature_prob(prob_of)
by_hand2 = (tree_num2 - tree_num) / delta
print(f"{by_hand}\n{est_deriv}\n{theo_deriv}\n{by_hand2}")

## All x all experiment

# theo_res = dict()
# est_res = dict()
# for prob_of_for in scd_res.iter_subsplits():
#     if prob_of_for not in theo_res:
#         theo_res[prob_of_for] = dict()
#     if prob_of_for not in est_res:
#         est_res[prob_of_for] = dict()
#     for wrt_for in scd.iter_pcss():
#         theo_res[prob_of_for][wrt_for] = scd_restricted_subsplit_derivative(restriction=restriction, prob_of=prob_of_for, wrt=wrt_for, scd=scd, s2s=s2s)
#         est_res[prob_of_for][wrt_for] = scd_estimate_restricted_subsplit_derivative(restriction=restriction, prob_of=prob_of_for, wrt=wrt_for, scd=scd, delta=delta)
#         print(f"{prob_of_for} wrt {wrt_for}")
#         print(f"Theo: {theo_res[prob_of_for][wrt_for]:8.4g}")
#         print(f"Est:  {est_res[prob_of_for][wrt_for]:8.4g}")


def brute4(scd: SCDSet, restriction, delta):
    theo_res = dict()
    est_res = dict()
    scd_res = scd.restrict(restriction)
    uncond = scd_res.subsplit_probabilities()
    transit = scd.transit_probabilities()
    for wrt_for in scd.iter_pcss():
        if wrt_for not in theo_res:
            theo_res[wrt_for] = dict()
        if wrt_for not in est_res:
            est_res[wrt_for] = dict()
        parent = wrt_for.parent_clade()
        scd2 = scd.copy()
        scd2.add_log(wrt_for, delta)
        scd2.normalize(parent)
        scd2_res = scd2.restrict(restriction)
        uncond2 = scd2_res.subsplit_probabilities()
        for prob_of_for in scd_res.iter_subsplits(include_root=True):
            theo = scd_restricted_subsplit_derivative(restriction=restriction, prob_of=prob_of_for, wrt=wrt_for, scd=scd, transit=transit)
            theo_res[wrt_for][prob_of_for] = theo
            est = scd_estimate_restricted_subsplit_derivative(restriction=restriction, prob_of=prob_of_for, wrt=wrt_for, scd=scd, delta=delta, uncond=uncond, uncond2=uncond2)
            est_res[wrt_for][prob_of_for] = est
            if not theo == est == 0.0:
                print(f"{prob_of_for} wrt {wrt_for}")
                print(f"Theo: {theo_res[wrt_for][prob_of_for]:8.4g}")
                print(f"Est:  {est_res[wrt_for][prob_of_for]:8.4g}")
    return theo_res, est_res


theo_res, est_res = brute4(scd, restriction, delta)

for wrt_for in theo_res:
    for prob_of_for in theo_res[wrt_for]:
        theo = theo_res[wrt_for][prob_of_for]
        est = est_res[wrt_for][prob_of_for]
        if abs(est) < 1e-12:
            abs_diff = abs(theo - est)
            if abs_diff > 1e-8:
                print(f"{prob_of_for} wrt {wrt_for}: est zero, theo={theo:10.6g}")
        elif abs(theo) < 1e-12:
            abs_diff = abs(theo - est)
            if abs_diff > 1e-8:
                print(f"{prob_of_for} wrt {wrt_for}: theo zero, est={est:10.6g}")
        else:
            rel_diff = abs(theo - est)/abs(est)
            if rel_diff > 1e-5:
                print(f"{prob_of_for} wrt {wrt_for}: rel diff={rel_diff:10.6g} theo={theo:10.6g}")

#### check

theo_res[wrt][prob_of]
est_res[wrt][prob_of]

prob_of_test = Subsplit("ABCD")
wrt_test = PCSS(Subsplit("ABCDE"), Subsplit("ABCD", "E"))
for subsplit in scd.iter_subsplits(include_root=True):
    restricted_subsplit = subsplit.restrict(restriction)
    if restricted_subsplit == prob_of_test:
        print(f"{subsplit} -> {restricted_subsplit} wrt {wrt_test}: {scd_subsplit_derivative(prob_of=subsplit, wrt=wrt_test, scd=scd, transit=transit)}")


## efficient brute force


def efficient_brute_force_restricted_gradient(restriction: Clade, scd: SCDSet, wrt: PCSS, delta: float=0.00001, scd_res: SCDSet=None, uncond=None, s2s=None):
    theo_res = dict()
    est_res = dict()
    if scd_res is None:
        scd_res = scd.restrict(restriction)
    if uncond is None:
        uncond = scd_res.subsplit_probabilities()
    if s2s is None:
        s2s = scd.subsplit_to_subsplit_probabilities()
    scd2 = scd.copy()
    scd2.add_log(wrt, delta)
    scd2.normalize(wrt.parent)
    scd2_res = scd2.restrict(restriction)
    uncond2 = scd2_res.subsplit_probabilities()
    for prob_of in scd_res.iter_subsplits():
        theo = scd_restricted_subsplit_derivative(restriction=restriction, prob_of=prob_of, wrt=wrt, scd=scd, transit=s2s)
        est = scd_estimate_restricted_subsplit_derivative(restriction=restriction, prob_of=prob_of, wrt=wrt, scd=scd, delta=delta, uncond=uncond, uncond2=uncond2)
        theo_res[prob_of] = theo
        est_res[prob_of] = est
        if not theo == est == 0.0:
            print(f"{prob_of} wrt {wrt}")
            print(f"Theo: {theo:8.4g}")
            print(f"Est:  {est:8.4g}")
    return theo_res, est_res


wrt_brute = random.choice(list(scd.iter_pcss()))
# wrt_brute = wrt

theo_res_brute, est_res_brute = efficient_brute_force_restricted_gradient(restriction=restriction, scd=scd, wrt=wrt_brute, delta=delta)
# theo_res_brute
# est_res_brute

for prob_of_for in theo_res_brute:
    theo = theo_res_brute[prob_of_for]
    est = est_res_brute[prob_of_for]
    if abs(est) < 1e-12:
        abs_diff = abs(theo - est)
        if abs_diff > 1e-8:
            print(f"{prob_of_for} wrt {wrt_brute}: abs diff={abs_diff:10.6g}")
    else:
        rel_diff = abs(theo - est) / abs(est)
        if rel_diff > 1e-4:
            print(f"{prob_of_for} wrt {wrt_brute}: rel diff={rel_diff:10.6g}")

### Looped

scd_res = scd.restrict(restriction)
uncond = scd_res.subsplit_probabilities()
s2s = scd.subsplit_to_subsplit_probabilities()
theo_res_brute2 = dict()
est_res_brute2 = dict()
for wrt_brute in scd.iter_pcss():
    theo_res_brute2[wrt_brute], est_res_brute2[wrt_brute] = efficient_brute_force_restricted_gradient(restriction=restriction, scd=scd, wrt=wrt_brute, delta=delta, scd_res=scd_res, uncond=uncond, s2s=s2s)

for wrt_brute in theo_res_brute2:
    for prob_of_for in theo_res_brute2[wrt_brute]:
        theo = theo_res_brute2[wrt_brute][prob_of_for]
        est = est_res_brute2[wrt_brute][prob_of_for]
        if abs(est) < 1e-12:
            abs_diff = abs(theo - est)
            if abs_diff > 1e-12:
                print(f"{prob_of_for} wrt {wrt_brute}: abs diff={abs_diff:10.6g}")
        else:
            rel_diff = abs(theo - est) / abs(est)
            if rel_diff > 1e-5:
                print(f"{prob_of_for} wrt {wrt_brute}: rel diff={rel_diff:10.6g}")


# restricted PCSS derivative

reload(classes)
from classes import *


scd = SCDSet.random("ABCDE")
restriction = Clade("ABCD")
delta = 0.00001

transit = scd.transit_probabilities()

scd_res = scd.restrict(restriction)

# prob_of = random.choice(list(scd_res.iter_pcss()))
# possible = [subsplit for subsplit in scd.iter_subsplits() if subsplit.restrict(restriction) == prob_of.parent]
# prob_of_full = random.choice(possible)
# wrt_parent = random.choice(list(s2s[prob_of_full].keys()))
# wrt_child = random.choice(list(scd[wrt_parent].compatible_distribution(prob_of_full).keys()))
# wrt = PCSS(wrt_parent, wrt_child)

prob_of = PCSS(Subsplit("AB", "D"), Subsplit("A", "B"))
wrt = PCSS(Subsplit("AB", "DE"), Subsplit("A", "B"))

scd2 = scd.copy()
scd2.add_log(wrt, delta)
scd2.normalize(wrt.parent_clade())
scd2_res = scd2.restrict(restriction)

res_pcss_probs = scd_res.pcss_probabilities()
res_pcss_probs2 = scd2_res.pcss_probabilities()
by_hand = (res_pcss_probs2[prob_of] - res_pcss_probs[prob_of]) / delta

est_deriv = scd_estimate_restricted_pcss_derivative(restriction=restriction, prob_of=prob_of, wrt=wrt, scd=scd, delta=delta)
theo_deriv = scd_restricted_pcss_derivative(restriction=restriction, prob_of=prob_of, wrt=wrt, scd=scd, transit=transit)
print(abs(theo_deriv - est_deriv))
print(abs(theo_deriv - est_deriv)/abs(est_deriv))

tree_dist = scd_res.tree_distribution()
tree_dist2 = scd2_res.tree_distribution()
tree_num = tree_dist.feature_prob(prob_of)
tree_num2 = tree_dist2.feature_prob(prob_of)
by_hand2 = (tree_num2 - tree_num) / delta
print(f"{by_hand}\n{est_deriv}\n{theo_deriv}\n{by_hand2}")

## All x all


def brute5(scd: SCDSet, restriction, delta):
    theo_res = dict()
    est_res = dict()
    scd_res = scd.restrict(restriction)
    res_pcss_probs = scd_res.pcss_probabilities()
    transit = scd.transit_probabilities()
    for wrt_for in scd.iter_pcss():
        if wrt_for not in theo_res:
            theo_res[wrt_for] = dict()
        if wrt_for not in est_res:
            est_res[wrt_for] = dict()
        parent = wrt_for.parent_clade()
        scd2 = scd.copy()
        scd2.add_log(wrt_for, delta)
        scd2.normalize(parent)
        scd2_res = scd2.restrict(restriction)
        res_pcss_probs2 = scd2_res.pcss_probabilities()
        for prob_of_for in scd_res.iter_pcss():
            theo = scd_restricted_pcss_derivative(restriction=restriction, prob_of=prob_of_for, wrt=wrt_for, scd=scd, transit=transit)
            theo_res[wrt_for][prob_of_for] = theo
            est = scd_estimate_restricted_pcss_derivative(restriction=restriction, prob_of=prob_of_for, wrt=wrt_for, scd=scd, delta=delta, res_pcss_probs=res_pcss_probs, res_pcss_probs2=res_pcss_probs2)
            est_res[wrt_for][prob_of_for] = est
            if not theo == est == 0.0:
                print(f"{prob_of_for} wrt {wrt_for}")
                print(f"Theo: {theo_res[wrt_for][prob_of_for]:8.4g}")
                print(f"Est:  {est_res[wrt_for][prob_of_for]:8.4g}")
    return theo_res, est_res


theo_res, est_res = brute5(scd, restriction, delta)

for wrt_for in theo_res:
    for prob_of_for in theo_res[wrt_for]:
        theo = theo_res[wrt_for][prob_of_for]
        est = est_res[wrt_for][prob_of_for]
        if abs(est) < 1e-12:
            abs_diff = abs(theo - est)
            if abs_diff > 1e-8:
                print(f"{prob_of_for} wrt {wrt_for}: est around zero, theo = {theo:10.6g}")
        elif abs(theo) < 1e-12:
            abs_diff = abs(theo - est)
            if abs_diff > 1e-8:
                print(f"{prob_of_for} wrt {wrt_for}: theo around zero, est={est:10.6g}")
        else:
            rel_diff = abs(theo - est)/abs(est)
            if rel_diff > 5e-6:
                print(f"{prob_of_for} wrt {wrt_for}: rel diff={rel_diff:10.6g} theo={theo:10.6g}")


### checking code

root_subsplit = scd.root_subsplit()
child_equiv = [dest for dest in transit if dest.restrict(restriction) == prob_of.child]
child_equiv
parent_equiv = dict()
for destination in child_equiv:
    parent_equiv[destination] = []
    for ancestor in transit[destination]:
        if not isinstance(ancestor, Subsplit):
            continue
        ansr_res = ancestor.restrict(restriction)
        if ansr_res != prob_of.parent or (ansr_res.is_trivial() and ancestor != root_subsplit):
            continue
        parent_equiv[destination].append(ancestor)
parent_equiv

paths = []
for destination in child_equiv:
    parent_equiv[destination] = []
    for ancestor in transit[destination]:
        if not isinstance(ancestor, Subsplit):
            continue
        ansr_res = ancestor.restrict(restriction)
        if ansr_res != prob_of.parent or (ansr_res.is_trivial() and ancestor != root_subsplit):
            continue
        paths.append((ancestor, destination))
paths

result = 0.0
for path in paths:
    ancestor, destination = path
    val = scd_subsplit_via_subsplit_derivative(prob_of=destination, via=ancestor, wrt=wrt, scd=scd, transit=transit)
    print(f"from {ancestor} to {destination}: {val}")
    result += val
result
str(wrt)

## Efficient brute force


def efficient_brute_force_restricted_pcss_gradient(restriction: Clade, scd: SCDSet, wrt: PCSS, delta: float=0.00001, scd_res: SCDSet=None, res_pcss_probs=None, s2s=None):
    theo_res = dict()
    est_res = dict()
    if scd_res is None:
        scd_res = scd.restrict(restriction)
    if res_pcss_probs is None:
        res_pcss_probs = scd_res.pcss_probabilities()
    if s2s is None:
        s2s = scd.subsplit_to_subsplit_probabilities()
    scd2 = scd.copy()
    scd2.add_log(wrt, delta)
    scd2.normalize(wrt.parent)
    scd2_res = scd2.restrict(restriction)
    res_pcss_probs2 = scd2_res.pcss_probabilities()
    for prob_of in scd_res.iter_pcss():
        theo = scd_restricted_pcss_derivative(restriction=restriction, prob_of=prob_of, wrt=wrt, scd=scd, transit=s2s)
        est = scd_estimate_restricted_pcss_derivative(restriction=restriction, prob_of=prob_of, wrt=wrt, scd=scd, delta=delta, res_pcss_probs=res_pcss_probs, res_pcss_probs2=res_pcss_probs2)
        theo_res[prob_of] = theo
        est_res[prob_of] = est
        if not theo == est == 0.0:
            print(f"{prob_of} wrt {wrt}")
            print(f"Theo: {theo:8.4g}")
            print(f"Est:  {est:8.4g}")
    return theo_res, est_res


wrt_brute = random.choice(list(scd.iter_pcss()))
# wrt_brute = wrt

theo_res_brute, est_res_brute = efficient_brute_force_restricted_pcss_gradient(restriction=restriction, scd=scd, wrt=wrt_brute, delta=delta)
# theo_res_brute
# est_res_brute

for prob_of_for in theo_res_brute:
    theo = theo_res_brute[prob_of_for]
    est = est_res_brute[prob_of_for]
    if abs(est) < 1e-12:
        abs_diff = abs(theo - est)
        if abs_diff > 1e-8:
            print(f"{prob_of_for} wrt {wrt_brute}: abs diff={abs_diff:10.6g}")
    else:
        rel_diff = abs(theo - est) / abs(est)
        if rel_diff > 1e-4:
            print(f"{prob_of_for} wrt {wrt_brute}: rel diff={rel_diff:10.6g}")

### Looped

scd_res = scd.restrict(restriction)
res_pcss_probs = scd_res.pcss_probabilities()
s2s = scd.subsplit_to_subsplit_probabilities()
theo_res_brute2 = dict()
est_res_brute2 = dict()
for wrt_brute in scd.iter_pcss():
    theo_res_brute2[wrt_brute], est_res_brute2[wrt_brute] = efficient_brute_force_restricted_pcss_gradient(restriction=restriction, scd=scd, wrt=wrt_brute, delta=delta, scd_res=scd_res, res_pcss_probs=res_pcss_probs, s2s=s2s)

for wrt_brute in theo_res_brute2:
    for prob_of_for in theo_res_brute2[wrt_brute]:
        theo = theo_res_brute2[wrt_brute][prob_of_for]
        est = est_res_brute2[wrt_brute][prob_of_for]
        if bool(est) ^ bool(theo):
            diff = theo - est
            print(f"{prob_of_for} wrt {wrt_brute}: theo-est={diff:10.6g}")
        elif abs(est) < 1e-12:
            abs_diff = abs(theo - est)
            if abs_diff > 1e-12:
                print(f"{prob_of_for} wrt {wrt_brute}: abs diff={abs_diff:10.6g}")
        else:
            rel_diff = abs(theo - est) / abs(est)
            if rel_diff > 1e-5:
                print(f"{prob_of_for} wrt {wrt_brute}: rel diff={rel_diff:10.6g}")





# restricted conditional derivative


reload(classes)
from classes import *


scd = SCDSet.random("ABCDE")
restriction = Clade("ABCD")
delta = 0.00001

transit = scd.transit_probabilities()

scd_res = scd.restrict(restriction)

# prob_of = random.choice(list(scd_res.iter_pcss()))
# possible = [subsplit for subsplit in scd.iter_subsplits() if subsplit.restrict(restriction) == prob_of.parent]
# prob_of_full = random.choice(possible)
# wrt_parent = random.choice(list(s2s[prob_of_full].keys()))
# wrt_child = random.choice(list(scd[wrt_parent].compatible_distribution(prob_of_full).keys()))
# wrt = PCSS(wrt_parent, wrt_child)

# prob_of = PCSS(Subsplit("ABC", "D"), Subsplit("AB", "C"))
# prob_of=PCSS(Subsplit("ACE", "BD"), Subsplit("AC", "E"))
# wrt = PCSS(Subsplit("ABCDEF"), Subsplit("ACEF", "BD"))
prob_of = PCSS(Subsplit("ABCD"), Subsplit("A", "BCD"))
wrt = PCSS(Subsplit(Clade("ABCDE")), Subsplit(Clade("A"), Clade("BCDE")))

scd2 = scd.copy()
scd2.add_log(wrt, delta)
scd2.normalize()
scd2_res = scd2.restrict(restriction)

by_hand = (scd2_res[prob_of] - scd_res[prob_of]) / delta
by_hand

est_deriv = scd_estimate_restricted_conditional_derivative(restriction=restriction, prob_of=prob_of, wrt=wrt, scd=scd, delta=delta)
theo_deriv = scd_restricted_conditional_derivative(restriction=restriction, prob_of=prob_of, wrt=wrt, scd=scd, transit=transit)
print(abs(theo_deriv - est_deriv))
print(abs(theo_deriv - est_deriv)/abs(est_deriv))

restricted_pcss_probs = scd_res.pcss_probabilities()
restricted_subsplit_probs = scd_res.subsplit_probabilities()
# Quotient rule d(top/bot) = (bot*dtop-top*dbot) / bot**2
top = restricted_pcss_probs[prob_of]
print(f"top: {top}")
bot = restricted_subsplit_probs[prob_of.parent]
print(f"bot: {bot}")
dtop = scd_restricted_pcss_derivative(restriction=restriction, prob_of=prob_of, wrt=wrt, scd=scd, transit=transit)
print(f"dtop: {dtop}")
dbot = scd_restricted_subsplit_derivative(restriction=restriction, prob_of=prob_of.parent, wrt=wrt, scd=scd, transit=transit)
print(f"dbot: {dbot}")

scd_estimate_restricted_subsplit_derivative(restriction=restriction, prob_of=prob_of.parent, wrt=wrt, scd=scd, delta=delta)

tree_dist = scd_res.tree_distribution()
tree_dist2 = scd2_res.tree_distribution()
tree_num = tree_dist.feature_prob(prob_of)
tree_den = tree_dist.feature_prob(prob_of.parent)
tree_num2 = tree_dist2.feature_prob(prob_of)
tree_den2 = tree_dist2.feature_prob(prob_of.parent)
by_hand2 = (tree_num2/tree_den2 - tree_num/tree_den) / delta
print(f"{by_hand}\n{est_deriv}\n{theo_deriv}\n{by_hand2}")

scd_restricted_pcss_derivative(restriction=restriction, prob_of=prob_of, wrt=wrt, scd=scd, transit=s2s)
scd_estimate_restricted_pcss_derivative(restriction=restriction, prob_of=prob_of, wrt=wrt, scd=scd, delta=delta)
scd_restricted_subsplit_derivative(restriction=restriction, prob_of=prob_of.parent, wrt=wrt, scd=scd, transit=s2s)
scd_estimate_restricted_subsplit_derivative(restriction=restriction, prob_of=prob_of.parent, wrt=wrt, scd=scd, delta=delta)

## All x all


def brute6(scd: SCDSet, restriction, delta):
    theo_res = dict()
    est_res = dict()
    scd_res = scd.restrict(restriction)
    transit = scd.transit_probabilities()
    restricted_subsplit_probs = scd_res.subsplit_probabilities()
    restricted_pcss_probs = scd_res.pcss_probabilities()
    for wrt_for in scd.iter_pcss():
        if wrt_for not in theo_res:
            theo_res[wrt_for] = dict()
        if wrt_for not in est_res:
            est_res[wrt_for] = dict()
        parent = wrt_for.parent_clade()
        scd2 = scd.copy()
        scd2.add_log(wrt_for, delta)
        scd2.normalize(parent)
        scd2_res = scd2.restrict(restriction)
        # res_pcss_probs2 = scd2_res.pcss_probabilities()
        for prob_of_for in scd_res.iter_pcss():
            theo = scd_restricted_conditional_derivative(restriction=restriction, prob_of=prob_of_for, wrt=wrt_for, scd=scd, transit=transit,
                                                         restricted_scd=scd_res, restricted_subsplit_probs=restricted_subsplit_probs, restricted_pcss_probs=restricted_pcss_probs)
            theo_res[wrt_for][prob_of_for] = theo
            est = scd_estimate_restricted_conditional_derivative(restriction=restriction, prob_of=prob_of_for, wrt=wrt_for, scd=scd, delta=delta,
                                                                 scd_res=scd_res, scd2_res=scd2_res)
            est_res[wrt_for][prob_of_for] = est
            if not theo == est == 0.0:
                print(f"{prob_of_for} wrt {wrt_for}")
                print(f"Theo: {theo_res[wrt_for][prob_of_for]:8.4g}")
                print(f"Est:  {est_res[wrt_for][prob_of_for]:8.4g}")
    return theo_res, est_res


theo_res, est_res = brute6(scd, restriction, delta)

for wrt_for in theo_res:
    for prob_of_for in theo_res[wrt_for]:
        theo = theo_res[wrt_for][prob_of_for]
        est = est_res[wrt_for][prob_of_for]
        if abs(est) < 1e-12:
            abs_diff = abs(theo - est)
            if abs_diff > 1e-8:
                print(f"{prob_of_for} wrt {wrt_for}: est around zero, theo = {theo:10.6g}")
        elif abs(theo) < 1e-12:
            abs_diff = abs(theo - est)
            if abs_diff > 1e-8:
                print(f"{prob_of_for} wrt {wrt_for}: theo around zero, est={est:10.6g}")
        else:
            rel_diff = abs(theo - est)/abs(est)
            if rel_diff > 5e-6:
                print(f"{prob_of_for} wrt {wrt_for}: rel diff={rel_diff:10.6g} theo={theo:10.6g}")


# restricted KL derivative


reload(classes)
from classes import *


scd = SCDSet.random("ABCDE")
restriction = Clade("ABCD")
scd_small = SCDSet.random(restriction)
delta = 0.00001

transit = scd.transit_probabilities()

scd_res = scd.restrict(restriction)

wrt = random.choice(list(scd.iter_pcss()))
# possible = [subsplit for subsplit in scd.iter_subsplits() if subsplit.restrict(restriction) == prob_of.parent]
# prob_of_full = random.choice(possible)
# wrt_parent = random.choice(list(s2s[prob_of_full].keys()))
# wrt_child = random.choice(list(scd[wrt_parent].compatible_distribution(prob_of_full).keys()))
# wrt = PCSS(wrt_parent, wrt_child)

scd2 = scd.copy()
scd2.add_log(wrt, delta)
scd2.normalize()
scd2_res = scd2.restrict(restriction)

by_hand = (scd_small.kl_divergence(scd2_res) - scd_small.kl_divergence(scd_res)) / delta
by_hand

est_deriv = scd_estimate_restricted_kl_derivative(wrt=wrt, scd=scd, other=scd_small, delta=delta)
theo_deriv = scd_restricted_kl_derivative(wrt=wrt, scd=scd, other=scd_small, transit=transit)
print(abs(theo_deriv - est_deriv))
print(abs(theo_deriv - est_deriv)/abs(est_deriv))

tree_dist = scd_res.tree_distribution()
tree_dist2 = scd2_res.tree_distribution()
small_tree_dist = scd_small.tree_distribution()
by_hand2 = (small_tree_dist.kl_divergence(tree_dist2) - small_tree_dist.kl_divergence(tree_dist)) / delta
print(f"{by_hand}\n{est_deriv}\n{theo_deriv}\n{by_hand2}")

test_prob_of = PCSS(Subsplit("ADE", "B"), Subsplit("AD", "E"))
test_wrt=PCSS(Subsplit(Clade("ADE"), Clade("BF")), Subsplit(Clade("A"), Clade("DE")))
scd_restricted_conditional_derivative(restriction=restriction, prob_of=test_prob_of, wrt=test_wrt, scd=scd, transit=s2s)
scd_estimate_restricted_conditional_derivative(restriction=restriction, prob_of=test_prob_of, wrt=test_wrt, scd=scd, delta=delta)

## All


def brute7(scd: SCDSet, other: SCDSet, delta):
    theo_res = dict()
    est_res = dict()
    scd_res = scd.restrict(restriction)
    transit = scd.transit_probabilities()
    other_pcss_probs = other.pcss_probabilities()
    for wrt_for in scd.iter_pcss():
        parent = wrt_for.parent_clade()
        scd2 = scd.copy()
        scd2.add_log(wrt_for, delta)
        scd2.normalize(parent)
        scd2_res = scd2.restrict(restriction)
        theo = scd_restricted_kl_derivative(wrt=wrt_for, scd=scd, other=other, transit=transit,
                                            restricted_scd=scd_res, other_pcss_probs=other_pcss_probs)
        theo_res[wrt_for] = theo
        est = scd_estimate_restricted_kl_derivative(wrt=wrt_for, scd=scd, other=other, delta=delta,
                                                    scd_res=scd_res, scd2_res=scd2_res)
        est_res[wrt_for] = est
        if not theo == est == 0.0:
            print(f"KL wrt {wrt_for}")
            print(f"Theo: {theo_res[wrt_for]:8.4g}")
            print(f"Est:  {est_res[wrt_for]:8.4g}")
    return theo_res, est_res


theo_res, est_res = brute7(scd=scd, other=scd_small, delta=delta)

for wrt_for in theo_res:
    theo = theo_res[wrt_for]
    est = est_res[wrt_for]
    if abs(est) < 1e-12:
        abs_diff = abs(theo - est)
        if abs_diff > 1e-8:
            print(f"KL wrt {wrt_for}: est around zero, theo = {theo:10.6g}")
    elif abs(theo) < 1e-12:
        abs_diff = abs(theo - est)
        if abs_diff > 1e-8:
            print(f"KL wrt {wrt_for}: theo around zero, est={est:10.6g}")
    else:
        rel_diff = abs(theo - est)/abs(est)
        if rel_diff > 5e-6:
            print(f"KL wrt {wrt_for}: rel diff={rel_diff:10.6g} theo={theo:10.6g}")


# Testing new SCD backend

reload(classes)
from classes import *

# pcss_conditional_probs = {
#     PCSS(Subsplit("ABCD"), Subsplit("AB", "CD")): 0.4,
#     PCSS(Subsplit("ABCD"), Subsplit("ABC", "D")): 0.6,
#     PCSS(Subsplit("ABC", "D"), Subsplit("AB", "C")): 2 / 3,
#     PCSS(Subsplit("ABC", "D"), Subsplit("A", "BC")): 1 / 3,
#     PCSS(Subsplit("AB", "CD"), Subsplit("A", "B")): 1.0,
#     PCSS(Subsplit("AB", "C"), Subsplit("A", "B")): 1.0,
#     PCSS(Subsplit("A", "BC"), Subsplit("B", "C")): 1.0,
#     PCSS(Subsplit("AB", "CD"), Subsplit("C", "D")): 1.0,
# }
# scd = SCDSet(pcss_conditional_probs)
# scd.normalize()

scd = SCDSet.random("ABCDE")
tree_dist = scd.tree_distribution()
parent_probs = scd.subsplit_probabilities()
parent, prob = random.choice(list(parent_probs.items()))
print(f"{tree_dist.feature_prob(parent)}\n{prob}")
for parent, prob in parent_probs.items():
    tree_prob = tree_dist.feature_prob(parent)
    if abs(tree_prob - prob) > 1e-15:
        print(f"{parent}\n{tree_prob}\n{prob}")

pcss_probs = scd.pcss_probabilities()
pcss, prob = random.choice(list(pcss_probs.items()))
print(f"{tree_dist.prob_all([pcss.parent, pcss.child])}\n{prob}")
for pcss, prob in pcss_probs.items():
    tree_prob = tree_dist.prob_all([pcss.parent, pcss.child])
    if abs(tree_prob - prob) > 1e-15:
        print(f"{pcss}\n{tree_prob}\n{prob}")

subsplit_probs, pcss_probs = scd.probabilities()
for parent, prob in parent_probs.items():
    tree_prob = tree_dist.feature_prob(parent)
    if abs(tree_prob - prob) > 1e-15:
        print(f"{parent}\n{tree_prob}\n{prob}")
for pcss, prob in pcss_probs.items():
    tree_prob = tree_dist.prob_all([pcss.parent, pcss.child])
    if abs(tree_prob - prob) > 1e-15:
        print(f"{pcss}\n{tree_prob}\n{prob}")

# New SCD transit probabilities

reload(classes)
from classes import *

root_clade = Clade("ABCDE")
root_subsplit = Subsplit(root_clade)
scd = SCDSet.random(root_clade)
tree_dist = scd.tree_distribution()

t_probs = scd.transit_probabilities()
s_probs = scd.subsplit_probabilities()

parent = Subsplit(Clade("ABD"), Clade("C"))
s_probs[parent]
# t_probs[parent]
t_probs[parent][root_subsplit]

t_probs[parent][parent], 1.0

child = Subsplit(Clade("A"), Clade("BD"))
den = tree_dist.feature_prob(parent)
num = tree_dist.prob_all([parent, child])
t_probs[child][parent], num / den

## All x all experiment


def estimate_transit_probability(tree_dist: TreeDistribution, ancestor, descendant: Subsplit):
    descendant_clade = descendant.clade()
    if isinstance(ancestor, Subsplit):
        if ancestor == descendant:
            return 1.0
        if not (descendant_clade.issubset(ancestor.clade1) or descendant_clade.issubset(ancestor.clade2)):
            return 0.0
    if isinstance(ancestor, SubsplitClade):
        if not descendant_clade.issubset(ancestor.clade):
            return 0.0
    den = tree_dist.feature_prob(ancestor)
    if den == 0.0:
        return 0.0
    num = tree_dist.prob_all([ancestor, descendant])
    return num / den


estimate_transit_probability(tree_dist, parent, child)
t_probs[child][parent]

result = dict()
for ancestor in scd.iter_parents():
    print(f"Ancestor = {ancestor}")
    if ancestor not in result:
        result[ancestor] = dict()
    for descendant in scd.iter_subsplits():
        print(f"\tDescendant = {descendant}")
        est = estimate_transit_probability(tree_dist, ancestor, descendant)
        calc = t_probs.get(descendant, dict()).get(ancestor, 0.0)
        result[ancestor][descendant] = est, calc
        if abs(est - calc) > 1e-9:
            # print(f"{ancestor} to {descendant} : Error!")
            print(f"{ancestor} to {descendant} :\nEst={est}\nCal={calc}")
        else:
            pass
            # print(f"{ancestor} to {descendant} : Correct")

for ancestor in result:
    for descendant in result[ancestor]:
        est, calc = result[ancestor][descendant]
        if abs(est - calc) > 1e-9:
            # print(f"{ancestor} to {descendant} : Error!")
            print(f"{ancestor} to {descendant} :\nEst={est}\nCal={calc}")
        else:
            # print(f"{ancestor} to {descendant} : Correct")
            pass

ancestor = SubsplitClade(Subsplit("ABCDE", ""), Clade("ABCDE"))
descendant = Subsplit("ABE", "CD")
estimate_transit_probability(tree_dist, ancestor, descendant)
t_probs[descendant][ancestor]

result = dict()
for ancestor in scd.iter_subsplits():
    if ancestor not in result:
        result[ancestor] = dict()
    for descendant in scd.iter_subsplits():
        est = estimate_transit_probability(tree_dist, ancestor, descendant)
        calc = t_probs.get(descendant, dict()).get(ancestor, 0.0)
        result[ancestor][descendant] = est, calc
        if abs(est - calc) > 1e-9:
            # print(f"{ancestor} to {descendant} : Error!")
            print(f"{ancestor} to {descendant} :\nEst={est}\nCal={calc}")
        else:
            # print(f"{ancestor} to {descendant} : Correct")
            pass

# New PCSSSupport tests

reload(classes)
from classes import *

taxa = Clade("ABCDE")
tree = MyTree.random(taxa)
pcsss = PCSSSupport.from_tree(tree)
pcsss.to_set()
pcsss.get_taxon_set()

# new SCDSet restrict

reload(classes)
from classes import *

taxa = Clade("ABCDEF")
restriction = Clade("ABCDE")

scd = SCDSet.random(taxa)
t_probs = scd.transit_probabilities()

root_subsplit_clade = scd.root_subsplit_clade()
root_subsplit = scd.root_subsplit()
t_probs[root_subsplit]

# Fast KL gradient

from importlib import reload
import classes

reload(classes)
from classes import *


scd = SCDSet.random("ABCDE")
transit = scd.transit_probabilities()
restriction = Clade("ABCD")
scd_res = scd.restrict(restriction)
scd_small = SCDSet.random(restriction)

kl_grad = scd_restricted_kl_gradient(scd=scd, other=scd_small, transit=transit, restricted_self=scd_res)

theo_res = brute_kl_gradient(scd=scd, other=scd_small)

for wrt_for in theo_res:
    theo = theo_res[wrt_for]
    if abs(theo) < 1e-16:
        theo = 0.0
    grad = kl_grad[wrt_for]
    if abs(grad) < 1e-16:
        grad = 0.0
    print(f"KL wrt {wrt_for}: theo={theo:10.6g} grad={grad:10.6g}")
    # if abs(grad) < 1e-12:
    #     abs_diff = abs(theo - grad)
    #     if abs_diff > 1e-8:
    #         print(f"KL wrt {wrt_for}: grad around zero, theo = {theo:10.6g}")
    # elif abs(theo) < 1e-12:
    #     abs_diff = abs(theo - grad)
    #     if abs_diff > 1e-8:
    #         print(f"KL wrt {wrt_for}: theo around zero, grad={grad:10.6g}")
    # else:
    #     rel_diff = abs(theo - grad)/abs(grad)
    #     if rel_diff > 5e-6:
    #     print(f"KL wrt {wrt_for}: rel diff={rel_diff:10.6g} theo={theo:10.6g} grad={grad:10.6g}")

wrt_samp = random.choices(list(theo_res.keys()), k=10)
for wrt_for in wrt_samp:
    theo = theo_res[wrt_for]
    if abs(theo) < 1e-16:
        theo = 0.0
    grad = kl_grad[wrt_for]
    if abs(grad) < 1e-16:
        grad = 0.0
    est = scd_estimate_restricted_kl_derivative(wrt=wrt_for, scd=scd, other=scd_small, scd_res=scd_res)
    if abs(est) < 1e-16:
        est = 0.0
    print(f"KL wrt {wrt_for}: theo={theo:10.6g} grad={grad:10.6g} est={est:10.6g}")

# Taking apart the gradient


scd = SCDSet.random("ABCDE")
transit = scd.transit_probabilities()
restriction = Clade("ABCD")
scd_res = scd.restrict(restriction)
scd_small = SCDSet.random(restriction)

kl_grad = scd_restricted_kl_gradient(scd=scd, other=scd_small, transit=transit, restricted_self=scd_res)
kl_anc_grad = scd_restricted_kl_ancestor_gradient(scd=scd, other=scd_small, transit=transit, restricted_self=scd_res)
kl_jnt_grad = scd_restricted_kl_joint_gradient(scd=scd, other=scd_small, transit=transit, restricted_self=scd_res)

wrt = random.choice(list(kl_grad.keys()))
kl_grad[wrt]
kl_anc_grad[wrt]
kl_jnt_grad[wrt]
kl_anc_grad[wrt] - kl_jnt_grad[wrt]

kl_deriv = scd_restricted_kl_derivative(wrt=wrt, scd=scd, other=scd_small, transit=transit, restricted_scd=scd_res)
kl_anc_deriv = scd_restricted_kl_ancestor_derivative(wrt=wrt, scd=scd, other=scd_small, transit=transit, restricted_scd=scd_res)
kl_jnt_deriv = scd_restricted_kl_joint_derivative(wrt=wrt, scd=scd, other=scd_small, transit=transit, restricted_scd=scd_res)
kl_deriv
kl_anc_deriv
kl_jnt_deriv
kl_anc_deriv - kl_jnt_deriv

est = scd_estimate_restricted_kl_derivative(wrt=wrt, scd=scd, other=scd_small, scd_res=scd_res)
est

## conditional deriv cancellation experiment

wrts = random.choices(list(scd.iter_pcss()), k=20)
prob_ofs = random.choices(list(scd_res.iter_pcss()), k=20)
for wrt_for, prob_of_for in product(wrts, prob_ofs):
    orig = scd_restricted_conditional_derivative(
        restriction=restriction, prob_of=prob_of_for, wrt=wrt_for, scd=scd, transit=transit, restricted_scd=scd_res
    )
    if abs(orig) < 1e-17:
        orig = 0.0
    alt1 = scd_restricted_conditional_derivative_alt1(
        restriction=restriction, prob_of=prob_of_for, wrt=wrt_for, scd=scd, transit=transit, restricted_scd=scd_res
    )
    if abs(alt1) < 1e-17:
        alt1 = 0.0
    if orig == alt1 == 0.0:
        continue
    print(f"Prob of {prob_of_for} wrt {wrt_for}:")
    print(f"Orig: {orig:10.6g}")
    print(f"Alt1: {alt1:10.6g}")

def brute6alt1(scd: SCDSet, restriction, delta):
    theo_res = dict()
    est_res = dict()
    scd_res = scd.restrict(restriction)
    transit = scd.transit_probabilities()
    restricted_subsplit_probs = scd_res.subsplit_probabilities()
    restricted_pcss_probs = scd_res.pcss_probabilities()
    for wrt_for in scd.iter_pcss():
        if wrt_for not in theo_res:
            theo_res[wrt_for] = dict()
        if wrt_for not in est_res:
            est_res[wrt_for] = dict()
        parent = wrt_for.parent_clade()
        scd2 = scd.copy()
        scd2.add_log(wrt_for, delta)
        scd2.normalize(parent)
        scd2_res = scd2.restrict(restriction)
        # res_pcss_probs2 = scd2_res.pcss_probabilities()
        for prob_of_for in scd_res.iter_pcss():
            theo = scd_restricted_conditional_derivative_alt1(restriction=restriction, prob_of=prob_of_for, wrt=wrt_for, scd=scd, transit=transit,
                                                         restricted_scd=scd_res, restricted_subsplit_probs=restricted_subsplit_probs, restricted_pcss_probs=restricted_pcss_probs)
            theo_res[wrt_for][prob_of_for] = theo
            est = scd_estimate_restricted_conditional_derivative(restriction=restriction, prob_of=prob_of_for, wrt=wrt_for, scd=scd, delta=delta,
                                                                 scd_res=scd_res, scd2_res=scd2_res)
            est_res[wrt_for][prob_of_for] = est
            if not theo == est == 0.0:
                print(f"{prob_of_for} wrt {wrt_for}")
                print(f"Theo: {theo_res[wrt_for][prob_of_for]:8.4g}")
                print(f"Est:  {est_res[wrt_for][prob_of_for]:8.4g}")
    return theo_res, est_res


def brute6alt2(scd: SCDSet, restriction, delta):
    theo_res = dict()
    est_res = dict()
    scd_res = scd.restrict(restriction)
    transit = scd.transit_probabilities()
    restricted_subsplit_probs = scd_res.subsplit_probabilities()
    restricted_pcss_probs = scd_res.pcss_probabilities()
    for wrt_for in scd.iter_pcss():
        if wrt_for not in theo_res:
            theo_res[wrt_for] = dict()
        if wrt_for not in est_res:
            est_res[wrt_for] = dict()
        parent = wrt_for.parent_clade()
        scd2 = scd.copy()
        scd2.add_log(wrt_for, delta)
        scd2.normalize(parent)
        scd2_res = scd2.restrict(restriction)
        # res_pcss_probs2 = scd2_res.pcss_probabilities()
        for prob_of_for in scd_res.iter_pcss():
            theo = scd_restricted_conditional_derivative_alt2(restriction=restriction, prob_of=prob_of_for, wrt=wrt_for, scd=scd, transit=transit,
                                                         restricted_scd=scd_res, restricted_subsplit_probs=restricted_subsplit_probs, restricted_pcss_probs=restricted_pcss_probs)
            theo_res[wrt_for][prob_of_for] = theo
            est = scd_estimate_restricted_conditional_derivative(restriction=restriction, prob_of=prob_of_for, wrt=wrt_for, scd=scd, delta=delta,
                                                                 scd_res=scd_res, scd2_res=scd2_res)
            est_res[wrt_for][prob_of_for] = est
            if not theo == est == 0.0:
                print(f"{prob_of_for} wrt {wrt_for}")
                print(f"Theo: {theo_res[wrt_for][prob_of_for]:8.4g}")
                print(f"Est:  {est_res[wrt_for][prob_of_for]:8.4g}")
    return theo_res, est_res


delta = 0.00001
theo_res, est_res = brute6alt2(scd, restriction, delta)

for wrt_for in theo_res:
    for prob_of_for in theo_res[wrt_for]:
        theo = theo_res[wrt_for][prob_of_for]
        est = est_res[wrt_for][prob_of_for]
        if abs(est) < 1e-12:
            abs_diff = abs(theo - est)
            if abs_diff > 1e-8:
                print(f"{prob_of_for} wrt {wrt_for}: est around zero, theo = {theo:10.6g}")
        elif abs(theo) < 1e-12:
            abs_diff = abs(theo - est)
            if abs_diff > 1e-8:
                print(f"{prob_of_for} wrt {wrt_for}: theo around zero, est={est:10.6g}")
        else:
            rel_diff = abs(theo - est)/abs(est)
            if rel_diff > 5e-6:
                print(f"{prob_of_for} wrt {wrt_for}: rel diff={rel_diff:10.6g} theo={theo:10.6g}")

kl_deriv_alt1 = scd_restricted_kl_derivative_alt1(wrt=wrt, scd=scd, other=scd_small, transit=transit, restricted_scd=scd_res)
kl_deriv_alt1

kl_deriv_alt2 = scd_restricted_kl_derivative_alt2(wrt=wrt, scd=scd, other=scd_small, transit=transit, restricted_scd=scd_res)
kl_deriv_alt2


