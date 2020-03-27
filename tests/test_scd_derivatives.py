import unittest
import sys
import time

sys.path.append("..")
from derivatives import *


class TestBrute(unittest.TestCase):
    def test_subsplit_derivative(self):
        scd = SCDSet.random("ABCDE")
        delta = 0.00001

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

        abs_flags = 0
        rel_flags = 0
        for wrt_for in theo_res_simple:
            for prob_of_for in theo_res_simple[wrt_for]:
                theo = theo_res_simple[wrt_for][prob_of_for]
                est = est_res_simple[wrt_for][prob_of_for]
                if abs(est) < 1e-12:
                    abs_diff = abs(theo - est)
                    if abs_diff > 1e-8:
                        abs_flags += 1
                        # print(f"{prob_of_for} wrt {wrt_for}: est zero, theo={theo:10.6g}")
                elif abs(theo) < 1e-12:
                    abs_diff = abs(theo - est)
                    if abs_diff > 1e-8:
                        abs_flags += 1
                        # print(f"{prob_of_for} wrt {wrt_for}: theo zero, est={est:10.6g}")
                else:
                    rel_diff = abs(theo - est) / abs(est)
                    if rel_diff > 5e-5:
                        rel_flags += 1
                        # print(f"{prob_of_for} wrt {wrt_for}: rel diff={rel_diff:10.6g}")
        self.assertLess(abs_flags, 3)
        self.assertLess(rel_flags, 20)

    def test_subsplit_to_subsplit_derivative(self):
        scd = SCDSet.random("ABCDE")
        delta = 0.00001

        theo_res_simple = dict()
        est_res_simple = dict()
        transit_for = scd.transit_probabilities()
        for wrt_for in scd.iter_pcss():
            if wrt_for not in theo_res_simple:
                theo_res_simple[wrt_for] = dict()
            if wrt_for not in est_res_simple:
                est_res_simple[wrt_for] = dict()
            scd2_for = scd.copy()
            scd2_for.add_log(wrt_for, delta)
            scd2_for.normalize(wrt_for.parent_clade())
            transit2_for = scd2_for.transit_probabilities()
            for cond_on_for, prob_of_for in product(scd.iter_subsplits(include_root=True),
                                                    scd.iter_subsplits(include_root=True)):
                theo = scd_subsplit_to_subsplit_cond_derivative(
                    prob_of=prob_of_for, cond_on=cond_on_for, wrt=wrt_for, scd=scd, transit=transit_for
                )
                theo_res_simple[wrt_for][(cond_on_for, prob_of_for)] = theo
                est = scd_estimate_subsplit_to_subsplit_cond_derivative(
                    prob_of=prob_of_for, cond_on=cond_on_for, wrt=wrt_for, scd=scd, delta=delta,
                    transit=transit_for, transit2=transit2_for
                )
                est_res_simple[wrt_for][(cond_on_for, prob_of_for)] = est

        abs_flags = 0
        rel_flags = 0
        for wrt_for in theo_res_simple:
            for (cond_on_for, prob_of_for) in theo_res_simple[wrt_for]:
                theo = theo_res_simple[wrt_for][(cond_on_for, prob_of_for)]
                est = est_res_simple[wrt_for][(cond_on_for, prob_of_for)]
                if abs(est) < 1e-12:
                    abs_diff = abs(theo - est)
                    if abs_diff > 1e-8:
                        abs_flags += 1
                elif abs(theo) < 1e-12:
                    abs_diff = abs(theo - est)
                    if abs_diff > 1e-8:
                        abs_flags += 1
                else:
                    rel_diff = abs(theo - est) / abs(est)
                    if rel_diff > 5e-5:
                        rel_flags += 1
        self.assertLess(abs_flags, 3)
        self.assertLess(rel_flags, 20)

    def test_subsplit_via_subsplit_derivative(self):
        scd = SCDSet.random("ABCDE")
        delta = 0.00001

        theo_res_simple = dict()
        est_res_simple = dict()
        transit_for = scd.transit_probabilities()
        for wrt_for in scd.iter_pcss():
            if wrt_for not in theo_res_simple:
                theo_res_simple[wrt_for] = dict()
            if wrt_for not in est_res_simple:
                est_res_simple[wrt_for] = dict()
            scd2_for = scd.copy()
            scd2_for.add_log(wrt_for, delta)
            scd2_for.normalize(wrt_for.parent_clade())
            transit2_for = scd2_for.transit_probabilities()
            for via_for, prob_of_for in product(scd.iter_subsplits(include_root=True),
                                                scd.iter_subsplits(include_root=True)):
                theo = scd_subsplit_via_subsplit_derivative(
                    prob_of=prob_of_for, via=via_for, wrt=wrt_for, scd=scd, transit=transit_for
                )
                theo_res_simple[wrt_for][(via_for, prob_of_for)] = theo
                est = scd_estimate_subsplit_via_subsplit_derivative(
                    prob_of=prob_of_for, via=via_for, wrt=wrt_for, scd=scd, delta=delta,
                    transit=transit_for, transit2=transit2_for
                )
                est_res_simple[wrt_for][(via_for, prob_of_for)] = est

        abs_flags = 0
        rel_flags = 0
        for wrt_for in theo_res_simple:
            for (via_for, prob_of_for) in theo_res_simple[wrt_for]:
                theo = theo_res_simple[wrt_for][(via_for, prob_of_for)]
                est = est_res_simple[wrt_for][(via_for, prob_of_for)]
                if abs(est) < 1e-12:
                    abs_diff = abs(theo - est)
                    if abs_diff > 1e-8:
                        abs_flags += 1
                elif abs(theo) < 1e-12:
                    abs_diff = abs(theo - est)
                    if abs_diff > 1e-8:
                        abs_flags += 1
                else:
                    rel_diff = abs(theo - est) / abs(est)
                    if rel_diff > 5e-5:
                        rel_flags += 1
        self.assertLess(abs_flags, 3)
        self.assertLess(rel_flags, 20)

    def test_restricted_subsplit_derivative(self):
        scd = SCDSet.random("ABCDE")
        restriction = Clade("ABCD")
        delta = 0.00001

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
                theo = scd_restricted_subsplit_derivative(restriction=restriction, prob_of=prob_of_for, wrt=wrt_for,
                                                          scd=scd, transit=transit)
                theo_res[wrt_for][prob_of_for] = theo
                est = scd_estimate_restricted_subsplit_derivative(restriction=restriction, prob_of=prob_of_for,
                                                                  wrt=wrt_for, scd=scd, delta=delta, uncond=uncond,
                                                                  uncond2=uncond2)
                est_res[wrt_for][prob_of_for] = est

        abs_flags = 0
        rel_flags = 0
        for wrt_for in theo_res:
            for prob_of_for in theo_res[wrt_for]:
                theo = theo_res[wrt_for][prob_of_for]
                est = est_res[wrt_for][prob_of_for]
                if abs(est) < 1e-12:
                    abs_diff = abs(theo - est)
                    if abs_diff > 1e-8:
                        abs_flags += 1
                        # print(f"{prob_of_for} wrt {wrt_for}: est zero, theo={theo:10.6g}")
                elif abs(theo) < 1e-12:
                    abs_diff = abs(theo - est)
                    if abs_diff > 1e-8:
                        abs_flags += 1
                        # print(f"{prob_of_for} wrt {wrt_for}: theo zero, est={est:10.6g}")
                else:
                    rel_diff = abs(theo - est) / abs(est)
                    if rel_diff > 5e-5:
                        rel_flags += 1
                        # print(f"{prob_of_for} wrt {wrt_for}: rel diff={rel_diff:10.6g}")
        self.assertLess(abs_flags, 3)
        self.assertLess(rel_flags, 20)

    def test_restricted_pcss_derivative(self):
        scd = SCDSet.random("ABCDE")
        restriction = Clade("ABCD")
        delta = 0.00001

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
                theo = scd_restricted_pcss_derivative(restriction=restriction, prob_of=prob_of_for, wrt=wrt_for,
                                                      scd=scd, transit=transit)
                theo_res[wrt_for][prob_of_for] = theo
                est = scd_estimate_restricted_pcss_derivative(restriction=restriction, prob_of=prob_of_for, wrt=wrt_for,
                                                              scd=scd, delta=delta, res_pcss_probs=res_pcss_probs,
                                                              res_pcss_probs2=res_pcss_probs2)
                est_res[wrt_for][prob_of_for] = est

        abs_flags = 0
        rel_flags = 0
        for wrt_for in theo_res:
            for prob_of_for in theo_res[wrt_for]:
                theo = theo_res[wrt_for][prob_of_for]
                est = est_res[wrt_for][prob_of_for]
                if abs(est) < 1e-12:
                    abs_diff = abs(theo - est)
                    if abs_diff > 1e-8:
                        abs_flags += 1
                        # print(f"{prob_of_for} wrt {wrt_for}: est zero, theo={theo:10.6g}")
                elif abs(theo) < 1e-12:
                    abs_diff = abs(theo - est)
                    if abs_diff > 1e-8:
                        abs_flags += 1
                        # print(f"{prob_of_for} wrt {wrt_for}: theo zero, est={est:10.6g}")
                else:
                    rel_diff = abs(theo - est) / abs(est)
                    if rel_diff > 5e-5:
                        rel_flags += 1
                        # print(f"{prob_of_for} wrt {wrt_for}: rel diff={rel_diff:10.6g}")
        self.assertLess(abs_flags, 3)
        self.assertLess(rel_flags, 20)

    def test_restricted_conditional_derivative(self):
        scd = SCDSet.random("ABCDE")
        restriction = Clade("ABCD")
        delta = 0.00001

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
            for prob_of_for in scd_res.iter_pcss():
                theo = scd_restricted_conditional_derivative(restriction=restriction, prob_of=prob_of_for, wrt=wrt_for,
                                                             scd=scd, transit=transit,
                                                             restricted_scd=scd_res,
                                                             restricted_subsplit_probs=restricted_subsplit_probs,
                                                             restricted_pcss_probs=restricted_pcss_probs)
                theo_res[wrt_for][prob_of_for] = theo
                est = scd_estimate_restricted_conditional_derivative(restriction=restriction, prob_of=prob_of_for,
                                                                     wrt=wrt_for, scd=scd, delta=delta,
                                                                     scd_res=scd_res, scd2_res=scd2_res)
                est_res[wrt_for][prob_of_for] = est

        abs_flags = 0
        rel_flags = 0
        for wrt_for in theo_res:
            for prob_of_for in theo_res[wrt_for]:
                theo = theo_res[wrt_for][prob_of_for]
                est = est_res[wrt_for][prob_of_for]
                if abs(est) < 1e-12:
                    abs_diff = abs(theo - est)
                    if abs_diff > 1e-8:
                        abs_flags += 1
                elif abs(theo) < 1e-12:
                    abs_diff = abs(theo - est)
                    if abs_diff > 1e-8:
                        abs_flags += 1
                else:
                    rel_diff = abs(theo - est) / abs(est)
                    if rel_diff > 5e-5:
                        rel_flags += 1
        self.assertLess(abs_flags, 3)
        self.assertLess(rel_flags, 20)

    def test_restricted_kl_derivative(self):
        scd = SCDSet.random("ABCDE")
        restriction = Clade("ABCD")
        other = SCDSet.random(restriction)
        delta = 0.00001

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

        abs_flags = 0
        rel_flags = 0
        for wrt_for in theo_res:
            theo = theo_res[wrt_for]
            est = est_res[wrt_for]
            if abs(est) < 1e-12:
                abs_diff = abs(theo - est)
                if abs_diff > 1e-8:
                    abs_flags += 1
            elif abs(theo) < 1e-12:
                abs_diff = abs(theo - est)
                if abs_diff > 1e-8:
                    abs_flags += 1
            else:
                rel_diff = abs(theo - est) / abs(est)
                if rel_diff > 5e-5:
                    rel_flags += 1
        self.assertLess(abs_flags, 3)
        self.assertLess(rel_flags, 20)

    def test_restricted_kl_gradient(self):
        scd = SCDSet.random("ABCDE")
        restriction = Clade("ABCD")
        scd_res = scd.restrict(restriction)
        transit = scd.transit_probabilities()
        other = SCDSet.random(restriction)
        other_pcss_probs = other.pcss_probabilities()

        kl_grad = scd_restricted_kl_gradient(scd=scd, other=other, transit=transit, restricted_self=scd_res)
        brute_grad = brute_kl_gradient(scd=scd, other=other, scd_res=scd_res, transit=transit,
                                       other_pcss_probs=other_pcss_probs)

        n_flags = 0
        for wrt_for in brute_grad:
            theo = brute_grad[wrt_for]
            if abs(theo) < 1e-16:
                theo = 0.0
            grad = kl_grad[wrt_for]
            if abs(grad) < 1e-16:
                grad = 0.0
            if abs(theo - grad) > 1e-12:
                n_flags += 1
        self.assertLess(n_flags, 5)

    def test_restricted_conditional_derivative_alt1(self):
        scd = SCDSet.random("ABCDE")
        restriction = Clade("ABCD")
        delta = 0.00001

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
            for prob_of_for in scd_res.iter_pcss():
                theo = scd_restricted_conditional_derivative_alt1(restriction=restriction, prob_of=prob_of_for,
                                                                  wrt=wrt_for, scd=scd, transit=transit,
                                                                  restricted_scd=scd_res,
                                                                  restricted_subsplit_probs=restricted_subsplit_probs,
                                                                  restricted_pcss_probs=restricted_pcss_probs)
                theo_res[wrt_for][prob_of_for] = theo
                est = scd_estimate_restricted_conditional_derivative(restriction=restriction, prob_of=prob_of_for,
                                                                     wrt=wrt_for, scd=scd, delta=delta,
                                                                     scd_res=scd_res, scd2_res=scd2_res)
                est_res[wrt_for][prob_of_for] = est

        abs_flags = 0
        rel_flags = 0
        for wrt_for in theo_res:
            for prob_of_for in theo_res[wrt_for]:
                theo = theo_res[wrt_for][prob_of_for]
                est = est_res[wrt_for][prob_of_for]
                if abs(est) < 1e-12:
                    abs_diff = abs(theo - est)
                    if abs_diff > 1e-8:
                        abs_flags += 1
                elif abs(theo) < 1e-12:
                    abs_diff = abs(theo - est)
                    if abs_diff > 1e-8:
                        abs_flags += 1
                else:
                    rel_diff = abs(theo - est) / abs(est)
                    if rel_diff > 5e-5:
                        rel_flags += 1
        self.assertLess(abs_flags, 3)
        self.assertLess(rel_flags, 20)

    def test_restricted_conditional_derivative_alt2(self):
        scd = SCDSet.random("ABCDE")
        restriction = Clade("ABCD")
        delta = 0.00001

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
            for prob_of_for in scd_res.iter_pcss():
                theo = scd_restricted_conditional_derivative_alt2(restriction=restriction, prob_of=prob_of_for,
                                                                  wrt=wrt_for, scd=scd, transit=transit,
                                                                  restricted_scd=scd_res,
                                                                  restricted_subsplit_probs=restricted_subsplit_probs,
                                                                  restricted_pcss_probs=restricted_pcss_probs)
                theo_res[wrt_for][prob_of_for] = theo
                est = scd_estimate_restricted_conditional_derivative(restriction=restriction, prob_of=prob_of_for,
                                                                     wrt=wrt_for, scd=scd, delta=delta,
                                                                     scd_res=scd_res, scd2_res=scd2_res)
                est_res[wrt_for][prob_of_for] = est

        abs_flags = 0
        rel_flags = 0
        for wrt_for in theo_res:
            for prob_of_for in theo_res[wrt_for]:
                theo = theo_res[wrt_for][prob_of_for]
                est = est_res[wrt_for][prob_of_for]
                if abs(est) < 1e-12:
                    abs_diff = abs(theo - est)
                    if abs_diff > 1e-8:
                        abs_flags += 1
                elif abs(theo) < 1e-12:
                    abs_diff = abs(theo - est)
                    if abs_diff > 1e-8:
                        abs_flags += 1
                else:
                    rel_diff = abs(theo - est) / abs(est)
                    if rel_diff > 5e-5:
                        rel_flags += 1
        self.assertLess(abs_flags, 3)
        self.assertLess(rel_flags, 20)

    def test_restricted_kl_derivative_alt1(self):
        scd = SCDSet.random("ABCDE")
        restriction = Clade("ABCD")
        other = SCDSet.random(restriction)
        # delta = 0.00001

        theo_res = dict()
        alt1_res = dict()
        scd_res = scd.restrict(restriction)
        transit = scd.transit_probabilities()
        other_pcss_probs = other.pcss_probabilities()
        for wrt_for in scd.iter_pcss():
            # parent = wrt_for.parent_clade()
            # scd2 = scd.copy()
            # scd2.add_log(wrt_for, delta)
            # scd2.normalize(parent)
            # scd2_res = scd2.restrict(restriction)
            theo = scd_restricted_kl_derivative(wrt=wrt_for, scd=scd, other=other, transit=transit,
                                                restricted_scd=scd_res, other_pcss_probs=other_pcss_probs)
            theo_res[wrt_for] = theo
            alt1 = scd_restricted_kl_derivative_alt1(wrt=wrt_for, scd=scd, other=other, transit=transit,
                                                     restricted_scd=scd_res, other_pcss_probs=other_pcss_probs)
            alt1_res[wrt_for] = alt1

        abs_flags = 0
        rel_flags = 0
        for wrt_for in theo_res:
            theo = theo_res[wrt_for]
            alt1 = alt1_res[wrt_for]
            if abs(alt1) < 1e-12:
                abs_diff = abs(theo - alt1)
                if abs_diff > 1e-9:
                    abs_flags += 1
            elif abs(theo) < 1e-12:
                abs_diff = abs(theo - alt1)
                if abs_diff > 1e-9:
                    abs_flags += 1
            else:
                rel_diff = abs(theo - alt1) / abs(alt1)
                if rel_diff > 5e-6:
                    rel_flags += 1
        self.assertLess(abs_flags, 3)
        self.assertLess(rel_flags, 20)

    def test_restricted_kl_derivative_alt2(self):
        scd = SCDSet.random("ABCDE")
        restriction = Clade("ABCD")
        other = SCDSet.random(restriction)
        # delta = 0.00001

        theo_res = dict()
        alt2_res = dict()
        scd_res = scd.restrict(restriction)
        transit = scd.transit_probabilities()
        other_pcss_probs = other.pcss_probabilities()
        for wrt_for in scd.iter_pcss():
            # parent = wrt_for.parent_clade()
            # scd2 = scd.copy()
            # scd2.add_log(wrt_for, delta)
            # scd2.normalize(parent)
            # scd2_res = scd2.restrict(restriction)
            theo = scd_restricted_kl_derivative(wrt=wrt_for, scd=scd, other=other, transit=transit,
                                                restricted_scd=scd_res, other_pcss_probs=other_pcss_probs)
            theo_res[wrt_for] = theo
            alt2 = scd_restricted_kl_derivative_alt2(wrt=wrt_for, scd=scd, other=other, transit=transit,
                                                     restricted_scd=scd_res, other_pcss_probs=other_pcss_probs)
            alt2_res[wrt_for] = alt2

        abs_flags = 0
        rel_flags = 0
        for wrt_for in theo_res:
            theo = theo_res[wrt_for]
            alt2 = alt2_res[wrt_for]
            if abs(alt2) < 1e-12:
                abs_diff = abs(theo - alt2)
                if abs_diff > 1e-9:
                    abs_flags += 1
            elif abs(theo) < 1e-12:
                abs_diff = abs(theo - alt2)
                if abs_diff > 1e-9:
                    abs_flags += 1
            else:
                rel_diff = abs(theo - alt2) / abs(alt2)
                if rel_diff > 5e-6:
                    rel_flags += 1
        self.assertLess(abs_flags, 3)
        self.assertLess(rel_flags, 20)

    def test_restricted_kl_derivative_alt3(self):
        scd = SCDSet.random("ABCDE")
        restriction = Clade("ABCD")
        other = SCDSet.random(restriction)
        # delta = 0.00001

        theo_res = dict()
        alt3_res = dict()
        scd_res = scd.restrict(restriction)
        transit = scd.transit_probabilities()
        other_pcss_probs = other.pcss_probabilities()
        for wrt_for in scd.iter_pcss():
            # parent = wrt_for.parent_clade()
            # scd2 = scd.copy()
            # scd2.add_log(wrt_for, delta)
            # scd2.normalize(parent)
            # scd2_res = scd2.restrict(restriction)
            theo = scd_restricted_kl_derivative(wrt=wrt_for, scd=scd, other=other, transit=transit,
                                                restricted_scd=scd_res, other_pcss_probs=other_pcss_probs)
            theo_res[wrt_for] = theo
            alt3 = scd_restricted_kl_derivative_alt3(wrt=wrt_for, scd=scd, other=other, transit=transit,
                                                     restricted_scd=scd_res, other_pcss_probs=other_pcss_probs)
            alt3_res[wrt_for] = alt3

        abs_flags = 0
        rel_flags = 0
        for wrt_for in theo_res:
            theo = theo_res[wrt_for]
            alt3 = alt3_res[wrt_for]
            if abs(alt3) < 1e-12:
                abs_diff = abs(theo - alt3)
                if abs_diff > 1e-9:
                    abs_flags += 1
            elif abs(theo) < 1e-12:
                abs_diff = abs(theo - alt3)
                if abs_diff > 1e-9:
                    abs_flags += 1
            else:
                rel_diff = abs(theo - alt3) / abs(alt3)
                if rel_diff > 5e-6:
                    rel_flags += 1
        self.assertLess(abs_flags, 3)
        self.assertLess(rel_flags, 20)

    def test_restricted_all_subsplits_derivative(self):
        scd = SCDSet.random("ABCDE")
        restriction = Clade("ABCD")

        theo_res = dict()
        asd_res = dict()
        scd_res = scd.restrict(restriction)
        transit = scd.transit_probabilities()
        for wrt_for in scd.iter_pcss():
            if wrt_for not in theo_res:
                theo_res[wrt_for] = dict()
            if wrt_for not in asd_res:
                asd_res[wrt_for] = dict()
            asd = scd_restricted_all_subsplits_derivative(restriction=restriction, wrt=wrt_for, scd=scd,
                                                          transit=transit)
            for prob_of_for in scd_res.iter_subsplits(include_root=True):
                theo = scd_restricted_subsplit_derivative(restriction=restriction, prob_of=prob_of_for, wrt=wrt_for,
                                                          scd=scd, transit=transit)
                theo_res[wrt_for][prob_of_for] = theo
                asd_res[wrt_for][prob_of_for] = asd[prob_of_for]

        abs_flags = 0
        rel_flags = 0
        for wrt_for in theo_res:
            for prob_of_for in theo_res[wrt_for]:
                theo = theo_res[wrt_for][prob_of_for]
                asd = asd_res[wrt_for][prob_of_for]
                if abs(asd) < 1e-12:
                    abs_diff = abs(theo - asd)
                    if abs_diff > 1e-8:
                        abs_flags += 1
                elif abs(theo) < 1e-12:
                    abs_diff = abs(theo - asd)
                    if abs_diff > 1e-8:
                        abs_flags += 1
                else:
                    rel_diff = abs(theo - asd) / abs(asd)
                    if rel_diff > 1e-7:
                        rel_flags += 1
        self.assertLess(abs_flags, 3)
        self.assertLess(rel_flags, 5)


class TestDecomposition(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        scd = SCDSet.random("ABCDE")
        transit = scd.transit_probabilities()
        restriction = Clade("ABCD")
        scd_res = scd.restrict(restriction)
        scd_small = SCDSet.random(restriction)

        kl_grad = scd_restricted_kl_gradient(scd=scd, other=scd_small, transit=transit, restricted_self=scd_res)
        kl_anc_grad = scd_restricted_kl_ancestor_gradient(scd=scd, other=scd_small, transit=transit,
                                                          restricted_self=scd_res)
        kl_jnt_grad = scd_restricted_kl_joint_gradient(scd=scd, other=scd_small, transit=transit,
                                                       restricted_self=scd_res)
        cls.scd = scd
        cls.transit = transit
        cls.restriction = restriction
        cls.scd_res = scd_res
        cls.scd_small = scd_small
        cls.kl_grad = kl_grad
        cls.kl_anc_grad = kl_anc_grad
        cls.kl_jnt_grad = kl_jnt_grad

    def test_gradient_equals_decomposed_gradient(self):
        # scd = self.scd
        # transit = self.transit
        # scd_res = self.scd_res
        # scd_small = self.scd_small
        kl_grad = self.kl_grad
        kl_anc_grad = self.kl_anc_grad
        kl_jnt_grad = self.kl_jnt_grad

        wrt_samps = random.choices(list(kl_grad.keys()), k=10)
        for wrt in wrt_samps:
            self.assertAlmostEqual(kl_grad[wrt], kl_anc_grad[wrt] - kl_jnt_grad[wrt],
                                   msg="Gradient not equal to decomposed gradient.")

    def test_gradient_equals_derivative(self):
        scd = self.scd
        transit = self.transit
        scd_res = self.scd_res
        scd_small = self.scd_small
        kl_grad = self.kl_grad
        # kl_anc_grad = self.kl_anc_grad
        # kl_jnt_grad = self.kl_jnt_grad

        wrt_samps = random.choices(list(kl_grad.keys()), k=10)
        for wrt in wrt_samps:
            kl_deriv = scd_restricted_kl_derivative(wrt=wrt, scd=scd, other=scd_small, transit=transit,
                                                    restricted_scd=scd_res)
            # kl_anc_deriv = scd_restricted_kl_ancestor_derivative(wrt=wrt, scd=scd, other=scd_small, transit=transit,
            #                                                      restricted_scd=scd_res)
            # kl_jnt_deriv = scd_restricted_kl_joint_derivative(wrt=wrt, scd=scd, other=scd_small, transit=transit,
            #                                                   restricted_scd=scd_res)
            self.assertAlmostEqual(kl_deriv, kl_grad[wrt],
                                   msg="Derivative not equal to gradient component.")

    def test_derivative_equals_decomposed_derivative(self):
        scd = self.scd
        transit = self.transit
        scd_res = self.scd_res
        scd_small = self.scd_small
        kl_grad = self.kl_grad
        # kl_anc_grad = self.kl_anc_grad
        # kl_jnt_grad = self.kl_jnt_grad

        wrt_samps = random.choices(list(kl_grad.keys()), k=10)
        for wrt in wrt_samps:
            kl_deriv = scd_restricted_kl_derivative(wrt=wrt, scd=scd, other=scd_small, transit=transit,
                                                    restricted_scd=scd_res)
            kl_anc_deriv = scd_restricted_kl_ancestor_derivative(wrt=wrt, scd=scd, other=scd_small, transit=transit,
                                                                 restricted_scd=scd_res)
            kl_jnt_deriv = scd_restricted_kl_joint_derivative(wrt=wrt, scd=scd, other=scd_small, transit=transit,
                                                              restricted_scd=scd_res)
            self.assertAlmostEqual(kl_deriv, kl_anc_deriv - kl_jnt_deriv)

    def test_ancestor_derivative_equals_ancestor_gradient(self):
        scd = self.scd
        transit = self.transit
        scd_res = self.scd_res
        scd_small = self.scd_small
        kl_grad = self.kl_grad
        kl_anc_grad = self.kl_anc_grad
        # kl_jnt_grad = self.kl_jnt_grad

        wrt_samps = random.choices(list(kl_grad.keys()), k=10)
        for wrt in wrt_samps:
            # kl_deriv = scd_restricted_kl_derivative(wrt=wrt, scd=scd, other=scd_small, transit=transit,
            #                                         restricted_scd=scd_res)
            kl_anc_deriv = scd_restricted_kl_ancestor_derivative(wrt=wrt, scd=scd, other=scd_small, transit=transit,
                                                                 restricted_scd=scd_res)
            # kl_jnt_deriv = scd_restricted_kl_joint_derivative(wrt=wrt, scd=scd, other=scd_small, transit=transit,
            #                                                   restricted_scd=scd_res)
            self.assertAlmostEqual(kl_anc_grad[wrt], kl_anc_deriv)

    def test_joint_derivative_equals_joint_gradient(self):
        scd = self.scd
        transit = self.transit
        scd_res = self.scd_res
        scd_small = self.scd_small
        kl_grad = self.kl_grad
        # kl_anc_grad = self.kl_anc_grad
        kl_jnt_grad = self.kl_jnt_grad

        wrt_samps = random.choices(list(kl_grad.keys()), k=10)
        for wrt in wrt_samps:
            # kl_deriv = scd_restricted_kl_derivative(wrt=wrt, scd=scd, other=scd_small, transit=transit,
            #                                         restricted_scd=scd_res)
            # kl_anc_deriv = scd_restricted_kl_ancestor_derivative(wrt=wrt, scd=scd, other=scd_small, transit=transit,
            #                                                      restricted_scd=scd_res)
            kl_jnt_deriv = scd_restricted_kl_joint_derivative(wrt=wrt, scd=scd, other=scd_small, transit=transit,
                                                              restricted_scd=scd_res)
            self.assertAlmostEqual(kl_jnt_grad[wrt], kl_jnt_deriv)

    def test_derivative_equals_estimated_derivative(self):
        scd = self.scd
        transit = self.transit
        scd_res = self.scd_res
        scd_small = self.scd_small
        kl_grad = self.kl_grad
        # kl_anc_grad = self.kl_anc_grad
        # kl_jnt_grad = self.kl_jnt_grad

        wrt_samps = random.choices(list(kl_grad.keys()), k=10)
        for wrt in wrt_samps:
            kl_deriv = scd_restricted_kl_derivative(wrt=wrt, scd=scd, other=scd_small, transit=transit,
                                                    restricted_scd=scd_res)
            # kl_anc_deriv = scd_restricted_kl_ancestor_derivative(wrt=wrt, scd=scd, other=scd_small, transit=transit,
            #                                                      restricted_scd=scd_res)
            # kl_jnt_deriv = scd_restricted_kl_joint_derivative(wrt=wrt, scd=scd, other=scd_small, transit=transit,
            #                                                   restricted_scd=scd_res)
            est = scd_estimate_restricted_kl_derivative(wrt=wrt, scd=scd, other=scd_small, scd_res=scd_res)
            self.assertAlmostEqual(kl_deriv, est, 5)

    def test_gradient_equals_estimated_derivative(self):
        scd = self.scd
        transit = self.transit
        scd_res = self.scd_res
        scd_small = self.scd_small
        kl_grad = self.kl_grad
        # kl_anc_grad = self.kl_anc_grad
        # kl_jnt_grad = self.kl_jnt_grad

        wrt_samps = random.choices(list(kl_grad.keys()), k=10)
        for wrt in wrt_samps:
            # kl_deriv = scd_restricted_kl_derivative(wrt=wrt, scd=scd, other=scd_small, transit=transit,
            #                                         restricted_scd=scd_res)
            # kl_anc_deriv = scd_restricted_kl_ancestor_derivative(wrt=wrt, scd=scd, other=scd_small, transit=transit,
            #                                                      restricted_scd=scd_res)
            # kl_jnt_deriv = scd_restricted_kl_joint_derivative(wrt=wrt, scd=scd, other=scd_small, transit=transit,
            #                                                   restricted_scd=scd_res)
            est = scd_estimate_restricted_kl_derivative(wrt=wrt, scd=scd, other=scd_small, scd_res=scd_res)
            self.assertAlmostEqual(kl_grad[wrt], est, 5)

    def test_restricted_kl_gradient_decomposition(self):
        scd = self.scd
        transit = self.transit
        scd_res = self.scd_res
        scd_small = self.scd_small
        kl_grad = self.kl_grad
        kl_anc_grad = self.kl_anc_grad
        kl_jnt_grad = self.kl_jnt_grad

        wrt_samps = random.choices(list(kl_grad.keys()), k=10)
        for wrt in wrt_samps:
            self.assertAlmostEqual(kl_grad[wrt], kl_anc_grad[wrt] - kl_jnt_grad[wrt],
                                   msg="Gradient not equal to decomposed gradient.")

            kl_deriv = scd_restricted_kl_derivative(wrt=wrt, scd=scd, other=scd_small, transit=transit,
                                                    restricted_scd=scd_res)
            kl_anc_deriv = scd_restricted_kl_ancestor_derivative(wrt=wrt, scd=scd, other=scd_small, transit=transit,
                                                                 restricted_scd=scd_res)
            kl_jnt_deriv = scd_restricted_kl_joint_derivative(wrt=wrt, scd=scd, other=scd_small, transit=transit,
                                                              restricted_scd=scd_res)
            self.assertAlmostEqual(kl_deriv, kl_grad[wrt],
                                   msg="Derivative not equal to gradient component.")
            self.assertAlmostEqual(kl_deriv, kl_anc_deriv - kl_jnt_deriv)
            self.assertAlmostEqual(kl_anc_grad[wrt], kl_anc_deriv)
            self.assertAlmostEqual(kl_jnt_grad[wrt], kl_jnt_deriv)

            est = scd_estimate_restricted_kl_derivative(wrt=wrt, scd=scd, other=scd_small, scd_res=scd_res)
            self.assertAlmostEqual(kl_deriv, est, 5)
            self.assertAlmostEqual(kl_grad[wrt], est, 5)


class TestSpeed(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.scd = SCDSet.random("ABCDEF")
        cls.restriction = Clade("ABCDE")
        cls.other = SCDSet.random(cls.restriction)
        cls.scd_res = cls.scd.restrict(cls.restriction)
        cls.restricted_subsplit_probs = cls.scd_res.subsplit_probabilities()
        cls.transit = cls.scd.transit_probabilities()
        cls.other_subsplit_probs = cls.other.subsplit_probabilities()
        cls.other_pcss_probs = cls.other.pcss_probabilities()

    def setUp(self) -> None:
        self._started_at = time.time()

    def tearDown(self) -> None:
        elapsed = time.time() - self._started_at
        print(f'{self.id()} ({round(elapsed, 2)}s)')

    def test_brute_kl_gradient_speed_headstart(self):
        brute_kl_gradient(scd=self.scd, other=self.other,
                          scd_res=self.scd_res, transit=self.transit,
                          other_pcss_probs=self.other_pcss_probs)

    def test_brute_kl_gradient_speed(self):
        brute_kl_gradient(scd=self.scd, other=self.other)

    def test_kl_gradient_speed_headstart(self):
        scd_restricted_kl_gradient(scd=self.scd, other=self.other, transit=self.transit,
                                   restricted_self=self.scd_res,
                                   restricted_subsplit_probs=self.restricted_subsplit_probs,
                                   other_subsplit_probs=self.other_subsplit_probs)

    def test_kl_gradient_speed(self):
        scd_restricted_kl_gradient(scd=self.scd, other=self.other)


if __name__ == '__main__':
    unittest.main()
