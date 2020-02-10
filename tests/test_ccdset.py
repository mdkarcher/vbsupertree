import unittest
import sys
import math
import random
sys.path.append("..")
from classes import *


class TestCCDSet(unittest.TestCase):
    def test_log_likelihood(self):
        ccd = CCDSet({Subsplit("AB", "CD"): 0.4, Subsplit("ABC", "D"): 0.6,
                      Subsplit("AB", "C"): 2 / 3, Subsplit("A", "BC"): 1 / 3,
                      Subsplit("A", "B"): 1.0, Subsplit("B", "C"): 1.0,
                      Subsplit("C", "D"): 1.0})
        tree = MyTree("((A,(B,C)),D);")
        self.assertAlmostEqual(ccd.log_likelihood(tree), math.log(0.2))

    def test_likelihood(self):
        ccd = CCDSet({Subsplit("AB", "CD"): 0.4, Subsplit("ABC", "D"): 0.6,
                      Subsplit("AB", "C"): 2 / 3, Subsplit("A", "BC"): 1 / 3,
                      Subsplit("A", "B"): 1.0, Subsplit("B", "C"): 1.0,
                      Subsplit("C", "D"): 1.0})
        tree = MyTree("((A,(B,C)),D);")
        self.assertAlmostEqual(ccd.likelihood(tree), 0.2)

    def test_tree_distribution(self):
        ccd = CCDSet.random_with_sparsity("ABCDE", 0.5)
        tree_dist = ccd.tree_distribution()
        tree, prob = random.choice(list(tree_dist.items()))
        lik = ccd.likelihood(tree)
        self.assertAlmostEqual(prob, lik)

    def test_clade_and_subsplit_probabilities(self):
        ccd = CCDSet.random_with_sparsity("ABCDEF", sparsity=0.5)
        tree_dist = ccd.tree_distribution()

        clade_probs = ccd.clade_probabilities()

        clade = Clade("BCDF")
        self.assertAlmostEqual(tree_dist.feature_prob(clade), clade_probs.get(clade, 0.0))

        subsplit_probs = ccd.unconditional_probabilities()
        subsplit = Subsplit("AD", "B")
        self.assertAlmostEqual(tree_dist.feature_prob(subsplit), subsplit_probs.get(subsplit, 0.0))

    def test_clade_to_clade(self):
        root_clade = Clade("ABCD")
        ccd = CCDSet.random(root_clade)
        tree_dist = ccd.tree_distribution()

        c2c_probs = ccd.clade_to_clade_probabilities()
        c_probs = ccd.clade_probabilities()

        clade = Clade("ABD")
        self.assertAlmostEqual(c_probs[clade], c2c_probs[clade][root_clade])
        self.assertAlmostEqual(c2c_probs[clade][clade], 1.0)

        parent_clade = Clade("ACD")
        child_clade = Clade("CD")
        den = tree_dist.feature_prob(parent_clade)
        num = tree_dist.prob_all([parent_clade, child_clade])
        self.assertAlmostEqual(c2c_probs[child_clade][parent_clade], num / den)

    def test_max_prob_tree(self):
        root_clade = Clade("ABCD")
        ccd = CCDSet.random(root_clade)
        tree_dist = ccd.tree_distribution()

        best_tree = tree_dist.max_item()
        best_lik = tree_dist.max_likelihood()
        ccd_best_tree, ccd_best_lik = ccd.max_prob_tree()
        self.assertEqual(best_tree, ccd_best_tree)
        self.assertAlmostEqual(best_lik, ccd_best_lik)

    def test_max_clade_tree(self):
        ccd = CCDSet(
            {Subsplit("ABC", "D"): 0.40, Subsplit("A", "BCD"): 0.60, Subsplit("AB", "C"): 1.0, Subsplit("A", "B"): 1.0,
             Subsplit("BC", "D"): 0.33, Subsplit("BD", "C"): 0.33, Subsplit("B", "CD"): 0.34,
             Subsplit("B", "C"): 1.0, Subsplit("B", "D"): 1.0, Subsplit("C", "D"): 1.0})
        ccd.normalize()
        tree_dist = ccd.tree_distribution()
        clade_probs = ccd.clade_probabilities()
        max_clade_cred_tree = max(tree_dist, key=lambda tree: tree.clade_score(clade_probs))
        max_clade_tree, max_clade_score = ccd.max_clade_tree()
        self.assertEqual(max_clade_cred_tree, max_clade_tree)
        self.assertAlmostEqual(max_clade_cred_tree.clade_score(clade_probs), max_clade_score)

    def test_kl_implementations(self):
        ccd = CCDSet({Subsplit("AB", "CD"): 0.4, Subsplit("ABC", "D"): 0.6,
                      Subsplit("AB", "C"): 2 / 3, Subsplit("A", "BC"): 1 / 3,
                      Subsplit("A", "B"): 1.0, Subsplit("B", "C"): 1.0,
                      Subsplit("C", "D"): 1.0})
        dist = ccd.tree_distribution()
        ccd2 = CCDSet({Subsplit("AB", "CD"): 0.5, Subsplit("ABC", "D"): 0.5,
                       Subsplit("AB", "C"): 0.5, Subsplit("A", "BC"): 0.5,
                       Subsplit("A", "B"): 1.0, Subsplit("B", "C"): 1.0,
                       Subsplit("C", "D"): 1.0})
        dist2 = ccd2.tree_distribution()

        self.assertAlmostEqual(dist.kl_divergence(dist2), 0.05411532090976828)
        self.assertAlmostEqual(ccd.kl_divergence(ccd2), 0.05411532090976828)
        self.assertAlmostEqual(dist.kl_divergence(ccd2), 0.05411532090976828)
        self.assertAlmostEqual(ccd.kl_divergence(dist2), 0.05411532090976828)

    def test_kl_random(self):
        taxa = Clade("ABCDE")
        ccd1 = CCDSet.random(taxa)
        ccd2 = CCDSet.random(taxa)
        dist1 = ccd1.tree_distribution()
        dist2 = ccd2.tree_distribution()
        dist_dist = dist1.kl_divergence(dist2)
        ccd_dist = ccd1.kl_divergence(dist2)
        ccd_ccd = ccd1.kl_divergence(ccd2)
        dist_ccd = dist1.kl_divergence(ccd2)
        self.assertAlmostEqual(dist_dist, ccd_dist)
        self.assertAlmostEqual(ccd_dist, ccd_ccd)
        self.assertAlmostEqual(ccd_ccd, dist_ccd)
        self.assertAlmostEqual(dist_ccd, dist_dist)

    def test_restriction_identity(self):
        root_clade = Clade("ABCDEF")
        restriction = Clade("ABCDE")

        ccd = CCDSet.random_with_sparsity(root_clade, sparsity=0.7)
        dist = ccd.tree_distribution()
        dist_res = dist.restrict(restriction)

        res_ccd = ccd.restrict(restriction)
        res_ccd_dist = res_ccd.tree_distribution()

        # self.assertNotAlmostEqual(dist_res.kl_divergence(res_ccd_dist), 0.0)
        # self.assertNotAlmostEqual(res_ccd_dist.kl_divergence(dist_res), 0.0)

        res_ccd_dist_ccd = CCDSet.from_tree_distribution(res_ccd_dist)

        self.assertAlmostEqual(res_ccd.kl_divergence(res_ccd_dist_ccd), 0.0)
        self.assertAlmostEqual(res_ccd_dist_ccd.kl_divergence(res_ccd), 0.0)

        dist_res_ccd = CCDSet.from_tree_distribution(dist_res)

        self.assertAlmostEqual(dist_res_ccd.kl_divergence(res_ccd), 0.0)
        self.assertAlmostEqual(res_ccd.kl_divergence(dist_res_ccd), 0.0)

        dist_res_ccd_dist = dist_res_ccd.tree_distribution()

        self.assertAlmostEqual(res_ccd_dist.kl_divergence(dist_res_ccd_dist), 0.0)
        self.assertAlmostEqual(dist_res_ccd_dist.kl_divergence(res_ccd_dist), 0.0)

    def test_gradient1(self):
        ccd = CCDSet(
            {Subsplit("ABC", "D"): 0.40, Subsplit("A", "BCD"): 0.60, Subsplit("AB", "C"): 1.0, Subsplit("A", "B"): 1.0,
             Subsplit("BC", "D"): 0.33, Subsplit("BD", "C"): 0.33, Subsplit("B", "CD"): 0.34,
             Subsplit("B", "C"): 1.0, Subsplit("B", "D"): 1.0, Subsplit("C", "D"): 1.0})
        ccd.normalize()
        ccd2 = ccd.copy()
        clade = Clade("BCD")
        subsplit = Subsplit("BC", "D")
        delta = 0.0001
        ccd2[clade].add_log(subsplit, delta)
        ccd2.normalize(clade)

        uncond = ccd.unconditional_probabilities()
        uncond2 = ccd2.unconditional_probabilities()

        theo_deriv = uncond[subsplit] * (1.0 - ccd[subsplit])
        est_deriv = (uncond2[subsplit] - uncond[subsplit]) / delta
        self.assertAlmostEqual(theo_deriv, est_deriv, 5)

    def test_gradient2(self):
        ccd = CCDSet(
            {Subsplit("ABC", "D"): 0.40, Subsplit("A", "BCD"): 0.60, Subsplit("AB", "C"): 1.0, Subsplit("A", "B"): 1.0,
             Subsplit("BC", "D"): 0.33, Subsplit("BD", "C"): 0.33, Subsplit("B", "CD"): 0.34,
             Subsplit("B", "C"): 1.0, Subsplit("B", "D"): 1.0, Subsplit("C", "D"): 1.0})
        ccd.normalize()

        ccd2 = ccd.copy()

        subsplit1 = Subsplit("A", "BCD")
        clade1 = subsplit1.clade()
        subsplit2 = Subsplit("B", "CD")
        clade2 = subsplit2.clade()
        delta = 0.0001
        ccd2[clade1].add_log(subsplit1, delta)
        ccd2.normalize(clade1)

        uncond = ccd.unconditional_probabilities()
        c2c = ccd.clade_to_clade_probabilities()
        uncond2 = ccd2.unconditional_probabilities()

        theo_deriv = uncond[subsplit1] * (c2c[clade2][clade2] - c2c[clade2][clade1]) * ccd[subsplit2]
        est_deriv = (uncond2[subsplit2] - uncond[subsplit2]) / delta
        self.assertAlmostEqual(theo_deriv, est_deriv, 5)

    def test_gradient3(self):
        ccd = CCDSet.random("ABCDEF")
        delta = 0.0001
        subsplit1 = Subsplit("A", "BCDE")
        subsplit2 = Subsplit("B", "CD")

        clade1 = subsplit1.clade()
        clade2 = subsplit2.clade()
        subsplit1_ch = subsplit1.compatible_child(subsplit2)

        ccd2 = ccd.copy()
        ccd2[clade1].add_log(subsplit1, delta)
        ccd2.normalize(clade1)

        uncond = ccd.unconditional_probabilities()
        c2c = ccd.clade_to_clade_probabilities()
        uncond2 = ccd2.unconditional_probabilities()

        theo_deriv = uncond[subsplit1] * (c2c[clade2][subsplit1_ch] - c2c[clade2][clade1]) * ccd[subsplit2]
        est_deriv = (uncond2[subsplit2] - uncond[subsplit2]) / delta
        self.assertAlmostEqual(theo_deriv, est_deriv, 7)

    def test_gradient4(self):
        ccd = CCDSet.random("ABCDEF")
        delta = 0.00001

        c2c = ccd.clade_to_clade_probabilities()

        clade2 = random.choice(list(c2c.keys()))
        prob_of = random.choice(list(ccd[clade2].keys()))
        clade1 = random.choice(list(c2c[clade2].keys()))
        wrt = random.choice(list(ccd[clade1].keys()))

        est_deriv = estimate_derivative(prob_of, wrt, ccd, delta=delta)
        theo_deriv = ccd.uncond_prob_derivative(prob_of, wrt, c2c)
        deviation = abs(theo_deriv - est_deriv)/abs(est_deriv)
        self.assertAlmostEqual(deviation, 0.0, 5)

    def test_restricted_uncond_prob_derivative(self):
        X = "ABCDEFG"
        Xbar = "ABCDE"
        prob_of = Subsplit("BC", "D")
        wrt = Subsplit("AE", "BCDF")

        ccd = CCDSet.random(X)
        delta = 0.000001

        c2c = ccd.clade_to_clade_probabilities()

        ccd_r = ccd.restrict(Xbar)
        uncond_r = ccd_r.unconditional_probabilities()

        ccd2 = ccd.copy()
        ccd2[wrt.clade()].add_log(wrt, delta)
        ccd2.normalize()
        ccd2_r = ccd2.restrict(Xbar)
        uncond2_r = ccd2_r.unconditional_probabilities()
        man_est_deriv = (uncond2_r[prob_of] - uncond_r[prob_of]) / delta

        est_deriv = estimate_restricted_derivative(restriction=Xbar, prob_of=prob_of, wrt=wrt, ccd=ccd, delta=delta)
        theo_deriv = ccd.restricted_uncond_prob_derivative(restriction=Xbar, prob_of=prob_of, wrt=wrt, c2c=c2c)
        self.assertAlmostEqual(man_est_deriv, est_deriv, 9)
        self.assertAlmostEqual(est_deriv, theo_deriv, 9)

    def test_restricted_clade_prob_derivative(self):
        X = Clade("ABCDEFG")
        Xbar = Clade("ABCDE")
        prob_of = Clade("BCD")
        wrt = Subsplit("AE", "BCDF")

        ccd = CCDSet.random(X)
        delta = 0.000001

        c2c = ccd.clade_to_clade_probabilities()

        ccd_r = ccd.restrict(Xbar)
        uncond_r = ccd_r.clade_probabilities()

        ccd2 = ccd.copy()
        ccd2[wrt.clade()].add_log(wrt, delta)
        ccd2.normalize()
        ccd2_r = ccd2.restrict(Xbar)
        uncond2_r = ccd2_r.clade_probabilities()
        man_est_deriv = (uncond2_r[prob_of] - uncond_r[prob_of]) / delta

        est_deriv = estimate_restricted_clade_derivative(restriction=Xbar, prob_of=prob_of, wrt=wrt, ccd=ccd,
                                                         delta=delta)
        theo_deriv = ccd.restricted_clade_prob_derivative(restriction=Xbar, prob_of=prob_of, wrt=wrt, c2c=c2c)
        self.assertAlmostEqual(man_est_deriv, est_deriv, 9)
        self.assertAlmostEqual(est_deriv, theo_deriv, 8)

    def test_restricted_cond_prob_derivative(self):
        X = Clade("ABCDEFG")
        Xbar = Clade("ABCDE")
        prob_of = Subsplit("A", "BCD")
        wrt = Subsplit("ABCD", "F")

        ccd = CCDSet.random(X)
        delta = 0.000001

        c2c = ccd.clade_to_clade_probabilities()

        ccd_r = ccd.restrict(Xbar)

        ccd2 = ccd.copy()
        ccd2[wrt.clade()].add_log(wrt, delta)
        ccd2.normalize()
        ccd2_r = ccd2.restrict(Xbar)
        man_est_deriv = (ccd2_r[prob_of] - ccd_r[prob_of]) / delta

        est_deriv = estimate_restricted_conditional_derivative(restriction=Xbar, prob_of=prob_of, wrt=wrt, ccd=ccd,
                                                               delta=delta)
        theo_deriv = ccd.restricted_cond_prob_derivative(restriction=Xbar, prob_of=prob_of, wrt=wrt, c2c=c2c,
                                                         restricted_self=ccd_r)
        self.assertAlmostEqual(man_est_deriv, est_deriv, 9)
        self.assertAlmostEqual(est_deriv, theo_deriv, 8)

    def test_restricted_kl_divergence_derivative(self):
        X = Clade("ABCDEFG")
        Xbar = Clade("ABCDE")
        # wrt = Subsplit("A", "BC")
        ccd = CCDSet.random(X)
        ccd_small = CCDSet.random(Xbar)
        delta = 0.000001

        c2c = ccd.clade_to_clade_probabilities()

        ccd_r = ccd.restrict(Xbar)
        ccd_small.kl_divergence(ccd_r)

        # clade1 = random.choice(list(ccd.clades()))
        wrt = random.choice(list(ccd.support()))

        ccd2 = ccd.copy()
        ccd2[wrt.clade()].add_log(wrt, delta)
        ccd2.normalize()
        ccd2_r = ccd2.restrict(Xbar)
        man_est_deriv = (ccd_small.kl_divergence(ccd2_r) - ccd_small.kl_divergence(ccd_r)) / delta

        est_deriv = estimate_kl_divergence_derivative(ccd=ccd, ccd_small=ccd_small, wrt=wrt, delta=delta)
        theo_deriv = ccd.restricted_kl_divergence_derivative(other=ccd_small, wrt=wrt, c2c=c2c, restricted_self=ccd_r)
        self.assertAlmostEqual(man_est_deriv, est_deriv, 9)
        self.assertAlmostEqual(est_deriv, theo_deriv, 8)

    def test_restricted_kl_divergence_gradient(self):
        X = Clade("ABCDEF")
        Xbar = Clade("ABCD")

        ccd = CCDSet.random(X)
        ccd_small = CCDSet.random(Xbar)
        delta = 0.000001

        c2c = ccd.clade_to_clade_probabilities()

        ccd_r = ccd.restrict(Xbar)
        ccd_small.kl_divergence(ccd_r)

        theo_grad = ccd.restricted_kl_divergence_gradient(other=ccd_small, c2c=c2c, restricted_self=ccd_r)

        wrt = random.choice(list(theo_grad.keys()))
        est_deriv = estimate_kl_divergence_derivative(ccd=ccd, ccd_small=ccd_small, wrt=wrt, delta=delta)
        self.assertAlmostEqual(est_deriv, theo_grad[wrt], 8)


if __name__ == '__main__':
    unittest.main()
