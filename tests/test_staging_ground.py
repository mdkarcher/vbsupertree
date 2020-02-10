import unittest
import sys
import math
sys.path.append("..")
from classes import *


class MyTestCase(unittest.TestCase):
    def test_equality(self):
        self.assertEqual(True, True)


class TestProbabilityDistribution(unittest.TestCase):
    def test_normalization(self):
        foo = ProbabilityDistribution()
        foo.set_lin('bar', 0.2)
        self.assertAlmostEqual(foo.params['bar'], math.log(0.2))
        foo.set_lin('baz', 0.3)
        self.assertAlmostEqual(foo.params['baz'], math.log(0.3))
        foo.normalize()
        self.assertAlmostEqual(foo.params['bar'], -0.916290731874155)
        self.assertAlmostEqual(foo.params['baz'], -0.5108256237659908)
        probs = foo.probs()
        self.assertAlmostEqual(probs['bar'], 0.4)
        self.assertAlmostEqual(probs['baz'], 0.6)

    def test_set_many(self):
        foo = ProbabilityDistribution()
        bar = ProbabilityDistribution.random(set("ABC"))
        foo.set_many(bar)
        self.assertEqual(foo, bar)


class TestSubsplitSupport(unittest.TestCase):
    def test_mutualization(self):
        taxa = Clade("ABCDE")
        big_tree = MyTree.random(taxa)
        restriction1 = Clade("ABCD")
        restriction2 = Clade("ABCE")
        tree1 = big_tree.restrict(restriction1)
        tree2 = big_tree.restrict(restriction2)
        ss1 = SubsplitSupport.from_tree(tree1)
        ss2 = SubsplitSupport.from_tree(tree2)
        mut_ss = ss1.mutualize(ss2)
        big_ss = SubsplitSupport.from_tree(big_tree)
        self.assertTrue(big_ss.to_set().issubset(mut_ss.to_set()))


class TestPCSSSupport(unittest.TestCase):
    def test_mutualization(self):
        taxa = Clade("ABCDE")
        big_tree = MyTree.random(taxa)
        restriction1 = Clade("ABCD")
        restriction2 = Clade("ABCE")
        tree1 = big_tree.restrict(restriction1)
        tree2 = big_tree.restrict(restriction2)
        ss1 = PCSSSupport.from_tree(tree1)
        ss2 = PCSSSupport.from_tree(tree2)
        mut_ss = ss1.mutualize(ss2)
        big_ss = PCSSSupport.from_tree(big_tree)
        self.assertTrue(big_ss.to_set().issubset(mut_ss.to_set()))


class TestPCSSSet(unittest.TestCase):
    def test_log_likelihood(self):
        pcss_conditional_probs = {
            PCSS(Subsplit("ABCD"), Subsplit("AB", "CD")): 0.4,
            PCSS(Subsplit("ABCD"), Subsplit("ABC", "D")): 0.6,
            PCSS(Subsplit("ABC", "D"), Subsplit("AB", "C")): 2 / 3,
            PCSS(Subsplit("ABC", "D"), Subsplit("A", "BC")): 1 / 3,
            PCSS(Subsplit("AB", "CD"), Subsplit("A", "B")): 1.0,
            PCSS(Subsplit("AB", "C"), Subsplit("A", "B")): 1.0,
            PCSS(Subsplit("A", "BC"), Subsplit("B", "C")): 1.0,
            PCSS(Subsplit("AB", "CD"), Subsplit("C", "D")): 1.0,
        }
        scd = SCDSet(pcss_conditional_probs)
        scd.normalize()
        tree = MyTree("((A,(B,C)),D);")
        self.assertAlmostEqual(scd.log_likelihood(tree), math.log(0.2))

    def test_likelihood(self):
        pcss_conditional_probs = {
            PCSS(Subsplit("ABCD"), Subsplit("AB", "CD")): 0.4,
            PCSS(Subsplit("ABCD"), Subsplit("ABC", "D")): 0.6,
            PCSS(Subsplit("ABC", "D"), Subsplit("AB", "C")): 2 / 3,
            PCSS(Subsplit("ABC", "D"), Subsplit("A", "BC")): 1 / 3,
            PCSS(Subsplit("AB", "CD"), Subsplit("A", "B")): 1.0,
            PCSS(Subsplit("AB", "C"), Subsplit("A", "B")): 1.0,
            PCSS(Subsplit("A", "BC"), Subsplit("B", "C")): 1.0,
            PCSS(Subsplit("AB", "CD"), Subsplit("C", "D")): 1.0,
        }
        scd = SCDSet(pcss_conditional_probs)
        scd.normalize()
        tree = MyTree("((A,(B,C)),D);")
        self.assertAlmostEqual(scd.likelihood(tree), 0.2)

    def test_tree_distribution(self):
        scd = SCDSet.random_with_sparsity("ABCDE", 0.5)
        tree_dist = scd.tree_distribution()
        tree, prob = random.choice(list(tree_dist.items()))
        lik = scd.likelihood(tree)
        self.assertAlmostEqual(prob, lik)

    def test_parent_probabilities(self):
        scd = SCDSet.random("ABCDE")
        tree_dist = scd.tree_distribution()
        parent_probs = scd.subsplit_probabilities()
        parent, prob = random.choice(list(parent_probs.items()))
        self.assertAlmostEqual(tree_dist.feature_prob(parent), prob)

    def test_pcss_probabilities(self):
        scd = SCDSet.random("ABCDE")
        tree_dist = scd.tree_distribution()
        pcss_probs = scd.pcss_probabilities()
        pcss, prob = random.choice(list(pcss_probs.items()))
        self.assertAlmostEqual(tree_dist.prob_all([pcss.parent, pcss.child]), prob)

    def test_probabilities(self):
        scd = SCDSet.random("ABCDE")
        tree_dist = scd.tree_distribution()
        parent_probs, pcss_probs = scd.probabilities()
        parent, parent_prob = random.choice(list(parent_probs.items()))
        pcss, pcss_prob = random.choice(list(pcss_probs.items()))
        self.assertAlmostEqual(tree_dist.feature_prob(parent), parent_prob)
        self.assertAlmostEqual(tree_dist.prob_all([pcss.parent, pcss.child]), pcss_prob)

    def test_kl_implementations(self):
        pcss_conditional_probs1 = {
            PCSS(Subsplit("ABCD"), Subsplit("AB", "CD")): 0.4,
            PCSS(Subsplit("ABCD"), Subsplit("ABC", "D")): 0.6,
            PCSS(Subsplit("ABC", "D"), Subsplit("AB", "C")): 2 / 3,
            PCSS(Subsplit("ABC", "D"), Subsplit("A", "BC")): 1 / 3,
            PCSS(Subsplit("AB", "CD"), Subsplit("A", "B")): 1.0,
            PCSS(Subsplit("AB", "C"), Subsplit("A", "B")): 1.0,
            PCSS(Subsplit("A", "BC"), Subsplit("B", "C")): 1.0,
            PCSS(Subsplit("AB", "CD"), Subsplit("C", "D")): 1.0,
        }
        pcss_conditional_probs2 = {
            PCSS(Subsplit("ABCD"), Subsplit("AB", "CD")): 0.5,
            PCSS(Subsplit("ABCD"), Subsplit("ABC", "D")): 0.5,
            PCSS(Subsplit("ABC", "D"), Subsplit("AB", "C")): 0.5,
            PCSS(Subsplit("ABC", "D"), Subsplit("A", "BC")): 0.5,
            PCSS(Subsplit("AB", "CD"), Subsplit("A", "B")): 1.0,
            PCSS(Subsplit("AB", "C"), Subsplit("A", "B")): 1.0,
            PCSS(Subsplit("A", "BC"), Subsplit("B", "C")): 1.0,
            PCSS(Subsplit("AB", "CD"), Subsplit("C", "D")): 1.0,
        }
        scd1 = SCDSet(pcss_conditional_probs1)
        scd1.normalize()
        dist1 = scd1.tree_distribution()
        scd2 = SCDSet(pcss_conditional_probs2)
        scd2.normalize()
        dist2 = scd2.tree_distribution()

        self.assertAlmostEqual(dist1.kl_divergence(dist2), 0.05411532090976828)
        self.assertAlmostEqual(scd1.kl_divergence(scd2), 0.05411532090976828)
        self.assertAlmostEqual(dist1.kl_divergence(scd2), 0.05411532090976828)
        self.assertAlmostEqual(scd1.kl_divergence(dist2), 0.05411532090976828)

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

    def test_subsplit_to_subsplit(self):
        root_clade = Clade("ABCDE")
        root_subsplit = Subsplit(root_clade)
        scd = SCDSet.random(root_clade)
        tree_dist = scd.tree_distribution()

        s2s_probs = scd.subsplit_to_subsplit_probabilities()
        s_probs = scd.subsplit_probabilities()

        parent = Subsplit(Clade("ABD"), Clade("C"))
        self.assertAlmostEqual(s_probs[parent], s2s_probs[parent][root_subsplit])
        self.assertAlmostEqual(s2s_probs[parent][parent], 1.0)

        child = Subsplit(Clade("A"), Clade("BD"))
        den = tree_dist.feature_prob(parent)
        num = tree_dist.prob_all([parent, child])
        self.assertAlmostEqual(s2s_probs[child][parent], num / den)

    def test_restriction_identity(self):
        root_clade = Clade("ABCDEF")
        restriction = Clade("ABCD")

        scd = SCDSet.random_with_sparsity(root_clade, sparsity=0.7)
        dist = scd.tree_distribution()
        dist_res = dist.restrict(restriction)

        scd_res = scd.restrict(restriction)
        scd_res_dist = scd_res.tree_distribution()

        scd_res_dist_scd = SCDSet.from_tree_distribution(scd_res_dist)

        self.assertAlmostEqual(scd_res.kl_divergence(scd_res_dist_scd), 0.0)
        self.assertAlmostEqual(scd_res_dist_scd.kl_divergence(scd_res), 0.0)

        dist_res_scd = SCDSet.from_tree_distribution(dist_res)

        self.assertAlmostEqual(dist_res_scd.kl_divergence(scd_res), 0.0)
        self.assertAlmostEqual(scd_res.kl_divergence(dist_res_scd), 0.0)

        dist_res_scd_dist = dist_res_scd.tree_distribution()

        self.assertAlmostEqual(scd_res_dist.kl_divergence(dist_res_scd_dist), 0.0)
        self.assertAlmostEqual(dist_res_scd_dist.kl_divergence(scd_res_dist), 0.0)


if __name__ == '__main__':
    unittest.main()
