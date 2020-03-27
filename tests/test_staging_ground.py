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


class TestPCSSSupport_old(unittest.TestCase):
    def test_mutualization(self):
        taxa = Clade("ABCDE")
        big_tree = MyTree.random(taxa)
        restriction1 = Clade("ABCD")
        restriction2 = Clade("ABCE")
        tree1 = big_tree.restrict(restriction1)
        tree2 = big_tree.restrict(restriction2)
        ss1 = PCSSSupport_old.from_tree(tree1)
        ss2 = PCSSSupport_old.from_tree(tree2)
        mut_ss = ss1.mutualize(ss2)
        big_ss = PCSSSupport_old.from_tree(big_tree)
        self.assertTrue(big_ss.to_set().issubset(mut_ss.to_set()))


class TestPCSSSupport(unittest.TestCase):
    def test_to_set(self):
        tree = MyTree("((A,(B,C)),D);")
        pcsss = PCSSSupport.from_tree(tree)
        answer = {PCSS(Subsplit(Clade(), Clade("ABCD")), Subsplit(Clade("ABC"), Clade('D'))),
                  PCSS(Subsplit(Clade('A'), Clade("BC")), Subsplit(Clade('B'), Clade('C'))),
                  PCSS(Subsplit(Clade("ABC"), Clade('D')), Subsplit(Clade('A'), Clade("BC")))}
        self.assertEqual(pcsss.to_set(), answer)
        string_set_answer = {':ABCD, ABC:D', 'A:BC, B:C', 'ABC:D, A:BC'}
        self.assertEqual(pcsss.to_string_set(), string_set_answer)
        string_dict_answer = {'/ABCD': {'ABC:D'}, 'D/ABC': {'A:BC'}, 'A/BC': {'B:C'}}
        self.assertEqual(pcsss.to_string_dict(), string_dict_answer)

    def test_taxon_set(self):
        taxa = Clade("ABCDE")
        tree = MyTree.random(taxa)
        pcsss = PCSSSupport.from_tree(tree)
        self.assertEqual(pcsss.get_taxon_set(), taxa)

    def test_is_complete(self):
        complete = {PCSS(Subsplit(Clade(), Clade("ABCD")), Subsplit(Clade("ABC"), Clade('D'))),
                    PCSS(Subsplit(Clade('A'), Clade("BC")), Subsplit(Clade('B'), Clade('C'))),
                    PCSS(Subsplit(Clade("ABC"), Clade('D')), Subsplit(Clade('A'), Clade("BC")))}
        incomplete = {PCSS(Subsplit(Clade(), Clade("ABCD")), Subsplit(Clade("ABC"), Clade('D'))),
                  PCSS(Subsplit(Clade('A'), Clade("BC")), Subsplit(Clade('B'), Clade('C')))}
        pcsss = PCSSSupport(complete)
        self.assertTrue(pcsss.is_complete())
        pcsss = PCSSSupport(incomplete)
        self.assertFalse(pcsss.is_complete())

    def test_random_tree(self):
        answer = {PCSS(Subsplit(Clade(), Clade("ABCD")), Subsplit(Clade("ABC"), Clade('D'))),
                  PCSS(Subsplit(Clade('A'), Clade("BC")), Subsplit(Clade('B'), Clade('C'))),
                  PCSS(Subsplit(Clade("ABC"), Clade('D')), Subsplit(Clade('A'), Clade("BC")))}
        pcsss = PCSSSupport(answer)
        tree = pcsss.random_tree()
        pcsss2 = PCSSSupport.from_tree(tree)
        self.assertTrue(pcsss2.to_set().issubset(answer))

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


class TestSCDSet_old(unittest.TestCase):
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
        scd = SCDSet_old(pcss_conditional_probs)
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
        scd = SCDSet_old(pcss_conditional_probs)
        scd.normalize()
        tree = MyTree("((A,(B,C)),D);")
        self.assertAlmostEqual(scd.likelihood(tree), 0.2)

    def test_tree_distribution(self):
        scd = SCDSet_old.random_with_sparsity("ABCDE", 0.5)
        tree_dist = scd.tree_distribution()
        tree, prob = random.choice(list(tree_dist.items()))
        lik = scd.likelihood(tree)
        self.assertAlmostEqual(prob, lik)

    def test_parent_probabilities(self):
        scd = SCDSet_old.random("ABCDE")
        tree_dist = scd.tree_distribution()
        parent_probs = scd.subsplit_probabilities()
        parent, prob = random.choice(list(parent_probs.items()))
        self.assertAlmostEqual(tree_dist.feature_prob(parent), prob)

    def test_pcss_probabilities(self):
        scd = SCDSet_old.random("ABCDE")
        tree_dist = scd.tree_distribution()
        pcss_probs = scd.pcss_probabilities()
        pcss, prob = random.choice(list(pcss_probs.items()))
        self.assertAlmostEqual(tree_dist.prob_all([pcss.parent, pcss.child]), prob)

    def test_probabilities(self):
        scd = SCDSet_old.random("ABCDE")
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
        scd1 = SCDSet_old(pcss_conditional_probs1)
        scd1.normalize()
        dist1 = scd1.tree_distribution()
        scd2 = SCDSet_old(pcss_conditional_probs2)
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
        scd = SCDSet_old.random(root_clade)
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

        scd = SCDSet_old.random_with_sparsity(root_clade, sparsity=0.7)
        dist = scd.tree_distribution()
        dist_res = dist.restrict(restriction)

        scd_res = scd.restrict(restriction)
        scd_res_dist = scd_res.tree_distribution()

        scd_res_dist_scd = SCDSet_old.from_tree_distribution(scd_res_dist)

        self.assertAlmostEqual(scd_res.kl_divergence(scd_res_dist_scd), 0.0)
        self.assertAlmostEqual(scd_res_dist_scd.kl_divergence(scd_res), 0.0)

        dist_res_scd = SCDSet_old.from_tree_distribution(dist_res)

        self.assertAlmostEqual(dist_res_scd.kl_divergence(scd_res), 0.0)
        self.assertAlmostEqual(scd_res.kl_divergence(dist_res_scd), 0.0)

        dist_res_scd_dist = dist_res_scd.tree_distribution()

        self.assertAlmostEqual(scd_res_dist.kl_divergence(dist_res_scd_dist), 0.0)
        self.assertAlmostEqual(dist_res_scd_dist.kl_divergence(scd_res_dist), 0.0)


class TestSCDSSet(unittest.TestCase):
    def test_likelihoods(self):
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
        self.assertAlmostEqual(scd.log_likelihood(tree), math.log(0.2))

    def test_random_scd(self):
        scd = SCDSet.random("ABCDE")
        self.assertEqual(len(scd), 210)
        tree_dist = scd.tree_distribution()
        self.assertEqual(len(tree_dist), 105)
        tree, prob = random.choice(list(tree_dist.items()))
        lik = scd.likelihood(tree)
        self.assertAlmostEqual(prob, lik)

    def test_tree_distribution(self):
        scd = SCDSet.random_with_sparsity("ABCDEF", 0.5)
        tree_dist = scd.tree_distribution()
        tree, prob = random.choice(list(tree_dist.items()))
        lik = scd.likelihood(tree)
        self.assertAlmostEqual(prob, lik)

    def test_subsplit_probabilities(self):
        scd = SCDSet.random("ABCDE")
        tree_dist = scd.tree_distribution()
        subsplit_probs = scd.subsplit_probabilities()
        subsplit, prob = random.choice(list(subsplit_probs.items()))
        self.assertAlmostEqual(tree_dist.feature_prob(subsplit), prob)

    def test_clade_and_subsplit_probabilities(self):
        scd = SCDSet.random("ABCDE")
        tree_dist = scd.tree_distribution()

        clade_probs = scd.clade_probabilities()

        clade = random.choice(list(clade_probs.keys()))
        self.assertAlmostEqual(tree_dist.feature_prob(clade), clade_probs.get(clade, 0.0))

    def test_pcss_probabilities(self):
        scd = SCDSet.random("ABCDE")
        tree_dist = scd.tree_distribution()
        pcss_probs = scd.pcss_probabilities()
        pcss, prob = random.choice(list(pcss_probs.items()))
        self.assertAlmostEqual(tree_dist.prob_all([pcss.parent, pcss.child]), prob)

    def test_probabilities(self):
        scd = SCDSet.random("ABCDE")
        tree_dist = scd.tree_distribution()
        subsplit_probs, pcss_probs = scd.probabilities()
        subsplit, subsplit_prob = random.choice(list(subsplit_probs.items()))
        pcss, pcss_prob = random.choice(list(pcss_probs.items()))
        self.assertAlmostEqual(tree_dist.feature_prob(subsplit), subsplit_prob)
        self.assertAlmostEqual(tree_dist.prob_all([pcss.parent, pcss.child]), pcss_prob)

    def test_transit_probabilities(self):
        root_clade = Clade("ABCDE")
        root_subsplit = Subsplit(root_clade)
        scd = SCDSet.random(root_clade)
        tree_dist = scd.tree_distribution()

        t_probs = scd.transit_probabilities()
        s_probs = scd.subsplit_probabilities()

        parent = Subsplit(Clade("ABD"), Clade("C"))
        self.assertAlmostEqual(s_probs[parent], t_probs[parent][root_subsplit])
        self.assertAlmostEqual(t_probs[parent][parent], 1.0)

        child = Subsplit(Clade("A"), Clade("BD"))
        den = tree_dist.feature_prob(parent)
        num = tree_dist.prob_all([parent, child])
        self.assertAlmostEqual(t_probs[child][parent], num / den)

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
        scd1 = SCDSet.random(taxa)
        scd2 = SCDSet.random(taxa)
        dist1 = scd1.tree_distribution()
        dist2 = scd2.tree_distribution()
        dist_dist = dist1.kl_divergence(dist2)
        scd_dist = scd1.kl_divergence(dist2)
        scd_scd = scd1.kl_divergence(scd2)
        dist_scd = dist1.kl_divergence(scd2)
        self.assertAlmostEqual(dist_dist, scd_dist)
        self.assertAlmostEqual(scd_dist, scd_scd)
        self.assertAlmostEqual(scd_scd, dist_scd)
        self.assertAlmostEqual(dist_scd, dist_dist)

    def test_restriction_identity(self):
        root_clade = Clade("ABCDEF")
        restriction = Clade("ABCDE")

        scd = SCDSet.random_with_sparsity(root_clade, sparsity=0.5)
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

    def test_restriction_subsplit_probs(self):
        root_clade = Clade("ABCDE")
        restriction = Clade("ABCD")

        scd = SCDSet.random(root_clade)
        dist = scd.tree_distribution()
        dist_res = dist.restrict(restriction)

        scd_res = scd.restrict(restriction)

        scd_res_ssprobs = scd_res.subsplit_probabilities()

        for ss, prob in scd_res_ssprobs.items():
            self.assertAlmostEqual(prob, dist_res.feature_prob(ss))

    def test_restriction_pcss_probs(self):
        root_clade = Clade("ABCDE")
        restriction = Clade("ABCD")

        scd = SCDSet.random(root_clade)
        dist = scd.tree_distribution()
        dist_res = dist.restrict(restriction)

        scd_res = scd.restrict(restriction)

        scd_res_pcssprobs = scd_res.pcss_probabilities()

        for pcss, prob in scd_res_pcssprobs.items():
            self.assertAlmostEqual(prob, dist_res.feature_prob(pcss))

    def test_max_prob_tree(self):
        root_clade = Clade("ABCDE")
        scd = SCDSet.random(root_clade)
        tree_dist = scd.tree_distribution()

        best_tree = tree_dist.max_item()
        best_lik = tree_dist.max_likelihood()
        scd_best_tree, scd_best_lik = scd.max_prob_tree()
        self.assertEqual(best_tree, scd_best_tree)
        self.assertAlmostEqual(best_lik, scd_best_lik)

    def test_restriction_conditional_subsplit_probs(self):
        root_clade = Clade("ABCDE")
        restriction = Clade("ABCD")

        scd = SCDSet.random(root_clade)
        dist = scd.tree_distribution()
        dist_res = dist.restrict(restriction)

        scd_res = scd.restrict(restriction)

        for pcss in scd_res.iter_pcss():
            cond_prob = scd_res.get(pcss)
            joint_prob = dist_res.feature_prob(pcss)
            parent_prob = dist_res.feature_prob(pcss.parent)
            self.assertAlmostEqual(cond_prob, joint_prob/parent_prob)

    def test_max_clade_tree_fixed(self):
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
        tree_dist = scd.tree_distribution()
        clade_probs = scd.clade_probabilities()
        max_clade_cred_tree = max(tree_dist, key=lambda tree: tree.clade_score(clade_probs))
        max_clade_tree, max_clade_score = scd.max_clade_tree()
        self.assertEqual(max_clade_cred_tree, max_clade_tree)
        self.assertAlmostEqual(max_clade_cred_tree.clade_score(clade_probs), max_clade_score)

    def test_max_clade_tree(self):
        scd = SCDSet.random("ABCDE")
        tree_dist = scd.tree_distribution()
        clade_probs = scd.clade_probabilities()
        max_clade_cred_tree = max(tree_dist, key=lambda tree: tree.clade_score(clade_probs))
        max_clade_tree, max_clade_score = scd.max_clade_tree()
        self.assertEqual(max_clade_cred_tree, max_clade_tree)
        self.assertAlmostEqual(max_clade_cred_tree.clade_score(clade_probs), max_clade_score)




if __name__ == '__main__':
    unittest.main()
