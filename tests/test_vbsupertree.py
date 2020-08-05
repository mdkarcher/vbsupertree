import unittest
# import math
from vbsupertree import *


class MyTreeTestCase(unittest.TestCase):
    def test_root_clade(self):
        taxon_set = set("ABCDE")
        tree = MyTree.random(taxon_set)
        self.assertEqual(tree.root_clade(), Clade(taxon_set))

    def test_copy(self):
        taxon_set = set("ABCDE")
        tree1 = MyTree.random(taxon_set)
        tree2 = tree1.copy()
        self.assertEqual(tree1, tree2)
        subset = set("ABCD")
        tree2 = tree2.restrict(subset)
        self.assertNotEqual(tree1, tree2)


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


class TestPCSPSupport(unittest.TestCase):
    def test_to_set(self):
        tree = MyTree("((A,(B,C)),D);")
        pcsps = PCSPSupport.from_tree(tree)
        answer = {PCSP(Subsplit(Clade(), Clade("ABCD")), Subsplit(Clade("ABC"), Clade('D'))),
                  PCSP(Subsplit(Clade('A'), Clade("BC")), Subsplit(Clade('B'), Clade('C'))),
                  PCSP(Subsplit(Clade("ABC"), Clade('D')), Subsplit(Clade('A'), Clade("BC")))}
        self.assertEqual(pcsps.to_set(), answer)
        string_set_answer = {':ABCD, ABC:D', 'A:BC, B:C', 'ABC:D, A:BC'}
        self.assertEqual(pcsps.to_string_set(), string_set_answer)
        string_dict_answer = {'/ABCD': {'ABC:D'}, 'D/ABC': {'A:BC'}, 'A/BC': {'B:C'}}
        self.assertEqual(pcsps.to_string_dict(), string_dict_answer)

    def test_taxon_set(self):
        taxa = Clade("ABCDE")
        tree = MyTree.random(taxa)
        pcsps = PCSPSupport.from_tree(tree)
        self.assertEqual(pcsps.get_taxon_set(), taxa)

    def test_is_complete(self):
        complete = {PCSP(Subsplit(Clade(), Clade("ABCD")), Subsplit(Clade("ABC"), Clade('D'))),
                    PCSP(Subsplit(Clade('A'), Clade("BC")), Subsplit(Clade('B'), Clade('C'))),
                    PCSP(Subsplit(Clade("ABC"), Clade('D')), Subsplit(Clade('A'), Clade("BC")))}
        incomplete = {PCSP(Subsplit(Clade(), Clade("ABCD")), Subsplit(Clade("ABC"), Clade('D'))),
                      PCSP(Subsplit(Clade('A'), Clade("BC")), Subsplit(Clade('B'), Clade('C')))}
        pcsps = PCSPSupport(complete)
        self.assertTrue(pcsps.is_complete())
        pcsps = PCSPSupport(incomplete)
        self.assertFalse(pcsps.is_complete())

    def test_random_tree(self):
        answer = {PCSP(Subsplit(Clade(), Clade("ABCD")), Subsplit(Clade("ABC"), Clade('D'))),
                  PCSP(Subsplit(Clade('A'), Clade("BC")), Subsplit(Clade('B'), Clade('C'))),
                  PCSP(Subsplit(Clade("ABC"), Clade('D')), Subsplit(Clade('A'), Clade("BC")))}
        pcsps = PCSPSupport(answer)
        tree = pcsps.random_tree()
        pcsps2 = PCSPSupport.from_tree(tree)
        self.assertTrue(pcsps2.to_set().issubset(answer))

    def test_mutualization(self):
        taxa = Clade("ABCDE")
        big_tree = MyTree.random(taxa)
        restriction1 = Clade("ABCD")
        restriction2 = Clade("ABCE")
        tree1 = big_tree.restrict(restriction1)
        tree2 = big_tree.restrict(restriction2)
        ss1 = PCSPSupport.from_tree(tree1)
        ss2 = PCSPSupport.from_tree(tree2)
        mut_ss = ss1.mutualize(ss2)
        big_ss = PCSPSupport.from_tree(big_tree)
        self.assertTrue(big_ss.to_set().issubset(mut_ss.to_set()))


class TestSBN(unittest.TestCase):
    def test_likelihoods(self):
        pcsp_conditional_probs = {
            PCSP(Subsplit("ABCD"), Subsplit("AB", "CD")): 0.4,
            PCSP(Subsplit("ABCD"), Subsplit("ABC", "D")): 0.6,
            PCSP(Subsplit("ABC", "D"), Subsplit("AB", "C")): 2 / 3,
            PCSP(Subsplit("ABC", "D"), Subsplit("A", "BC")): 1 / 3,
            PCSP(Subsplit("AB", "CD"), Subsplit("A", "B")): 1.0,
            PCSP(Subsplit("AB", "C"), Subsplit("A", "B")): 1.0,
            PCSP(Subsplit("A", "BC"), Subsplit("B", "C")): 1.0,
            PCSP(Subsplit("AB", "CD"), Subsplit("C", "D")): 1.0,
        }
        scd = SBN(pcsp_conditional_probs)
        scd.normalize()
        tree = MyTree("((A,(B,C)),D);")
        self.assertAlmostEqual(scd.likelihood(tree), 0.2)
        self.assertAlmostEqual(scd.log_likelihood(tree), math.log(0.2))

    def test_random_scd(self):
        scd = SBN.random("ABCDE")
        self.assertEqual(len(scd), 210)
        tree_dist = scd.tree_distribution()
        self.assertEqual(len(tree_dist), 105)
        tree, prob = random.choice(list(tree_dist.items()))
        lik = scd.likelihood(tree)
        self.assertAlmostEqual(prob, lik)

    def test_tree_distribution(self):
        scd = SBN.random_with_sparsity("ABCDEF", 0.5)
        tree_dist = scd.tree_distribution()
        tree, prob = random.choice(list(tree_dist.items()))
        lik = scd.likelihood(tree)
        self.assertAlmostEqual(prob, lik)

    def test_subsplit_probabilities(self):
        scd = SBN.random("ABCDE")
        tree_dist = scd.tree_distribution()
        subsplit_probs = scd.subsplit_probabilities()
        subsplit, prob = random.choice(list(subsplit_probs.items()))
        self.assertAlmostEqual(tree_dist.feature_prob(subsplit), prob)

    def test_clade_and_subsplit_probabilities(self):
        scd = SBN.random("ABCDE")
        tree_dist = scd.tree_distribution()

        clade_probs = scd.clade_probabilities()

        clade = random.choice(list(clade_probs.keys()))
        self.assertAlmostEqual(tree_dist.feature_prob(clade), clade_probs.get(clade, 0.0))

    def test_pcsp_probabilities(self):
        scd = SBN.random("ABCDE")
        tree_dist = scd.tree_distribution()
        pcsp_probs = scd.pcsp_probabilities()
        pcsp, prob = random.choice(list(pcsp_probs.items()))
        self.assertAlmostEqual(tree_dist.prob_all([pcsp.parent, pcsp.child]), prob)

    def test_probabilities(self):
        scd = SBN.random("ABCDE")
        tree_dist = scd.tree_distribution()
        subsplit_probs, pcsp_probs = scd.probabilities()
        subsplit, subsplit_prob = random.choice(list(subsplit_probs.items()))
        pcsp, pcsp_prob = random.choice(list(pcsp_probs.items()))
        self.assertAlmostEqual(tree_dist.feature_prob(subsplit), subsplit_prob)
        self.assertAlmostEqual(tree_dist.prob_all([pcsp.parent, pcsp.child]), pcsp_prob)

    def test_transit_probabilities(self):
        root_clade = Clade("ABCDE")
        root_subsplit = Subsplit(root_clade)
        scd = SBN.random(root_clade)
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
        pcsp_conditional_probs1 = {
            PCSP(Subsplit("ABCD"), Subsplit("AB", "CD")): 0.4,
            PCSP(Subsplit("ABCD"), Subsplit("ABC", "D")): 0.6,
            PCSP(Subsplit("ABC", "D"), Subsplit("AB", "C")): 2 / 3,
            PCSP(Subsplit("ABC", "D"), Subsplit("A", "BC")): 1 / 3,
            PCSP(Subsplit("AB", "CD"), Subsplit("A", "B")): 1.0,
            PCSP(Subsplit("AB", "C"), Subsplit("A", "B")): 1.0,
            PCSP(Subsplit("A", "BC"), Subsplit("B", "C")): 1.0,
            PCSP(Subsplit("AB", "CD"), Subsplit("C", "D")): 1.0,
        }
        pcsp_conditional_probs2 = {
            PCSP(Subsplit("ABCD"), Subsplit("AB", "CD")): 0.5,
            PCSP(Subsplit("ABCD"), Subsplit("ABC", "D")): 0.5,
            PCSP(Subsplit("ABC", "D"), Subsplit("AB", "C")): 0.5,
            PCSP(Subsplit("ABC", "D"), Subsplit("A", "BC")): 0.5,
            PCSP(Subsplit("AB", "CD"), Subsplit("A", "B")): 1.0,
            PCSP(Subsplit("AB", "C"), Subsplit("A", "B")): 1.0,
            PCSP(Subsplit("A", "BC"), Subsplit("B", "C")): 1.0,
            PCSP(Subsplit("AB", "CD"), Subsplit("C", "D")): 1.0,
        }
        scd1 = SBN(pcsp_conditional_probs1)
        scd1.normalize()
        dist1 = scd1.tree_distribution()
        scd2 = SBN(pcsp_conditional_probs2)
        scd2.normalize()
        dist2 = scd2.tree_distribution()

        self.assertAlmostEqual(dist1.kl_divergence(dist2), 0.05411532090976828)
        self.assertAlmostEqual(scd1.kl_divergence(scd2), 0.05411532090976828)
        self.assertAlmostEqual(dist1.kl_divergence(scd2), 0.05411532090976828)
        self.assertAlmostEqual(scd1.kl_divergence(dist2), 0.05411532090976828)

    def test_kl_random(self):
        taxa = Clade("ABCDE")
        scd1 = SBN.random(taxa)
        scd2 = SBN.random(taxa)
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

        scd = SBN.random_with_sparsity(root_clade, sparsity=0.5)
        dist = scd.tree_distribution()
        dist_res = dist.restrict(restriction)

        scd_res = scd.restrict(restriction)
        scd_res_dist = scd_res.tree_distribution()

        scd_res_dist_scd = SBN.from_tree_distribution(scd_res_dist)

        self.assertAlmostEqual(scd_res.kl_divergence(scd_res_dist_scd), 0.0)
        self.assertAlmostEqual(scd_res_dist_scd.kl_divergence(scd_res), 0.0)

        dist_res_scd = SBN.from_tree_distribution(dist_res)

        self.assertAlmostEqual(dist_res_scd.kl_divergence(scd_res), 0.0)
        self.assertAlmostEqual(scd_res.kl_divergence(dist_res_scd), 0.0)

        dist_res_scd_dist = dist_res_scd.tree_distribution()

        self.assertAlmostEqual(scd_res_dist.kl_divergence(dist_res_scd_dist), 0.0)
        self.assertAlmostEqual(dist_res_scd_dist.kl_divergence(scd_res_dist), 0.0)

    def test_restriction_subsplit_probs(self):
        root_clade = Clade("ABCDE")
        restriction = Clade("ABCD")

        scd = SBN.random(root_clade)
        dist = scd.tree_distribution()
        dist_res = dist.restrict(restriction)

        scd_res = scd.restrict(restriction)

        scd_res_ssprobs = scd_res.subsplit_probabilities()

        for ss, prob in scd_res_ssprobs.items():
            self.assertAlmostEqual(prob, dist_res.feature_prob(ss))

    def test_restriction_pcsp_probs(self):
        root_clade = Clade("ABCDE")
        restriction = Clade("ABCD")

        scd = SBN.random(root_clade)
        dist = scd.tree_distribution()
        dist_res = dist.restrict(restriction)

        scd_res = scd.restrict(restriction)

        scd_res_pcspprobs = scd_res.pcsp_probabilities()

        for pcsp, prob in scd_res_pcspprobs.items():
            self.assertAlmostEqual(prob, dist_res.feature_prob(pcsp))

    def test_max_prob_tree(self):
        root_clade = Clade("ABCDE")
        scd = SBN.random(root_clade)
        tree_dist = scd.tree_distribution()

        best_tree = tree_dist.max_item()
        best_lik = tree_dist.max_likelihood()
        scd_best_tree, scd_best_lik = scd.max_prob_tree()
        self.assertEqual(best_tree, scd_best_tree)
        self.assertAlmostEqual(best_lik, scd_best_lik)

    def test_restriction_conditional_subsplit_probs(self):
        root_clade = Clade("ABCDE")
        restriction = Clade("ABCD")

        scd = SBN.random(root_clade)
        dist = scd.tree_distribution()
        dist_res = dist.restrict(restriction)

        scd_res = scd.restrict(restriction)

        for pcsp in scd_res.iter_pcsp():
            cond_prob = scd_res.get(pcsp)
            joint_prob = dist_res.feature_prob(pcsp)
            parent_prob = dist_res.feature_prob(pcsp.parent)
            self.assertAlmostEqual(cond_prob, joint_prob/parent_prob)

    def test_max_clade_tree_fixed(self):
        pcsp_conditional_probs = {
            PCSP(Subsplit("ABCD"), Subsplit("AB", "CD")): 0.4,
            PCSP(Subsplit("ABCD"), Subsplit("ABC", "D")): 0.6,
            PCSP(Subsplit("ABC", "D"), Subsplit("AB", "C")): 2 / 3,
            PCSP(Subsplit("ABC", "D"), Subsplit("A", "BC")): 1 / 3,
            PCSP(Subsplit("AB", "CD"), Subsplit("A", "B")): 1.0,
            PCSP(Subsplit("AB", "C"), Subsplit("A", "B")): 1.0,
            PCSP(Subsplit("A", "BC"), Subsplit("B", "C")): 1.0,
            PCSP(Subsplit("AB", "CD"), Subsplit("C", "D")): 1.0,
        }
        scd = SBN(pcsp_conditional_probs)
        scd.normalize()
        tree_dist = scd.tree_distribution()
        clade_probs = scd.clade_probabilities()
        max_clade_cred_tree = max(tree_dist, key=lambda tree: tree.clade_score(clade_probs))
        max_clade_tree, max_clade_score = scd.max_clade_tree()
        self.assertEqual(max_clade_cred_tree, max_clade_tree)
        self.assertAlmostEqual(max_clade_cred_tree.clade_score(clade_probs), max_clade_score)

    def test_max_clade_tree(self):
        scd = SBN.random("ABCDE")
        tree_dist = scd.tree_distribution()
        clade_probs = scd.clade_probabilities()
        max_clade_cred_tree = max(tree_dist, key=lambda tree: tree.clade_score(clade_probs))
        max_clade_tree, max_clade_score = scd.max_clade_tree()
        self.assertEqual(max_clade_cred_tree, max_clade_tree)
        self.assertAlmostEqual(max_clade_cred_tree.clade_score(clade_probs), max_clade_score)

    def test_restricted_kl_gradient_multi(self):
        pcsp_conditional_probs_big = {
            PCSP(Subsplit("ABCDE"), Subsplit("ABCD", "E")): 0.7,
            PCSP(Subsplit("ABCDE"), Subsplit("ABC", "DE")): 0.3,
            PCSP(Subsplit("ABCD", "E"), Subsplit("AB", "CD")): 0.4,
            PCSP(Subsplit("ABCD", "E"), Subsplit("ABC", "D")): 0.6,
            PCSP(Subsplit("ABC", "DE"), Subsplit("AB", "C")): 0.3,
            PCSP(Subsplit("ABC", "DE"), Subsplit("A", "BC")): 0.7,
            PCSP(Subsplit("ABC", "DE"), Subsplit("D", "E")): 1.0,
            PCSP(Subsplit("ABC", "D"), Subsplit("AB", "C")): 0.8,
            PCSP(Subsplit("ABC", "D"), Subsplit("A", "BC")): 0.2,
            PCSP(Subsplit("AB", "CD"), Subsplit("A", "B")): 1.0,
            PCSP(Subsplit("AB", "C"), Subsplit("A", "B")): 1.0,
            PCSP(Subsplit("A", "BC"), Subsplit("B", "C")): 1.0,
            PCSP(Subsplit("AB", "CD"), Subsplit("C", "D")): 1.0,
        }
        pcsp_conditional_probs1 = {
            PCSP(Subsplit("ABCD"), Subsplit("AB", "CD")): 0.4,
            PCSP(Subsplit("ABCD"), Subsplit("ABC", "D")): 0.6,
            PCSP(Subsplit("ABC", "D"), Subsplit("AB", "C")): 2 / 3,
            PCSP(Subsplit("ABC", "D"), Subsplit("A", "BC")): 1 / 3,
            PCSP(Subsplit("AB", "CD"), Subsplit("A", "B")): 1.0,
            PCSP(Subsplit("AB", "C"), Subsplit("A", "B")): 1.0,
            PCSP(Subsplit("A", "BC"), Subsplit("B", "C")): 1.0,
            PCSP(Subsplit("AB", "CD"), Subsplit("C", "D")): 1.0,
        }
        pcsp_conditional_probs2 = {
            PCSP(Subsplit("ABCD"), Subsplit("AB", "CD")): 0.5,
            PCSP(Subsplit("ABCD"), Subsplit("ABC", "D")): 0.5,
            PCSP(Subsplit("ABC", "D"), Subsplit("AB", "C")): 0.5,
            PCSP(Subsplit("ABC", "D"), Subsplit("A", "BC")): 0.5,
            PCSP(Subsplit("AB", "CD"), Subsplit("A", "B")): 1.0,
            PCSP(Subsplit("AB", "C"), Subsplit("A", "B")): 1.0,
            PCSP(Subsplit("A", "BC"), Subsplit("B", "C")): 1.0,
            PCSP(Subsplit("AB", "CD"), Subsplit("C", "D")): 1.0,
        }
        answer = {
            PCSP(Subsplit(Clade(), Clade({'A', 'B', 'C', 'D', 'E'})),
                 Subsplit(Clade({'A', 'B', 'C', 'D'}), Clade({'E'}))): -0.14124748490945693,
            PCSP(Subsplit(Clade(), Clade({'A', 'B', 'C', 'D', 'E'})),
                 Subsplit(Clade({'A', 'B', 'C'}), Clade({'D', 'E'}))): 0.14124748490945677,
            PCSP(Subsplit(Clade({'A', 'B', 'C', 'D'}), Clade({'E'})),
                 Subsplit(Clade({'A', 'B'}), Clade({'C', 'D'}))): -0.2835010060362173,
            PCSP(Subsplit(Clade({'A', 'B', 'C', 'D'}), Clade({'E'})),
                 Subsplit(Clade({'A', 'B', 'C'}), Clade({'D'}))): 0.28350100603621725,
            PCSP(Subsplit(Clade({'A', 'B', 'C'}), Clade({'D', 'E'})),
                 Subsplit(Clade({'A', 'B'}), Clade({'C'}))): 0.0003018108651911544,
            PCSP(Subsplit(Clade({'A', 'B', 'C'}), Clade({'D', 'E'})),
                 Subsplit(Clade({'A'}), Clade({'B', 'C'}))): -0.0003018108651911683,
            PCSP(Subsplit(Clade({'A', 'B', 'C'}), Clade({'D', 'E'})),
                 Subsplit(Clade({'D'}), Clade({'E'}))): 0.0,
            PCSP(Subsplit(Clade({'A', 'B', 'C'}), Clade({'D'})),
                 Subsplit(Clade({'A', 'B'}), Clade({'C'}))): 0.0003219315895372707,
            PCSP(Subsplit(Clade({'A', 'B', 'C'}), Clade({'D'})),
                 Subsplit(Clade({'A'}), Clade({'B', 'C'}))): -0.00032193158953724293,
            PCSP(Subsplit(Clade({'A', 'B'}), Clade({'C', 'D'})),
                 Subsplit(Clade({'A'}), Clade({'B'}))): 0.0,
            PCSP(Subsplit(Clade({'A', 'B'}), Clade({'C'})),
                 Subsplit(Clade({'A'}), Clade({'B'}))): 0.0,
            PCSP(Subsplit(Clade({'A'}), Clade({'B', 'C'})),
                 Subsplit(Clade({'B'}), Clade({'C'}))): 0.0,
            PCSP(Subsplit(Clade({'A', 'B'}), Clade({'C', 'D'})),
                 Subsplit(Clade({'C'}), Clade({'D'}))): 0.0
        }

        scd_big = SBN(pcsp_conditional_probs_big)
        scd_ref1 = SBN(pcsp_conditional_probs1)
        scd_ref2 = SBN(pcsp_conditional_probs2)
        result = scd_big.restricted_kl_gradient_multi([scd_ref1, scd_ref2])
        for parent_clade in result:
            self.assertAlmostEqual(result[parent_clade], answer[parent_clade])


if __name__ == '__main__':
    unittest.main()
