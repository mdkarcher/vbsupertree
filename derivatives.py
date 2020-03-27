from classes import *


def scd_subsplit_derivative(prob_of: Subsplit, wrt: PCSS, scd: SCDSet, transit: dict=None):
    root_subsplit = scd.root_subsplit()
    wrt_parent = wrt.parent
    wrt_parent_clade = wrt.parent_clade()
    wrt_child = wrt.child
    # parent2 = prob_of.clade()
    # if not prob_of.clade().issubset(wrt_child.clade()):
    #     return 0.0
    if transit is None:
        transit = scd.transit_probabilities()
    uncond_wrt = transit.get(wrt_parent, dict()).get(root_subsplit, 0.0) * scd[wrt]
    child_to_prob_of = transit.get(prob_of, dict()).get(wrt_child, 0.0)
    parent_to_prob_of = transit.get(prob_of, dict()).get(wrt_parent_clade, 0.0)
    return uncond_wrt * (child_to_prob_of - parent_to_prob_of)


def scd_estimate_subsplit_derivative(prob_of: Subsplit, wrt: PCSS, scd: SCDSet, delta: float=0.0001, uncond = None, uncond2 = None):
    if uncond is None:
        uncond = scd.subsplit_probabilities()
    if uncond2 is None:
        parent = wrt.parent_clade()
        scd2 = scd.copy()
        scd2.add_log(wrt, delta)
        scd2.normalize(parent)
        uncond2 = scd2.subsplit_probabilities()
    est_deriv = (uncond2[prob_of] - uncond[prob_of]) / delta
    return est_deriv


def scd_subsplit_to_subsplit_cond_derivative(prob_of: Subsplit, cond_on: Subsplit, wrt: PCSS, scd: SCDSet, transit: dict=None):
    # root_subsplit = scd.root_subsplit()
    parent = wrt.parent
    parent_clade = wrt.parent_clade()
    child = wrt.child
    # parent2 = prob_of.clade()
    # if not prob_of.valid_ancestor(parent):
    #     return 0.0
    if transit is None:
        transit = scd.transit_probabilities()
    uncond_wrt = transit.get(parent, dict()).get(cond_on, 0.0) * scd[wrt]
    child_to_prob_of = transit.get(prob_of, dict()).get(child, 0.0)
    parent_to_prob_of = transit.get(prob_of, dict()).get(parent_clade, 0.0)
    return uncond_wrt * (child_to_prob_of - parent_to_prob_of)


def scd_estimate_subsplit_to_subsplit_cond_derivative(prob_of: Subsplit, cond_on: Subsplit, wrt: PCSS, scd: SCDSet, delta: float=0.0001, transit = None, transit2 = None):
    if transit is None:
        transit = scd.transit_probabilities()
    if transit2 is None:
        parent = wrt.parent_clade()
        scd2 = scd.copy()
        scd2.add_log(wrt, delta)
        scd2.normalize(parent)
        transit2 = scd2.transit_probabilities()
    est_deriv = (transit2.get(prob_of, dict()).get(cond_on, 0.0) - transit.get(prob_of, dict()).get(cond_on, 0.0)) / delta
    return est_deriv


def scd_subsplit_via_subsplit_derivative(prob_of: Subsplit, via: Subsplit, wrt: PCSS, scd: SCDSet, transit: dict=None):
    root_subsplit = scd.root_subsplit()
    parent = wrt.parent
    parent_clade = wrt.parent_clade()
    # child = wrt.child
    if transit is None:
        transit = scd.transit_probabilities()
    # if parent.valid_descendant(via):
    #     return scd_subsplit_to_subsplit_cond_derivative(prob_of=via, cond_on=root_subsplit, wrt=wrt, scd=scd, transit=transit) * transit.get(prob_of, dict()).get(via, 0.0)
    # elif parent.valid_ancestor(via) and parent.valid_descendant(prob_of):
    #     return transit.get(via, dict()).get(root_subsplit, 0.0) * scd_subsplit_to_subsplit_cond_derivative(prob_of=prob_of, cond_on=via, wrt=wrt, scd=scd, transit=transit)
    # else:
    #     return 0.0
    return ((scd_subsplit_to_subsplit_cond_derivative(prob_of=via, cond_on=root_subsplit, wrt=wrt, scd=scd,
                                                      transit=transit)
             * transit.get(prob_of, dict()).get(via, 0.0))
            + (transit.get(via, dict()).get(root_subsplit, 0.0) * scd_subsplit_to_subsplit_cond_derivative(
                prob_of=prob_of, cond_on=via, wrt=wrt, scd=scd, transit=transit)))


def scd_estimate_subsplit_via_subsplit_derivative(prob_of: Subsplit, via: Subsplit, wrt: PCSS, scd: SCDSet, delta: float=0.0001, transit=None, transit2=None):
    root_subsplit = scd.root_subsplit()
    if transit is None:
        transit = scd.transit_probabilities()
    if transit2 is None:
        parent = wrt.parent_clade()
        scd2 = scd.copy()
        scd2.add_log(wrt, delta)
        scd2.normalize(parent)
        transit2 = scd2.transit_probabilities()
    est_deriv = (transit2.get(prob_of, dict()).get(via, 0.0)*transit2.get(via, dict()).get(root_subsplit, 0.0)
                 - transit.get(prob_of, dict()).get(via, 0.0)*transit.get(via, dict()).get(root_subsplit, 0.0)) / delta
    return est_deriv


def scd_restricted_subsplit_derivative(restriction: Clade, prob_of: Subsplit, wrt: PCSS, scd: SCDSet, transit: dict=None):
    root_subsplit = scd.root_subsplit()
    if transit is None:
        transit = scd.transit_probabilities()
    result = 0.0
    for subsplit in scd.iter_subsplits(include_root=True):
        restricted_subsplit = subsplit.restrict(restriction)
        if not restricted_subsplit == prob_of or (restricted_subsplit.is_trivial() and subsplit != root_subsplit):
            continue
        result += scd_subsplit_derivative(prob_of=subsplit, wrt=wrt, scd=scd, transit=transit)
    return result


def scd_estimate_restricted_subsplit_derivative(restriction: Clade, prob_of: Subsplit, wrt: PCSS, scd: SCDSet, delta: float=0.0001, uncond=None, uncond2=None):
    if uncond is None:
        scd_res = scd.restrict(restriction)
        uncond = scd_res.subsplit_probabilities()
    if uncond2 is None:
        parent = wrt.parent_clade()
        scd2 = scd.copy()
        scd2.add_log(wrt, delta)
        scd2.normalize(parent)
        scd2_res = scd2.restrict(restriction)
        uncond2 = scd2_res.subsplit_probabilities()
    est_deriv = (uncond2[prob_of] - uncond[prob_of]) / delta
    return est_deriv


def scd_restricted_pcss_derivative(restriction: Clade, prob_of: PCSS, wrt: PCSS, scd: SCDSet, transit: dict=None):
    root_subsplit = scd.root_subsplit()
    parent = prob_of.parent
    # parent_clade = prob_of.parent_clade()
    child = prob_of.child
    if transit is None:
        transit = scd.transit_probabilities()
    result = 0.0
    for destination in transit:
        dest_res = destination.restrict(restriction)
        if dest_res != child:
            continue
        for ancestor in transit[destination]:
            if not isinstance(ancestor, Subsplit):
                continue
            ansr_res = ancestor.restrict(restriction)
            if ansr_res != parent or (ansr_res.is_trivial() and ancestor != root_subsplit):
                continue
            result += scd_subsplit_via_subsplit_derivative(prob_of=destination, via=ancestor, wrt=wrt, scd=scd, transit=transit)
    return result


def scd_estimate_restricted_pcss_derivative(restriction: Clade, prob_of: PCSS, wrt: PCSS, scd: SCDSet, delta: float=0.0001, res_pcss_probs=None, res_pcss_probs2=None):
    if res_pcss_probs is None:
        scd_res = scd.restrict(restriction)
        res_pcss_probs = scd_res.pcss_probabilities()
    if res_pcss_probs2 is None:
        parent = wrt.parent
        scd2 = scd.copy()
        scd2.add_log(wrt, delta)
        scd2.normalize(parent)
        scd2_res = scd2.restrict(restriction)
        res_pcss_probs2 = scd2_res.pcss_probabilities()
    est_deriv = (res_pcss_probs2[prob_of] - res_pcss_probs[prob_of]) / delta
    return est_deriv


def scd_restricted_conditional_derivative(restriction: Clade, prob_of: PCSS, wrt: PCSS, scd: SCDSet, transit: dict=None, restricted_scd: SCDSet=None, restricted_subsplit_probs: dict=None, restricted_pcss_probs: dict=None):
    if transit is None:
        transit = scd.transit_probabilities()
    if restricted_subsplit_probs is None or restricted_pcss_probs is None:
        if restricted_scd is None:
            restricted_scd = scd.restrict(restriction)
        if restricted_subsplit_probs is None:
            restricted_subsplit_probs = restricted_scd.subsplit_probabilities()
        if restricted_pcss_probs is None:
            restricted_pcss_probs = restricted_scd.pcss_probabilities()
    # Quotient rule d(top/bot) = (bot*dtop-top*dbot) / bot**2
    top = restricted_pcss_probs[prob_of]
    # print(f"top: {top}")
    bot = restricted_subsplit_probs[prob_of.parent]
    # print(f"bot: {bot}")
    dtop = scd_restricted_pcss_derivative(restriction=restriction, prob_of=prob_of, wrt=wrt, scd=scd, transit=transit)
    # print(f"dtop: {dtop}")
    dbot = scd_restricted_subsplit_derivative(restriction=restriction, prob_of=prob_of.parent, wrt=wrt, scd=scd, transit=transit)
    # print(f"dbot: {dbot}")
    return (bot*dtop - top*dbot) / bot**2


def scd_estimate_restricted_conditional_derivative(restriction: Clade, prob_of: PCSS, wrt: PCSS, scd: SCDSet, delta: float=0.0001, scd_res: SCDSet=None, scd2_res:SCDSet=None):
    if scd_res is None:
        scd_res = scd.restrict(restriction)
    if scd2_res is None:
        parent = wrt.parent
        scd2 = scd.copy()
        scd2.add_log(wrt, delta)
        scd2.normalize(parent)
        scd2_res = scd2.restrict(restriction)
    est_deriv = (scd2_res[prob_of] - scd_res[prob_of]) / delta
    return est_deriv


def scd_restricted_kl_derivative(wrt: PCSS, scd: SCDSet, other: SCDSet, transit: dict=None, restricted_scd: SCDSet=None, other_pcss_probs: dict = None):
    restriction = other.root_clade()
    if not restriction.issubset(scd.root_clade()):
        raise ValueError("Non-concentric taxon sets.")
    if transit is None:
        transit = scd.transit_probabilities()
    if restricted_scd is None:
        restricted_scd = scd.restrict(restriction)
    # restricted_subsplit_probs = restricted_scd.subsplit_probabilities()
    # restricted_pcss_probs = restricted_scd.pcss_probabilities()
    if other_pcss_probs is None:
        other_pcss_probs = other.pcss_probabilities()
    result = 0.0
    for other_pcss in other.iter_pcss():
        other_pcss_prob = other_pcss_probs[other_pcss]
        # print(f"other_pcss_prob={other_pcss_prob}")
        restricted_cond = restricted_scd[other_pcss]
        # print(f"restricted_cond={restricted_cond}")
        # print(f"restriction={restriction}, prob_of={other_pcss}, wrt={wrt}")
        restricted_cond_deriv = scd_restricted_conditional_derivative(restriction=restriction, prob_of=other_pcss, wrt=wrt, scd=scd, transit=transit, restricted_scd=restricted_scd)
        # print(f"restricted_cond_deriv={restricted_cond_deriv}")
        intermediate_result = -other_pcss_prob * restricted_cond_deriv / restricted_cond
        # print(f"intermediate_result={intermediate_result}")
        result += intermediate_result
    return result


def scd_estimate_restricted_kl_derivative(wrt: PCSS, scd: SCDSet, other: SCDSet, delta: float=0.0001, scd_res: SCDSet = None, scd2_res: SCDSet = None):
    restriction = other.root_clade()
    if scd_res is None:
        scd_res = scd.restrict(restriction)
    if scd2_res is None:
        parent = wrt.parent
        scd2 = scd.copy()
        scd2.add_log(wrt, delta)
        scd2.normalize(parent)
        scd2_res = scd2.restrict(restriction)
    est_deriv = (other.kl_divergence(scd2_res) - other.kl_divergence(scd_res)) / delta
    return est_deriv


def scd_restricted_kl_gradient(scd: SCDSet, other: 'SCDSet',
                               transit: dict = None,
                               restricted_self: 'SCDSet' = None,
                               restricted_subsplit_probs: dict = None,
                               other_subsplit_probs: dict = None):
    restriction = other.root_clade()
    root_subsplit = scd.root_subsplit()
    if not restriction.issubset(scd.root_clade()):
        raise ValueError("Argument 'other' has taxon set not a subset of this taxon set.")
    if transit is None:
        transit = scd.transit_probabilities()
    if restricted_self is None:
        restricted_self = scd.restrict(restriction)
    if restricted_subsplit_probs is None:
        restricted_subsplit_probs = restricted_self.subsplit_probabilities()
    if other_subsplit_probs is None:
        other_subsplit_probs = other.subsplit_probabilities()
    result = dict()
    # ancestor_factors = dict()
    # cond_factors = dict()
    for ancestor in scd.iter_subsplits(include_root=True):
        anc_res = ancestor.restrict(restriction)
        if anc_res.is_trivial() and ancestor != root_subsplit:
            continue
        # if anc_res not in ancestor_factors:
        #     ancestor_factor = other_subsplit_probs[anc_res] / restricted_subsplit_probs[anc_res]
        #     ancestor_factors[anc_res] = ancestor_factor
        # else:
        #     ancestor_factor = ancestor_factors[anc_res]
        n_dist_factor = len(list(anc_res.nontrivial_children()))
        ancestor_factor = other_subsplit_probs[anc_res] / restricted_subsplit_probs[anc_res]
        anc_deriv_factor = dict()
        for wrt in scd.iter_pcss():
            if wrt not in result:
                result[wrt] = 0.0
            anc_deriv_factor[wrt] = scd_subsplit_derivative(prob_of=ancestor, wrt=wrt, scd=scd, transit=transit)
            result[wrt] += n_dist_factor * ancestor_factor * anc_deriv_factor[wrt]
        for descendant in scd.iter_subsplits(include_root=False):
            desc_res = descendant.restrict(restriction)
            if desc_res.is_trivial() or not desc_res.valid_parent(anc_res):
                continue
            pcss_res = PCSS(anc_res, desc_res)
            # if pcss_res not in cond_factors:
            #     cond_factor = other[pcss_res] / restricted_self[pcss_res]
            #     cond_factors[desc_res] = cond_factor
            # else:
            #     cond_factor = cond_factors[desc_res]
            cond_factor = other.get(pcss_res) / restricted_self.get(pcss_res)
            via_deriv_factor = dict()
            for wrt in scd.iter_pcss():
                via_deriv_factor[wrt] = scd_subsplit_via_subsplit_derivative(prob_of=descendant, via=ancestor, wrt=wrt, scd=scd, transit=transit)
                result[wrt] += -ancestor_factor * cond_factor * via_deriv_factor[wrt]
    return result


def brute_kl_gradient(scd: SCDSet, other: SCDSet, scd_res: SCDSet = None, transit: dict = None, other_pcss_probs: dict = None):
    theo_res = dict()
    # est_res = dict()
    restriction = other.root_clade()
    if scd_res is None:
        scd_res = scd.restrict(restriction)
    if transit is None:
        transit = scd.transit_probabilities()
    if other_pcss_probs is None:
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
        # est = scd_estimate_restricted_kl_derivative(wrt=wrt_for, scd=scd, other=other, delta=delta,
        #                                             scd_res=scd_res, scd2_res=scd2_res)
        # est_res[wrt_for] = est
        # if not theo == est == 0.0:
        #     print(f"KL wrt {wrt_for}")
        #     print(f"Theo: {theo_res[wrt_for]:8.4g}")
        #     print(f"Est:  {est_res[wrt_for]:8.4g}")
    return theo_res


def scd_restricted_kl_ancestor_derivative(wrt: PCSS, scd: SCDSet, other: SCDSet, transit: dict=None, restricted_scd: SCDSet=None, other_ss_probs: dict = None):
    restriction = other.root_clade()
    if not restriction.issubset(scd.root_clade()):
        raise ValueError("Non-concentric taxon sets.")
    if transit is None:
        transit = scd.transit_probabilities()
    if restricted_scd is None:
        restricted_scd = scd.restrict(restriction)
    restricted_ss_probs = restricted_scd.subsplit_probabilities()
    if other_ss_probs is None:
        other_ss_probs = other.subsplit_probabilities()
    result = 0.0
    for other_parent in other.iter_subsplits(include_root=True):
        n_dist_factor = len(list(other_parent.nontrivial_children()))
        other_ss_prob = other_ss_probs[other_parent]
        restricted_ss_prob = restricted_ss_probs[other_parent]
        restricted_ss_deriv = scd_restricted_subsplit_derivative(restriction=restriction, prob_of=other_parent, wrt=wrt, scd=scd, transit=transit)
        intermediate_result = n_dist_factor * other_ss_prob * restricted_ss_deriv / restricted_ss_prob
        result += intermediate_result
    return result


def scd_restricted_kl_joint_derivative(wrt: PCSS, scd: SCDSet, other: SCDSet, transit: dict=None, restricted_scd: SCDSet=None, other_pcss_probs: dict = None):
    restriction = other.root_clade()
    if not restriction.issubset(scd.root_clade()):
        raise ValueError("Non-concentric taxon sets.")
    if transit is None:
        transit = scd.transit_probabilities()
    if restricted_scd is None:
        restricted_scd = scd.restrict(restriction)
    restricted_pcss_probs = restricted_scd.pcss_probabilities()
    if other_pcss_probs is None:
        other_pcss_probs = other.pcss_probabilities()
    result = 0.0
    for other_pcss in other.iter_pcss():
        other_pcss_prob = other_pcss_probs[other_pcss]
        restricted_pcss_prob = restricted_pcss_probs[other_pcss]
        restricted_pcss_deriv = scd_restricted_pcss_derivative(restriction=restriction, prob_of=other_pcss, wrt=wrt, scd=scd, transit=transit)
        intermediate_result = other_pcss_prob * restricted_pcss_deriv / restricted_pcss_prob
        result += intermediate_result
    return result


def scd_restricted_kl_ancestor_gradient(scd: SCDSet, other: 'SCDSet', transit: dict = None, restricted_self: 'SCDSet' = None):
    restriction = other.root_clade()
    root_subsplit = scd.root_subsplit()
    if not restriction.issubset(scd.root_clade()):
        raise ValueError("Argument 'other' has taxon set not a subset of this taxon set.")
    if transit is None:
        transit = scd.transit_probabilities()
    if restricted_self is None:
        restricted_self = scd.restrict(restriction)
    restricted_ss_probs = restricted_self.subsplit_probabilities()
    # restricted_clade = restricted_self.clade_probabilities()
    other_ss_probs = other.subsplit_probabilities()
    # other_clade = other.clade_probabilities()
    # verbose = False
    # if isinstance(verbose_dict, dict):
    #     verbose = True
    result = dict()
    ancestor_factors = dict()
    cond_factors = dict()
    for ancestor in scd.iter_subsplits(include_root=True):
        anc_res = ancestor.restrict(restriction)
        if anc_res.is_trivial() and ancestor != root_subsplit:
            continue
        # if anc_res not in ancestor_factors:
        #     ancestor_factor = other_ss_probs[anc_res] / restricted_ss_probs[anc_res]
        #     ancestor_factors[anc_res] = ancestor_factor
        # else:
        #     ancestor_factor = ancestor_factors[anc_res]
        n_dist_factor = len(list(anc_res.nontrivial_children()))
        ancestor_factor = other_ss_probs[anc_res] / restricted_ss_probs[anc_res]
        anc_deriv_factor = dict()
        for wrt in scd.iter_pcss():
            if wrt not in result:
                result[wrt] = 0.0
            anc_deriv_factor[wrt] = scd_subsplit_derivative(prob_of=ancestor, wrt=wrt, scd=scd, transit=transit)
            result[wrt] += n_dist_factor * ancestor_factor * anc_deriv_factor[wrt]
    return result


def scd_restricted_kl_joint_gradient(scd: SCDSet, other: 'SCDSet', transit: dict = None, restricted_self: 'SCDSet' = None):
    restriction = other.root_clade()
    root_subsplit = scd.root_subsplit()
    if not restriction.issubset(scd.root_clade()):
        raise ValueError("Argument 'other' has taxon set not a subset of this taxon set.")
    if transit is None:
        transit = scd.transit_probabilities()
    if restricted_self is None:
        restricted_self = scd.restrict(restriction)
    restricted_ss_probs = restricted_self.subsplit_probabilities()
    # restricted_clade = restricted_self.clade_probabilities()
    other_ss_probs = other.subsplit_probabilities()
    # other_clade = other.clade_probabilities()
    # verbose = False
    # if isinstance(verbose_dict, dict):
    #     verbose = True
    result = dict()
    ancestor_factors = dict()
    cond_factors = dict()
    for ancestor in scd.iter_subsplits(include_root=True):
        anc_res = ancestor.restrict(restriction)
        if anc_res.is_trivial() and ancestor != root_subsplit:
            continue
        # if anc_res not in ancestor_factors:
        #     ancestor_factor = other_ss_probs[anc_res] / restricted_ss_probs[anc_res]
        #     ancestor_factors[anc_res] = ancestor_factor
        # else:
        #     ancestor_factor = ancestor_factors[anc_res]
        ancestor_factor = other_ss_probs[anc_res] / restricted_ss_probs[anc_res]
        anc_deriv_factor = dict()
        # for wrt in scd.iter_pcss():
        #     if wrt not in result:
        #         result[wrt] = 0.0
        #     anc_deriv_factor[wrt] = scd_subsplit_derivative(prob_of=ancestor, wrt=wrt, scd=scd, transit=transit)
        #     result[wrt] += ancestor_factor * anc_deriv_factor[wrt]
        for descendant in scd.iter_subsplits(include_root=False):
            desc_res = descendant.restrict(restriction)
            if desc_res.is_trivial() or not desc_res.valid_parent(anc_res):
                continue
            pcss_res = PCSS(anc_res, desc_res)
            # if pcss_res not in cond_factors:
            #     cond_factor = other[pcss_res] / restricted_self[pcss_res]
            #     cond_factors[desc_res] = cond_factor
            # else:
            #     cond_factor = cond_factors[desc_res]
            cond_factor = other.get(pcss_res) / restricted_self.get(pcss_res)
            via_deriv_factor = dict()
            for wrt in scd.iter_pcss():
                if wrt not in result:
                    result[wrt] = 0.0
                via_deriv_factor[wrt] = scd_subsplit_via_subsplit_derivative(prob_of=descendant, via=ancestor, wrt=wrt, scd=scd, transit=transit)
                result[wrt] += ancestor_factor * cond_factor * via_deriv_factor[wrt]
    return result


def scd_restricted_conditional_ancestor_derivative(restriction: Clade, prob_of: PCSS, wrt: PCSS, scd: SCDSet, transit: dict=None, restricted_scd: SCDSet=None, restricted_subsplit_probs: dict=None, restricted_pcss_probs: dict=None):
    if transit is None:
        transit = scd.transit_probabilities()
    if restricted_subsplit_probs is None or restricted_pcss_probs is None:
        if restricted_scd is None:
            restricted_scd = scd.restrict(restriction)
        if restricted_subsplit_probs is None:
            restricted_subsplit_probs = restricted_scd.subsplit_probabilities()

    par_prob = restricted_subsplit_probs[prob_of.parent]
    pcss_cond_prob = restricted_scd[prob_of]
    d_par_prob = scd_restricted_subsplit_derivative(restriction=restriction, prob_of=prob_of.parent, wrt=wrt, scd=scd, transit=transit)
    return pcss_cond_prob * d_par_prob / par_prob


def scd_restricted_conditional_joint_derivative(restriction: Clade, prob_of: PCSS, wrt: PCSS, scd: SCDSet, transit: dict=None, restricted_scd: SCDSet=None, restricted_subsplit_probs: dict=None, restricted_pcss_probs: dict=None):
    if transit is None:
        transit = scd.transit_probabilities()
    if restricted_subsplit_probs is None or restricted_pcss_probs is None:
        if restricted_scd is None:
            restricted_scd = scd.restrict(restriction)
        if restricted_subsplit_probs is None:
            restricted_subsplit_probs = restricted_scd.subsplit_probabilities()

    par_prob = restricted_subsplit_probs[prob_of.parent]
    d_pcss_prob = scd_restricted_pcss_derivative(restriction=restriction, prob_of=prob_of, wrt=wrt, scd=scd, transit=transit)
    return d_pcss_prob / par_prob


def scd_restricted_conditional_derivative_alt1(restriction: Clade, prob_of: PCSS, wrt: PCSS, scd: SCDSet, transit: dict=None, restricted_scd: SCDSet=None, restricted_subsplit_probs: dict=None, restricted_pcss_probs: dict=None):
    if transit is None:
        transit = scd.transit_probabilities()
    if restricted_subsplit_probs is None or restricted_pcss_probs is None:
        if restricted_scd is None:
            restricted_scd = scd.restrict(restriction)
        if restricted_subsplit_probs is None:
            restricted_subsplit_probs = restricted_scd.subsplit_probabilities()
        if restricted_pcss_probs is None:
            restricted_pcss_probs = restricted_scd.pcss_probabilities()

    par_prob = restricted_subsplit_probs[prob_of.parent]
    d_pcss_prob = scd_restricted_pcss_derivative(restriction=restriction, prob_of=prob_of, wrt=wrt, scd=scd, transit=transit)
    pcss_cond_prob = restricted_scd[prob_of]
    d_par_prob = scd_restricted_subsplit_derivative(restriction=restriction, prob_of=prob_of.parent, wrt=wrt, scd=scd, transit=transit)
    return (d_pcss_prob - pcss_cond_prob*d_par_prob) / par_prob


def scd_restricted_conditional_derivative_alt2(restriction: Clade, prob_of: PCSS, wrt: PCSS, scd: SCDSet, transit: dict=None, restricted_scd: SCDSet=None, restricted_subsplit_probs: dict=None, restricted_pcss_probs: dict=None):
    if transit is None:
        transit = scd.transit_probabilities()
    if restricted_subsplit_probs is None or restricted_pcss_probs is None:
        if restricted_scd is None:
            restricted_scd = scd.restrict(restriction)
        if restricted_subsplit_probs is None:
            restricted_subsplit_probs = restricted_scd.subsplit_probabilities()
        if restricted_pcss_probs is None:
            restricted_pcss_probs = restricted_scd.pcss_probabilities()

    ancestor_deriv = scd_restricted_conditional_ancestor_derivative(restriction=restriction, prob_of=prob_of, wrt=wrt, scd=scd, transit=transit)
    joint_deriv = scd_restricted_conditional_joint_derivative(restriction=restriction, prob_of=prob_of, wrt=wrt, scd=scd, transit=transit)
    return joint_deriv - ancestor_deriv


def scd_restricted_kl_derivative_alt1(wrt: PCSS, scd: SCDSet, other: SCDSet, transit: dict=None, restricted_scd: SCDSet=None, other_pcss_probs: dict = None):
    restriction = other.root_clade()
    if not restriction.issubset(scd.root_clade()):
        raise ValueError("Non-concentric taxon sets.")
    if transit is None:
        transit = scd.transit_probabilities()
    if restricted_scd is None:
        restricted_scd = scd.restrict(restriction)
    # restricted_subsplit_probs = restricted_scd.subsplit_probabilities()
    # restricted_pcss_probs = restricted_scd.pcss_probabilities()
    if other_pcss_probs is None:
        other_pcss_probs = other.pcss_probabilities()
    result = 0.0
    for other_pcss in other.iter_pcss():
        other_pcss_prob = other_pcss_probs[other_pcss]
        # print(f"other_pcss_prob={other_pcss_prob}")
        restricted_cond = restricted_scd[other_pcss]
        # print(f"restricted_cond={restricted_cond}")
        # print(f"restriction={restriction}, prob_of={other_pcss}, wrt={wrt}")
        restricted_cond_deriv = scd_restricted_conditional_derivative_alt1(restriction=restriction, prob_of=other_pcss, wrt=wrt, scd=scd, transit=transit, restricted_scd=restricted_scd)
        # print(f"restricted_cond_deriv={restricted_cond_deriv}")
        intermediate_result = -other_pcss_prob * restricted_cond_deriv / restricted_cond
        # print(f"intermediate_result={intermediate_result}")
        result += intermediate_result
    return result


def scd_restricted_kl_derivative_alt2(wrt: PCSS, scd: SCDSet, other: SCDSet, transit: dict=None, restricted_scd: SCDSet=None, other_pcss_probs: dict = None):
    restriction = other.root_clade()
    if not restriction.issubset(scd.root_clade()):
        raise ValueError("Non-concentric taxon sets.")
    if transit is None:
        transit = scd.transit_probabilities()
    if restricted_scd is None:
        restricted_scd = scd.restrict(restriction)
    # restricted_subsplit_probs = restricted_scd.subsplit_probabilities()
    # restricted_pcss_probs = restricted_scd.pcss_probabilities()
    if other_pcss_probs is None:
        other_pcss_probs = other.pcss_probabilities()
    result = 0.0
    for other_pcss in other.iter_pcss():
        other_pcss_prob = other_pcss_probs[other_pcss]

        restricted_cond = restricted_scd[other_pcss]
        ancestor_deriv = scd_restricted_conditional_ancestor_derivative(restriction=restriction, prob_of=other_pcss,
                                                                        wrt=wrt, scd=scd, transit=transit)
        joint_deriv = scd_restricted_conditional_joint_derivative(restriction=restriction, prob_of=other_pcss, wrt=wrt,
                                                                  scd=scd, transit=transit)

        # restricted_cond_deriv = scd_restricted_conditional_derivative_alt2(restriction=restriction, prob_of=other_pcss, wrt=wrt, scd=scd, transit=transit, restricted_scd=restricted_scd)
        intermediate_result = other_pcss_prob * (ancestor_deriv - joint_deriv) / restricted_cond
        result += intermediate_result
    return result


def scd_restricted_kl_derivative_alt3(wrt: PCSS, scd: SCDSet, other: SCDSet, transit: dict=None, restricted_scd: SCDSet=None, other_pcss_probs: dict = None):
    restriction = other.root_clade()
    if not restriction.issubset(scd.root_clade()):
        raise ValueError("Non-concentric taxon sets.")
    if transit is None:
        transit = scd.transit_probabilities()
    if restricted_scd is None:
        restricted_scd = scd.restrict(restriction)
    restricted_subsplit_probs = restricted_scd.subsplit_probabilities()
    restricted_pcss_probs = restricted_scd.pcss_probabilities()
    other_subsplit_probs = other.subsplit_probabilities()
    if other_pcss_probs is None:
        other_pcss_probs = other.pcss_probabilities()
    result = 0.0
    for other_pcss in other.iter_pcss():
        other_parent_prob = other_subsplit_probs[other_pcss.parent]
        other_cond_prob = other[other_pcss]
        # other_pcss_prob = other_pcss_probs[other_pcss]
        restricted_parent_prob = restricted_subsplit_probs[other_pcss.parent]
        restricted_cond_prob = restricted_scd[other_pcss]
        # restricted_pcss_prob = restricted_pcss_probs[other_pcss]
        ancestor_component = (
            other_parent_prob * other_cond_prob
            * scd_restricted_subsplit_derivative(restriction=restriction,
                                                 prob_of=other_pcss.parent,
                                                 wrt=wrt, scd=scd, transit=transit)
            / restricted_parent_prob)

        joint_component = (
            other_parent_prob * other_cond_prob
            * scd_restricted_pcss_derivative(restriction=restriction,
                                             prob_of=other_pcss,
                                             wrt=wrt, scd=scd, transit=transit)
            / (restricted_parent_prob * restricted_cond_prob)
        )

        intermediate_result = ancestor_component - joint_component
        result += intermediate_result
    return result


def scd_restricted_all_subsplits_derivative(restriction: Clade, wrt: PCSS, scd: SCDSet, transit: dict=None):
    root_subsplit = scd.root_subsplit()
    if transit is None:
        transit = scd.transit_probabilities()
    result = dict()
    for subsplit in scd.iter_subsplits(include_root=True):
        restricted_subsplit = subsplit.restrict(restriction)
        if restricted_subsplit.is_trivial() and subsplit != root_subsplit:
            continue
        if restricted_subsplit not in result:
            result[restricted_subsplit] = 0.0
        result[restricted_subsplit] += scd_subsplit_derivative(prob_of=subsplit, wrt=wrt, scd=scd, transit=transit)
    return result
