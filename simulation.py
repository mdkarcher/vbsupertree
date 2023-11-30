from vbsupertree import *

from matplotlib import pyplot as plt
import seaborn as sns

# Decompress data/ folder if not done so already

# Set up the names and locations of the simulated data
base_name = "data/sims/tc40_sl500"
scenarios = ["all", "z0", "z2"]
references = ["z0", "z2"]
infixes = {
    "all": "_alltips",
    "z0": "_minus_z0",
    "z2": "_minus_z2"
}

# Load the scenarios and train SBNs
tree_dists = dict()
sbns = dict()
supports = dict()
restrictions = dict()
pcsp_probabilities = dict()
for key in scenarios:
    infix = infixes[key]
    print(f"Starting scenario '{key}'...")
    all_trees = parse_beast_nexus(f"{base_name}{infix}.trees")
    print("  Trees parsed")
    n_all = len(all_trees)
    # Removing first half for burn-in
    tree_dist_raw = TreeDistribution.from_list(all_trees[(n_all//2 + 1):])
    print("  Tree distribution created")
    # Removing trees with probability below 0.002
    tree_dist = tree_dist_raw.dust(0.002)
    tree_dist.normalize()
    tree_dists[key] = tree_dist
    print("  Tree distribution dusted")
    sbn = SBN.from_tree_distribution(tree_dist)
    sbns[key] = sbn
    print("  SBN trained")
    support = sbn.support()
    supports[key] = support
    print("  PCSP support summarized")
    restriction = sbn.root_clade()
    restrictions[key] = restriction
    pcsp_probability = sbn.pcsp_probabilities()
    pcsp_probabilities[key] = pcsp_probability

# Analyzing two references, one missing tip z0, the other missing tip z2
key1 = "z0"
key2 = "z2"
print(f"Starting reference pair {key1}, {key2}...")

# Finding the mutual PCSP support
mutual_support = supports[key1].mutualize(supports[key2])
mutual_support = mutual_support.prune()
print("  Mutual support built")

# Find PCSPs in the reference supports that have no equivalent (post-restriction) in the mutual support
uncovered_support = (
    supports[key1].to_set() - mutual_support.restrict(restrictions[key1]).to_set(),
    supports[key2].to_set() - mutual_support.restrict(restrictions[key2]).to_set()
)
print("  Uncovered support found")

# Calculate an upper bound on the probabilities affected by the uncovered PCSPs above
total_uncovered_probabilities = (
    sum(pcsp_probabilities[key1][pcsp] for pcsp in uncovered_support[0]),
    sum(pcsp_probabilities[key2][pcsp] for pcsp in uncovered_support[1])
)
print("  Total (non-mutually exclusive) uncovered probability calculated")
print(f"    = {total_uncovered_probabilities[0]}")
print(f"    = {total_uncovered_probabilities[1]}")

# Remove uncovered PCSPs in the references for KL-divergence compatibility
a = sbns[key1].copy()
a.remove_many(uncovered_support[0])
a = a.prune()
a.normalize()
b = sbns[key2].copy()
b.remove_many(uncovered_support[1])
b = b.prune()
b.normalize()
trimmed_sbns = (a, b)
print("  Trimmed SBNs built")

# Sample a starting SBN on the mutual support, with high-entropy conditional distributions
starting_sbn = SBN.random_from_support(
    support=mutual_support, concentration=10
)
print("  Starting SBN drawn")

# Trim uncovered PCSPs from the full-taxa posterior sample SBN for KL-compatibility
true_sbn_trim = sbns["all"].copy()
tst_support = true_sbn_trim.support()
tst_uncovered = tst_support.to_set() - mutual_support.to_set()
true_sbn_uncovered_probs = sum(pcsp_probabilities["all"][pcsp] for pcsp in tst_uncovered)
print("  Total (non-mutually exclusive) uncovered full-tips probability calculated")
print(f"    = {true_sbn_uncovered_probs}")
true_sbn_trim.remove_many(tst_uncovered)
true_sbn_trim = true_sbn_trim.prune()
true_sbn_trim.normalize()
print("  Trimmed full-tips SBN built")

# Run gradient descent
penalty = 0.50
supertree_sbn, kl_list, true_kl_list = starting_sbn.gradient_descent_pen(
    references=trimmed_sbns, penalty=penalty, starting_gamma=2.0, max_iteration=50, true_reference=true_sbn_trim
)

# Plotting figures
sns.set_context("poster")
fig, (ax_kl, ax_true_kl) = plt.subplots(1, 2, figsize=(10, 5), sharey='none', sharex='none', constrained_layout=True)
ax_kl.set(title="Supertree Loss", xlabel="Iteration", ylabel="Nats", yscale="log")
ax_kl.plot(kl_list)
ax_true_kl.set(title="KL vs. Truth", xlabel="Iteration", ylabel="", yscale="log")
ax_true_kl.plot(true_kl_list)
