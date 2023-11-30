from vbsupertree import *

from matplotlib import pyplot as plt
import seaborn as sns

# Decompress data/ folder if not done so already

# Load posterior tree samples
tree_dict = parse_topology_count("data/HCV/hcv_topology_count.txt", dust=1)
tree_dict_1844 = parse_topology_count("data/HCV/hcv-1844_topology_count.txt", dust=1)
tree_dict_1879 = parse_topology_count("data/HCV/hcv-1879_topology_count.txt", dust=1)
# Generate tree distributions
tree_dist = TreeDistribution(tree_dict)
tree_dist.normalize()
tree_dist_1844 = TreeDistribution(tree_dict_1844)
tree_dist_1844.normalize()
tree_dist_1879 = TreeDistribution(tree_dict_1879)
tree_dist_1879.normalize()
# Train SBNs on the tree distributions
sbn = SBN.from_tree_distribution(tree_dist)
sbn_1844 = SBN.from_tree_distribution(tree_dist_1844)
sbn_1879 = SBN.from_tree_distribution(tree_dist_1879)
# Find the PCSP supports
support = sbn.support()
support_1844 = sbn_1844.support()
support_1879 = sbn_1879.support()
# Calculate PCSP probabilities
pcsp_probabilities = sbn.pcsp_probabilities()
pcsp_probabilities_1844 = sbn_1844.pcsp_probabilities()
pcsp_probabilities_1879 = sbn_1879.pcsp_probabilities()
# Find mutual support
mutual_support = support_1844.mutualize(support_1879)
# Check for completeness
mutual_support.is_complete()
mutual_support = mutual_support.prune(verbose=True)
mutual_support.is_complete()

# Find PCSPs in the reference supports that have no equivalent (post-restriction) in the mutual support
uncovered_support_1844 = support_1844.to_set() - mutual_support.restrict(sbn_1844.root_clade()).to_set()
uncovered_support_1879 = support_1879.to_set() - mutual_support.restrict(sbn_1879.root_clade()).to_set()
len(uncovered_support_1844)
len(uncovered_support_1879)

# Calculate an upper bound on the probabilities affected by the uncovered PCSPs above
total_uncovered_probabilities_1844 = sum(pcsp_probabilities_1844[pcsp] for pcsp in uncovered_support_1844)
total_uncovered_probabilities_1879 = sum(pcsp_probabilities_1879[pcsp] for pcsp in uncovered_support_1879)
print(total_uncovered_probabilities_1844)
print(total_uncovered_probabilities_1879)

# Remove uncovered PCSPs in the references for KL-divergence compatibility
trimmed_sbn_1844 = sbn_1844.copy()
trimmed_sbn_1844.remove_many(uncovered_support_1844)
trimmed_sbn_1844 = trimmed_sbn_1844.prune()
trimmed_sbn_1844.normalize()

trimmed_sbn_1879 = sbn_1879.copy()
trimmed_sbn_1879.remove_many(uncovered_support_1879)
trimmed_sbn_1879 = trimmed_sbn_1879.prune()
trimmed_sbn_1879.normalize()

# Sample a starting SBN on the mutual support, with high-entropy conditional distributions
starting_sbn = SBN.random_from_support(support=mutual_support, concentration=10)

# Trim uncovered PCSPs from the full-taxa posterior sample SBN for KL-compatibility
true_sbn_trim = sbn.copy()
tst_support = true_sbn_trim.support()
tst_uncovered = tst_support.to_set() - starting_sbn.support().to_set()

true_sbn_trim.remove_many(tst_uncovered)
true_sbn_trim = true_sbn_trim.prune()
true_sbn_trim.normalize()

# Calculate starting KL-divergence between the full-taxa posterior sample SBN and the starting supertree SBN candidate
true_sbn_trim.kl_divergence(starting_sbn)

# Run gradient descent
penalty=0.50
supertree_sbn, kl_list, true_kl_list = starting_sbn.gradient_descent_pen(
    references=[trimmed_sbn_1844, trimmed_sbn_1879], penalty=penalty,
    starting_gamma=2.0, max_iteration=50, true_reference=true_sbn_trim
)

# Plotting figures
plt.rcParams.update({'font.size': 100})
sns.set_context("poster")
fig, (ax_kl, ax_true_kl) = plt.subplots(1, 2, figsize=(10, 5), sharey='none', sharex='none', constrained_layout=True)
ax_kl.set(title="Supertree Loss", xlabel="Iteration", ylabel="Nats", yscale="log")
ax_kl.plot(kl_list)
ax_true_kl.set(title="KL vs. Truth", xlabel="Iteration", ylabel="", yscale="log")
ax_true_kl.plot(true_kl_list)
