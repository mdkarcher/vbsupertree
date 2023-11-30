import re
from collections import Counter
import vbsupertree as vbs


def parse_translation(translate_text):
    item_list = re.findall("[0-9]+ \w+", translate_text)
    result = dict()
    for item in item_list:
        fr, to = item.split(" ")
        result[fr] = to
    return result


def parse_beast_nexus(filename):
    with open(filename) as f:
        text_data = f.read()
    text_sections = text_data.split(";\n")
    translate_text = next(section for section in text_sections if "Translate" in section)
    map_dict = parse_translation(translate_text)

    tree_sections = filter(lambda x: "tree " in x, text_sections)
    trees = []
    for tree_section in tree_sections:
        _, newick_section = tree_section.split(" [&R] ")
        newick_fixed = newick_section.replace("[&rate=1.0]", "")
        tree = vbs.MyTree(newick_fixed + ";")
        tree.rename_tips(map_dict)
        trees.append(tree)
    return trees


def parse_beast_nexus_hetchron(filename):
    with open(filename) as f:
        text_data = f.read()
    text_sections = text_data.split(";\n")
    translate_text = next(section for section in text_sections if "Translate" in section)
    map_dict = parse_translation(translate_text)

    tree_sections = filter(lambda x: "tree " in x, text_sections)
    trees = []
    for tree_section in tree_sections:
        _, newick_section = tree_section.split(" [&R] ")
        # newick_fixed = re.sub(r"\[.*\]", "", newick_section)
        newick_fixed = re.sub(r"\[&rate=\d+.\d+(e|E)?-?\d*\]", "", newick_section)
        # newick_fixed = newick_section.replace("[&rate=1.0]", "")
        tree = vbs.MyTree(newick_fixed + ";")
        tree.rename_tips(map_dict)
        trees.append(tree)
    return trees


def parse_topology_count(filename, dust=0):
    result = Counter()
    with open(filename) as ff:
        for line in ff:
            count_str, newick = line.strip().split(sep=" ", maxsplit=1)
            count = int(count_str)
            if count <= dust:
                continue
            tree = vbs.MyTree(newick)
            result[tree] = count
    return result



