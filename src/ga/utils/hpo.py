import os
import tarfile
import tempfile
from typing import List, Tuple

import networkx as nx
import pandas as pd
import wget
from tqdm import tqdm


def make_hpo_labels_df(
        url = 'https://kg-hub.berkeleybop.io/kg-obo/hp/2023-04-05/hp_kgx_tsv.tar.gz',
        hp_prefix= ['HP:'],
        cols_to_keep=['id', 'name', 'description'],
        rename_id_col='hpo_term_id',
    ) -> List[Tuple]:
    # get tmp file name
    tmpdir = tempfile.TemporaryDirectory()
    tmpfile = tempfile.NamedTemporaryFile().file.name
    wget.download(url, tmpfile)

    this_tar = tarfile.open(tmpfile, 'r:gz')
    this_tar.extractall(path=tmpdir.name)

    # show files in tmpdir
    node_files = [f for f in os.listdir(tmpdir.name) if 'nodes' in f]
    if len(node_files) != 1:
        raise RuntimeError("Didn't find exactly one edge file in {}".format(tmpdir.name))
    node_file = node_files[0]

    nodes_df = pd.read_csv(os.path.join(tmpdir.name, node_file), sep='\t')
    # select only HP terms
    nodes_df = nodes_df[nodes_df['id'].str.startswith(tuple(hp_prefix))]
    # select only cols we want
    nodes_df = nodes_df[cols_to_keep]
    # rename id col
    nodes_df.rename(columns={'id': rename_id_col}, inplace=True)
    return nodes_df


def make_hpo_closures_and_graph(
        url = 'https://kg-hub.berkeleybop.io/kg-obo/hp/2023-04-05/hp_kgx_tsv.tar.gz',
        pred_col='predicate',
        subject_prefixes= ['HP:'],
        object_prefixes= ['HP:'],
        predicates = ['biolink:subclass_of'],
        root_node_to_use ='HP:0000118',
        include_self_in_closure=False,
    ) -> (List[Tuple], nx.DiGraph):
    # get tmp file name
    tmpdir = tempfile.TemporaryDirectory()
    tmpfile = tempfile.NamedTemporaryFile().file.name
    wget.download(url, tmpfile)

    this_tar = tarfile.open(tmpfile, 'r:gz')
    this_tar.extractall(path=tmpdir.name)

    # show files in tmpdir
    edge_files = [f for f in os.listdir(tmpdir.name) if 'edges' in f]
    if len(edge_files) != 1:
        raise RuntimeError("Didn't find exactly one edge file in {}".format(tmpdir.name))
    edge_file = edge_files[0]

    edges_df = pd.read_csv(os.path.join(tmpdir.name, edge_file), sep='\t')
    if pred_col not in edges_df.columns:
        raise RuntimeError("Didn't find predicate column {} in {} cols: {}".format(pred_col, edge_file, "\n".join(edges_df.columns)))

    # get edges of interest
    edges_df = edges_df[edges_df[pred_col].isin(predicates)]
    # get edges involving nodes of interest
    edges_df = edges_df[edges_df['subject'].str.startswith(tuple(subject_prefixes))]
    edges_df = edges_df[edges_df['object'].str.startswith(tuple(object_prefixes))]

    # make into list of tuples
    # note that we are swapping order of edges (object -> subject) so that descendants are leaf terms
    # and ancestors are root nodes (assuming edges are subclass_of edges)
    edges = list(edges_df[['object', 'subject']].itertuples(index=False, name=None))

    # Create a directed graph using NetworkX
    graph = nx.DiGraph(edges)

    # Create a subgraph from the descendants of phenotypic_abnormality
    descendants = nx.descendants(graph, root_node_to_use)
    pa_subgraph = graph.subgraph(descendants)

    def compute_closure(node):
        return set(nx.ancestors(graph, node))

    # Compute closures for each node
    closures = []
    # set message for tqdm

    for node in tqdm(pa_subgraph.nodes(), desc="Computing closures"):
        if include_self_in_closure:
            closures.append((node, 'dummy_predicate', node))
        for anc in compute_closure(node):
            closures.append((node, 'dummy_predicate', anc))

    return closures, graph


