import os
import networkx as nx


def graph_to_png(graph):
    pydot_graph = nx.nx_pydot.to_pydot(graph)
    return pydot_graph.create_png()


def write_to_png(graph, file, dpi=300, **kwargs):
    try:
        os.makedirs(os.path.abspath(os.path.dirname(file)))
    except:
        pass
    pydot_graph = nx.drawing.nx_pydot.to_pydot(graph)

    # print(pydot_graph.source)
    pydot_graph.set('dpi', dpi)
    pydot_graph.set('simplify', True)
    pydot_graph.set('nodesep', 0.5)
    pydot_graph.set('ranksep', 0.5)
    pydot_graph.set('layout', 'dot')
    # pydot_graph.set_size('"10!,10!"')
    ## pydot_graph.set('size', (5000,5000))
    # pydot_graph.set_node_defaults(fontsize=11)
    # pydot_graph.set_edge_defaults(fontsize=11, weight=0.1)
    pydot_graph.set_graph_defaults(fixedsize=False)
    for k in kwargs:
        pydot_graph.set(k, kwargs[k])
    pydot_graph.write_png(file)
"""
    = Digraph('companies', filename='companies.gv',
              edge_attr={'weight': '1',
                         'fontsize': '11',
                         'fontcolor': 'blue',
                         'len': '4'},
              graph_attr={'fixedsize': 'false',
                          'bgcolor': 'transparent'},
              node_attr={'fontsize': '11',
                         'shape': 'plaintext',
                         'color': 'none',
                         'fontcolor': 'black'})


f.attr(layout="neato")
f.attr(nodesep='3')
f.attr(ranksep='3')
f.attr(size='5000,5000')
"""


def display_mdp(mdp_spec):
    from IPython.display import display, Image
    display(Image(graph_to_png(mdp_spec.to_graph())))
