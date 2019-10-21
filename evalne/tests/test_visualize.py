#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Mara Alexandru Cristian
# Contact: alexandru.mara@ugent.be
# Date: 18/12/2018

from __future__ import division

import numpy as np

from evalne.utils.viz_utils import *


def test():
    # Generate set of nodes
    ids = range(10)

    # Set some random node embeddings
    emb = np.array([[0.82420207, 0.93905952, 0.0443836, 0.54250611, 0.30456824],
                    [0.16079168, 0.15119187, 0.40094691, 0.79790962, 0.84341248],
                    [0.57813155, 0.97857005, 0.65691974, 0.32131624, 0.22398546],
                    [0.12097439, 0.91100631, 0.13567747, 0.55608758, 0.46882953],
                    [0.81526756, 0.39482937, 0.57112954, 0.73773972, 0.93670739],
                    [0.59268631, 0.87080881, 0.73983155, 0.31100985, 0.77501675],
                    [0.53125864, 0.60695178, 0.91817668, 0.2321715, 0.19028287],
                    [0.13669606, 0.69945586, 0.59937428, 0.08156526, 0.21188543],
                    [0.64635913, 0.01367627, 0.90677346, 0.28922694, 0.59633913],
                    [0.06707187, 0.33893169, 0.36597878, 0.0011946, 0.07324235]])

    # Create a graph
    g = nx.Graph()

    # Add some nodes and random edges
    g.add_nodes_from(ids)
    aux1 = np.random.choice(ids, 10)
    aux2 = np.random.choice(ids, 10)
    g.add_edges_from(zip(aux1, aux2))

    # Create a set of labels
    labels = dict(zip(ids, ids))

    # Test embedding plotting
    plot_emb2d(emb)

    # Test graph plotting with spring layout
    plot_graph2d(g)
    plot_graph2d(g, labels=labels, colors=ids)

    # Test graph plotting with given positions
    plot_graph2d(g, emb, labels=labels, colors=ids)
    plot_graph2d(g, emb=emb[:, 0:2])


if __name__ == "__main__":
    test()
