SurpriseMeMore
-------------------------------------------------------------------

SurpriseMeMore is a toolbox for detecting mesoscale structure in networks, released as a python3 module. 

SurpriseMeMore provides the user with a variety of solvers, based on the _surprise_ framework, for the detection of mesoscale structures ( e.g. communities, core-periphery) in networks.

The models implemented in SurpriseMeMore are presented in a forthcoming [paper](https://arxiv.org/abs/2106.05055) on arXiv.
If you use the module for your scientific research, please consider citing us:

```
    @misc{marchese2021detecting,
      title={Detecting mesoscale structures by surprise}, 
      author={Emiliano Marchese and Guido Caldarelli and Tiziano Squartini},
      year={2021},
      eprint={2106.05055},
      archivePrefix={arXiv},
      primaryClass={physics.soc-ph}
    }
```

#### Table Of Contents
- [Currently Implemented Methods](#currently-implemented-methods)
- [Installation](#installation)
- [Dependencies](#dependencies)
- [Some Examples](#some-examples)
- [Development](#development)
- [Credits](#credits)

## Currently Implemented Methods
The available methods, for both directed and undirected networks, are:

* *Community detection on binary networks* 
* *Community detection on weighted networks with integer weights* 
* *Community detection on weighted networks with continuous weights* 
* *Core-Peryphery detection on binary networks* 
* *Core-Peryphery detection on weighted networks with integer weights*

Installation
------------
SurpriseMeMore can be installed via pip. You can get it from your terminal:

```
    $ pip install surprisememore
```

If you already install the package and wish to upgrade it,
you can simply type from your terminal:

```
    $ pip install surprisememore --upgrade
```

Dependencies
------------

NEMtropy uses <code>numba</code> library. It is installed automatically with surprisememore.
If you use <code>python3.5</code> you may incur in an error, we suggest installing numba with the following command:

```
    $ pip install --prefer-binary numba
```

It avoids an error during the installation of <code>llvmlite</code> due to 
the absence of its wheel in <code>python3.5</code>.

Some Examples
--------------
As an example, we run community detection on zachary karate club network.

```
    import numpy as np
    import networkx as nx
    from surprisememore import UndirectedGraph

    from surprisememore import UndirectedGraph
    import networkx as nx
    
    G = nx.karate_club_graph()
    adj_kar = nx.to_numpy_array(G)
    graph = UndirectedGraph(adj_kar)
    
    graph.run_discrete_community_detection(weighted=False,
                                           num_sim=2)
```
The algorithm will find the best partition by optimizing surprise score
function. At the end of the optimization process, the optimal partition is
saved as an attribute of the graph class.

```
    # optimal partition
    graph.solution
    
    # Surprise of the optimal partition
    graph.surprise
    
    # Log surprise
    graph.log_surprise
```

Similarly, we can run the algorithm detecting bimodular structure. In the case
of zachary karate club, the code snippet is the following.

#%% md

```
    from surprisememore import UndirectedGraph
    import networkx as nx
    
    G = nx.karate_club_graph()
    adj_kar = nx.to_numpy_array(G)
    graph = UndirectedGraph(adjacency=adj_kar)
```

Here we initialized our SupriseMeMore **UndirectedGraph** object with the adjacency
matrix. The available options are adjacency matrix or edgelist.

* If you use adjacency matrix, then you have to pass the matrix as a **numpy.ndarray**;

* If you use edgelist, then the edgelist has to be passed as a **list of tuple**:
    * [(u, v), (u, t), ...] for binary networks;
    * [(u, v, w1), (u, t, w2), ...] for weighted networks;

For more details about edgelist format you can see [link](https://networkx.org/documentation/stable/reference/classes/generated/networkx.DiGraph.add_weighted_edges_from.html?highlight=add_weighted_edges_from#networkx.DiGraph.add_weighted_edges_from).

```
    graph.run_discrete_cp_detection(weighted=False, num_sim=2)
```

In the previous example I passed two optional arguments to the function: *weighted*
and *num_sim*. The argument *weighted* specify which version of surprise you want 
to use: binary or weighted. If the network is binary, you don't need to pass 
the argument "weighted", the class detects by itself that the graph is binary 
and use the proper method for community/bimodular detection. Instead, if the 
network has weights, the default method is the weighted one. To run binary 
community/bimodular detection you must specify "weighted"=False.

The arguments *num_sim* specifies the number of time we run over all the edges 
of the network during the optimization problem. You can find more detail about the
algorithm in [1](https://arxiv.org/abs/2106.05055), [2](https://www.nature.com/articles/srep19250).

All the implemented algorithms are heuristic, we suggest running them more 
than once and pick the best solution (the one with higher log_surprise).

To learn more, please read the two ipython notebooks in the examples' directory:
one is a study case on a [community detection](https://github.com/nicoloval/NEMtropy/blob/master/examples/Community%20Detection.ipynb), 
while the other is on an [bimodular detection](https://github.com/nicoloval/NEMtropy/blob/master/examples/Mesoscale%20Structure%20Detection.ipynb).

Development
-----------
Please work on a feature branch and create a pull request to the development 
branch. If necessary to merge manually do so without fast-forward:

```
    $ git merge --no-ff myfeature
```

To build a development environment run:

```
    $ python3 -m venv venv 
    $ source venv/bin/activate 
    $ pip install -e '.[dev]'
```

Credits
-------

_Author_:

[Emiliano Marchese](https://www.imtlucca.it/en/emiliano.marchese/) (a.k.a. [EmilianoMarchese](https://github.com/EmilianoMarchese))


_Acknowledgements:_

The module was developed under the supervision of [Tiziano Squartini](http://www.imtlucca.it/en/tiziano.squartini/).
