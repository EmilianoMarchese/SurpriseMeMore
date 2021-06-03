Surprisemess
-------------------------------------------------------------------

SurpriseMeMore is a toolbox for detecting mesoscale structure in networks, released as a python3 module. 

SurpriseMeMore provides the user with a variety of solvers, based on the _surprise_ framework, for the detection of mesoscale structures ( e.g. communities, core-periphery) in networks.

The models implemented in SurpriseMeMore are presented in a forthcoming [paper](https://arxiv.org/) on arXiv.
If you use the module for your scientific research, please consider citing us:

```
    @misc{,
          title={}, 
          author=Emiliano Marchese and Tiziano Squartini},
          year={2021},
          eprint={},
          archivePrefix={},
          primaryClass={physics.data-an}
    }
```

#### Table Of Contents
- [Currently Implemented Methods](#currently-implemented-methods)
- [Installation](#installation)
- [Dependencies](#dependencies)
- [Some Examples](#some-examples)
- [Development](#development)
- [Testing](#testing)
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
As an example we run community detection on zachary karate club network.

```
    import numpy as np
    import networkx as nx
    from NEMtropy import UndirectedGraph

    G = nx.karate_club_graph()
    adj_kar = nx.to_numpy_array(G)
    graph = UndirectedGraph(adj_kar)

    graph.solve_tool(model="cm_exp",
                 method="newton",
                 initial_guess="random")
```

Given the UBCM model, we can generate ten random copies of zachary's karate club.

```
    graph.ensemble_sampler(10, cpu_n=2, output_dir="sample/")
```

These copies are saved as an edgelist, each edgelist can be converted to an
adjacency matrix by running the NEMtropy build graph function.

```
    from NEMtropy.network_functions import build_graph_from_edgelist

    edgelist_ens = np.loadtxt("sample/0.txt")
    ens_adj = build_graph_from_edgelist(edgelist = edgelist_ens,
                                    is_directed = False,
                                    is_sparse = False,
                                    is_weighted = False)
```

These collection of random adjacency matrices can be used as a null model:
it is enough to compute the expected value of the selected network feature 
on the ensemble of matrices and to compare it with its original value.

To learn more, please read the two ipython notebooks in the examples directory:
one is a study case on a [directed graph](https://github.com/nicoloval/NEMtropy/blob/master/examples/Directed%20Graphs.ipynb), 
while the other is on an [undirected graph](https://github.com/nicoloval/NEMtropy/blob/master/examples/Undirected%20Graphs.ipynb).

You can find complete documentation about NEMtropy library in [docs](https://nemtropy.readthedocs.io/en/master/index.html).

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

Testing
-------
If you want to test the package integrity, you can run the following 
bash command from the tests directory:

```
    $ bash run_all.sh
```

__P.S.__ _at the moment there may be some problems with the DECM solver functions_

Credits
-------

_Author_:

[Emiliano Marchese](https://www.imtlucca.it/en/emiliano.marchese/) (a.k.a. [EmilianoMarchese](https://github.com/EmilianoMarchese))


_Acknowledgements:_

The module was developed under the supervision of [Tiziano Squartini](http://www.imtlucca.it/en/tiziano.squartini/).
