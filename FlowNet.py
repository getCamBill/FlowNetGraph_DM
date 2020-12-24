import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms.flow.edmondskarp import edmonds_karp
from pathlib import Path
import time

# --------------------- DEBUT --------------------------------------------------------            
class FlowNet:
    """
    La classe FlowNet permet de créer et manipuler des graphes orientées avec 
    des flots et des capacités.
    """

    def __init__(self, graph):  
        """
        contructeur de la classe FlowNet qui créer le graph en fonction du paramètre donné
        Args:
            graph (string, Graph): soit un nom de fichier soit un Graph de la bilbiothèque NetworkX
        """
        self.reset()
        if isinstance(graph, nx.classes.graph.Graph):
            self.build_graph(graph)
        elif isinstance(graph, str):
            self.read_csv(graph)
        else:
            print("Le type de graphe fourni n'est pas reconnu\nnx.classes.graph.Graph OU string (nom de fichier)")
        
        self.R = None
        self.s = 's' # source
        self.t = 't' # puits
    
    def reset(self):
        """
        Met à None les graph pour une nouvelle utilisation
        """
        self.R = None
        self.G = None
    
    def build_graph(self, graph):
        """
        créer un réseau à partir d’un graphe orienté NetworkX 
        dans lequel les capacités sont des attributs capacity des arcs
        Args:
            G (Graph): graph passé au constructeur
        """
        self.G = graph
        
    def read_csv(self, path):
        """
        Créer un réseau à partir de sa description dans un fichier CSV
        Args:
            path (string): chemin du fichier
        """
        p = Path(path)
        data = open(p, "r")
        print("file loaded : ", p)
        graphtype = nx.DiGraph()
        self.G = nx.parse_edgelist(data, delimiter=';', create_using=graphtype,
                              nodetype=str, data=(('capacity', int),))
        
    def export(self, name):
        """
        méthode qui permet l'export du graphe actuel en image au format png
        Args:
            name (string): nom du fichier
        """
        self.maj_flow_graph_g();
        pos = nx.random_layout(self.G)
        labels={ e : '{}|{}'.format(self.G[e[0]][e[1]]['flow'],self.G[e[0]][e[1]]['capacity']) for e in self.G.edges}
        node_colors=[ 'lightgrey' for _ in self.G.nodes() ]
        nx.draw_networkx(self.G, pos=pos, node_color=node_colors)
        nx.draw_networkx_edge_labels(self.G, pos=pos, edge_labels=labels)
        plt.savefig(name, format="PNG")
    
    def maj_flow_graph_g(self):
        """
        Comme nous ne travaillons que sur le graphe residuel 
        nous ne mettons pas à jours ses flots lors des updates.
        La méthode est donc appelées avant l'export en PNG.
        """
        for e in self.G.edges:
            self.G[e[0]][e[1]]['flow'] = self.R[e[0]][e[1]]['flow']
            self.G[e[0]][e[1]]['capacity'] = self.R[e[0]][e[1]]['capacity']
        
    def show(self):
        """
        Affiche le graphe actuel
        """
        print("\n")
        for e in self.G.edges():
            print(e[0], "--->", e[1], " ", self.R[e[0]][e[1]]['flow'], "/",self.R[e[0]][e[1]]['capacity'])
      
    def capacity(self, s1, s2):
        """
        capacité d'un arc est un entier positif
        Args:
            s1 (char): sommet u 
            s2 (char): sommet v

        Returns:
            int: capacité
        """
        return self.G[s1][s2]["capacity"]

    def get_flow(self):
        """
        Renvoie le flot maximal
        Returns:
            [type]: [description]
        """
        return self.flow_value
    
    def update(self, s1, s2, c):
        """
        Après avoir calculé le flot max, on mets à jour la capacité de l'arc
        selon deux méthode différente en fonction d'une augmentation ou une 
        diminution de ce dernier.
        Args:
            s1 (char): sommet u 
            s2 (char): sommet v
            c (int): capacité
        """
        old = self.G[s1][s2]['capacity']
        self.G[s1][s2]['capacity'] = c
        self.R[s1][s2]['capacity'] = c
        
        # self.compute_max_flow() # pour verifier avec la méthode de base
        
        if (c <= old or old == 0): # si diminution de la capacité d'un arc
            diff = old - c
            self.worth_it(self.R, diff, '-', s1, s2)
        else: # si augmentation de la capacité d'un arc
            diff = c - old
            self.worth_it(self.R, diff, '+', s1, s2)
            
    def worth_it(self, R, diff, sign, s1, s2):
        """
        On ne recalcul le flot max seulement si l'arc passé en paramètre saturé
        sinon il n'y a rien à faire à part mettre à jour la capacité de l'arc

        Args:
            R (Graph): graphe résiduel
            diff (int): écart entre la capacité initiale et la nouvelle
            sign (char): permet de distinguer si augmentation ou diminution
            s1 (char): sommet u 
            s2 (char): sommet v
        """
        if sign == '-':
            # on fait le recalcul du flow en k itération avec k = nouvelle capacité
            for k in range(diff): 
                self.compute_max_flow_decrease(R, 1, s1, s2)
        if sign == '+':
            # si l'arc est saturé alors on recalcul le flot max
            if (R[s1][s2]['flow'] == R[s1][s2]['capacity']-diff): 
                self.compute_max_flow_increase(R)

    def compute_max_flow_decrease(self, R, diff, u, v):
        """
        Recalcul le flow max apres une diminution de la capacité
        Args:
            R (Graph): graphe résiduel
            diff (int): écart entre la capacité initiale et la nouvelle
            u (char): sommet u 
            v (char): sommet v
        """
        
        s = self.s
        t = self.t
        r_pred = R.pred
        r_succ = R.succ
        # on cherche un chemin simple de u à s 
        _, pred1, _ = bidirectional_bfs(u, s, r_pred, r_succ)
        # on cherche un chemin simple de t à v
        _, pred2, _ = bidirectional_bfs(t, v, r_pred, r_succ)
        
        if (R[u][v]['capacity'] < R[u][v]['flow']): # si l'arc changé est saturé
            R[u][v]['flow'] -= diff
            R[v][u]['flow'] += diff
        
            for arc in pred1:
                if (pred1[arc]): # si pas None
                    R[arc][pred1[arc]]["flow"] -= diff
                    R[pred1[arc]][arc]["flow"] += diff
            for arc in pred2:
                if (pred2[arc]): # si pas None
                    R[arc][pred2[arc]]["flow"] -= diff
                    R[pred2[arc]][arc]["flow"] += diff
                    
            self.flow_value -= diff
            self.compute_max_flow_increase(R)

    def compute_max_flow_increase(self, R):
        """[summary]
        Recalcul le flow max apres une augmentation de la capacité
        *Une partie du code ci-dessous provient de NetworkX
        Args:
            R (Graph): graphe résiduel
        """
        flow_val = self.get_flow()
        s = self.s
        t = self.t
        cutoff = float("inf")
        r_pred = R.pred
        r_succ = R.succ
        inf = R.graph["inf"]
        while flow_val < cutoff:
            v, pred, succ = bidirectional_bfs(s, t, r_pred, r_succ)
            if pred is None:
                break
            path = [v]
            u = v
            while u != s:
                u = pred[u]
                path.append(u)
            path.reverse()
            u = v
            while u != t:
                u = succ[u]
                path.append(u)
            flow_val += augment(path, r_succ, inf)

        self.flow_value = flow_val
        
    def compute_max_flow(self):
        """
        Calcul le folt max et créer le graph résiduel
        """
        self.R = edmonds_karp(self.G, "s", "t")
        self.flow_value = self.R.graph["flow_value"]
    
    def benchmark(self, s1, s2):
        """Ici nous ne faisons varier seulement la valeurs d'un seul arc
            mais cela montre quand même la différence de rapiditié entre
            la méthode base et "fines"
        Args:
            s1 (char)): sommet u
            s2 (char): sommet v
        """
        param = [(1,10000,1, "augmentation"), (10000, 1, -1, "diminution")]
        # compute_max_flow
        print("\nApproche avec mise a zero du residuel\nMoyenne de 10 iterations pour 10 000 update\n")
        for p in param:
            n = 10
            t = 0.0000000000000000
            start=p[0];end=p[1];step=p[2];way=p[3];
            for _ in range(1,n-1):
                st = time.time() 
                for i in range(start, end, step):  
                    self.G[s1][s2]['capacity'] = i
                    self.R[s1][s2]['capacity'] = i
                    self.compute_max_flow()
                fin = time.time()
                t += (fin - st)
            print(way, " temps = ",t/n)
        # update
        print("\nApproche avec utilisation du residuel\nMoyenne de 10 iterations pour 10 000 update\n")
        for p in param:
            n = 10
            t = 0.0000000000000000
            start=p[0];end=p[1];step=p[2];way=p[3];
            for _ in range(1,n-1):
                st = time.time() 
                for i in range(start, end, step):  
                    self.update(s1, s2, i)
                fin = time.time()
                t += (fin - st)
            print(way, " temps = ",t/n)
            
                
                
# --------------------- FIN --------------------------------------------------------

""" ****** FROM NETWORKX ********
 https://networkx.org/documentation/stable/_modules/networkx/algorithms/flow/edmondskarp.html#edmonds_karp
 légérement modifié 
"""
def augment(path, R_succ, inf):
    """Augment flow along a path from s to t.
    """
    # Determine the path residual capacity.
    flow = inf
    it = iter(path)
    u = next(it)
    for v in it:
        attr = R_succ[u][v]
        flow = min(flow, attr["capacity"] - attr["flow"])
        u = v
    if flow * 2 > inf:
        raise nx.NetworkXUnbounded("Infinite capacity path, flow unbounded above.")
    # Augment flow along the path.
    it = iter(path)
    u = next(it)
    for v in it:
        R_succ[u][v]["flow"] += flow
        R_succ[v][u]["flow"] -= flow
        u = v
    return flow
def bidirectional_bfs(s, t, R_pred, R_succ):
    """Bidirectional breadth-first search for an augmenting path.
    """
    pred = {s: None}
    q_s = [s]
    succ = {t: None}
    q_t = [t]
    while True:
        q = []
        if len(q_s) <= len(q_t):
            for u in q_s:
                for v, attr in R_succ[u].items():
                    if v not in pred and attr["flow"] < attr["capacity"]:
                        pred[v] = u
                        if v in succ:
                            return v, pred, succ
                        q.append(v)
            if not q:
                return None, None, None
            q_s = q
        else:
            for u in q_t:
                for v, attr in R_pred[u].items():
                    if v not in succ and attr["flow"] < attr["capacity"]:
                        succ[v] = u
                        if v in pred:
                            return v, pred, succ
                        q.append(v)
            if not q:
                return None, None, None
            q_t = q
"""
 ******** FROM NETWORKX ********   
"""
