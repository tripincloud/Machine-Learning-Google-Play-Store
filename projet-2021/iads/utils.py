# -*- coding: utf-8 -*-

"""
Package: iads
Fichier: utils.py
Année: semestre 2 - 2019-2020, Sorbonne Université
"""

# ---------------------------
# -*- coding: utf-8 -*-

"""
Package: iads
File: utils.py
Année: LU3IN026 - semestre 2 - 2020-2021, Sorbonne Université
"""


# Fonctions utiles pour les TDTME de LU3IN026
# Version de départ : Février 2021

# import externe
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------ 
def plot2DSet(desc,labels):    
    """ ndarray * ndarray -> affichage
        la fonction doit utiliser la couleur 'red' pour la classe -1 et 'blue' pour la +1
    """
    data_negatifs = desc[labels == -1]
    data_positifs = desc[labels == +1]
    plt.scatter(data_negatifs[:, 0], data_negatifs[:, 1], marker='o')
    plt.scatter(data_positifs[:, 0], data_positifs[:, 1], marker='x')
    
def plot_frontiere(desc_set, label_set, classifier, step=30):
    """ desc_set * label_set * Classifier * int -> NoneType
        Remarque: le 4e argument est optionnel et donne la "résolution" du tracé
        affiche la frontière de décision associée au classifieur
    """
    mmax=desc_set.max(0)
    mmin=desc_set.min(0)
    x1grid,x2grid=np.meshgrid(np.linspace(mmin[0],mmax[0],step),np.linspace(mmin[1],mmax[1],step))
    grid=np.hstack((x1grid.reshape(x1grid.size,1),x2grid.reshape(x2grid.size,1)))
    
    # calcul de la prediction pour chaque point de la grille
    res=np.array([classifier.predict(grid[i,:]) for i in range(len(grid)) ])
    res=res.reshape(x1grid.shape)
    # tracer des frontieres
    # colors[0] est la couleur des -1 et colors[1] est la couleur des +1
    plt.contourf(x1grid,x2grid,res,colors=["darksalmon","skyblue"],levels=[-1000,0,1000])    
# ------------------------ 
def genere_dataset_uniform(p, n, binf=-1, bsup=1):
    """ int * int * float^2 -> tuple[ndarray, ndarray]
        Hyp: n est pair
        p: nombre de dimensions de la description
        n: nombre d'exemples
        les valeurs générées uniformément sont dans [binf,bsup]
    """
    a = n / 2
    desc = np.random.uniform(inf, sup, (n, p))
    lab = np.asarray([-1 for i in range(0, int(a))] + [+1 for i in range(0, int(a))])
    np.random.shuffle(lab)
    return (desc, lab)
    
def genere_dataset_gaussian(positive_center, positive_sigma, negative_center, negative_sigma, nb_points):
    """ les valeurs générées suivent une loi normale
        rend un tuple (data_desc, data_labels)
    """
    data_positif=np.random.multivariate_normal(positive_center,positive_sigma,nb_points)
    data_negatif=np.random.multivariate_normal(negative_center,negative_sigma,nb_points)
    data_desc=np.vstack((data_positif,data_negatif))
    data_label=np.asarray([-1 for i in range(0,nb_points)] + [+1 for i in range(0,nb_points)])
    #np.random.shuffle(data_label)
    return (data_desc,data_label)
# ------------------------ 
def create_XOR(n, var):
    x = np.zeros(2)
    a = genere_dataset_gaussian(np.array([0, 1]), np.array([[var, 0], [0, var]]), np.array([0, 0]),
                                np.array([[var, 0], [0, var]]), nb_points)
    b = genere_dataset_gaussian(np.array([1, 0]), np.array([[var, 0], [0, var]]), np.array([1, 1]),
                                np.array([[var, 0], [0, var]]), nb_points)

    d1, l1 = a
    d2, l2 = b

    x = np.vstack((d1, d2))
    y = np.concatenate((l1, l2), axis=None)
    l = []
    l.append(x)
    l.append(y)
    return l


def crossval(X, Y, n_iterations, iteration):
    index = np.random.permutation(len(X)) # mélange des index
    nb_elem = len(X) // n_iterations
    test = [index[i] for i in range(nb_elem)]
    app = [index[i] for i in range(nb_elem, len(X))]
    Xtest = X[test]
    Ytest = Y[test]
    Xapp = X[app]
    Yapp = Y[app]
    return Xapp, Yapp, Xtest, Ytest


def crossval_strat(X, Y, n_iterations, iteration):
    classes = np.unique(Y)
    Xc = [X[Y==c] for c in classes]
    Yc = [Y[Y==c] for c in classes]
    d = X.shape[1]
    Xapp, Yapp, Xtest, Ytest = np.array([]).reshape(0,d), np.array([]).reshape(0,d), np.array([]).reshape(0,d), np.array([]).reshape(0,d)
    for i in range(len(classes)):
        Xappc, Yappc, Xtestc, Ytestc = crossval(Xc[i], Yc[i], n_iterations, iteration)
        Xapp = np.vstack((Xapp, Xappc)) if Xapp.size else Xappc
        Yapp = np.concatenate((Yapp, Yappc)) if Yapp.size else Yappc
        Xtest = np.vstack((Xtest, Xtestc)) if Xtest.size else Xtestc
        Ytest = np.concatenate((Ytest, Ytestc)) if Ytest.size else Ytestc
    return Xapp, Yapp, Xtest, Ytest


def plot2DSetMulticlass(desc, labels):
    """ ndarray * ndarray -> affichage
        la fonction doit utiliser la couleur 'red' pour la classe -1 et 'blue' pour la +1
    """
    val = np.unique(labels)
    data = []
    for i in val:
        data.append(desc[labels == i])
    for i in range(len(data)):
        rgb = np.random.rand(3, )
        plt.scatter(data[i][:, 0], data[i][:, 1], marker='o', color=[rgb])


class Kernel():
    """ Classe pour représenter des fonctions noyau
    """

    def __init__(self, dim_in, dim_out):
        """ Constructeur de Kernel
            Argument:
                - dim_in : dimension de l'espace de départ (entrée du noyau)
                - dim_out: dimension de l'espace de d'arrivée (sortie du noyau)
        """
        self.input_dim = dim_in
        self.output_dim = dim_out

    def get_input_dim(self):
        """ rend la dimension de l'espace de départ
        """
        return self.input_dim

    def get_output_dim(self):
        """ rend la dimension de l'espace d'arrivée
        """
        return self.output_dim

    def transform(self, V):
        """ ndarray -> ndarray
            fonction pour transformer V dans le nouvel espace de représentation
        """
        raise NotImplementedError("Please Implement this method")


class KernelBias(Kernel):
    """ Classe pour un noyau simple 2D -> 3D
    """
    def transform(self, V):
        """ ndarray de dim 2 -> ndarray de dim 3
            rajoute une 3e dimension au vecteur donné
        """
        V_proj = np.append(V,np.ones((len(V),1)),axis=1)
        return V_proj

def classe_majoritaire(Y):
    """ Y : (array) : array de labels
        rend la classe majoritaire ()
    """
    valeurs, nb_fois = np.unique(Y,return_counts=True)
    ind = np.argsort(nb_fois)[len(nb_fois)-1]
    return valeurs[ind]


def shannon(P):
    """ list[Number] -> float
        Hypothèse: la somme des nombres de P vaut 1
        P correspond à une distribution de probabilité
        rend la valeur de l'entropie de Shannon correspondante
    """
    #### A compléter pour répondre à la question posée
    Hs = 0.0
    k = len(P)

    if k == 1:
        return Hs

    for pi in P:
        if pi > 0.0:
            Hs -= pi * (math.log(pi) / math.log(k))
    return Hs

def entropie(Y):
    """ Y : (array) : ensemble de labels de classe
        rend l'entropie de l'ensemble Y
    """
    value, nb_fois = np.unique(Y,return_counts=True)
    lst = []
    for i in nb_fois:
        lst.append(i/len(Y))

    return shannon(lst)


class NoeudCategoriel:
    """ Classe pour représenter des noeuds d'un arbre de décision
    """

    def __init__(self, num_att=-1, nom=''):
        """ Constructeur: il prend en argument
            - num_att (int) : le numéro de l'attribut auquel il se rapporte: de 0 à ...
              si le noeud se rapporte à la classe, le numéro est -1, on n'a pas besoin
              de le préciser
            - nom (str) : une chaîne de caractères donnant le nom de l'attribut si
              il est connu (sinon, on ne met rien et le nom sera donné de façon
              générique: "att_Numéro")
        """
        self.attribut = num_att  # numéro de l'attribut
        if (nom == ''):  # son nom si connu
            self.nom_attribut = 'att_' + str(num_att)
        else:
            self.nom_attribut = nom
        self.Les_fils = None  # aucun fils à la création, ils seront ajoutés
        self.classe = None  # valeur de la classe si c'est une feuille

    def est_feuille(self):
        """ rend True si l'arbre est une feuille
            c'est une feuille s'il n'a aucun fils
        """
        return self.Les_fils == None

    def ajoute_fils(self, valeur, Fils):
        """ valeur : valeur de l'attribut de ce noeud qui doit être associée à Fils
                     le type de cette valeur dépend de la base
            Fils (NoeudCategoriel) : un nouveau fils pour ce noeud
            Les fils sont stockés sous la forme d'un dictionnaire:
            Dictionnaire {valeur_attribut : NoeudCategoriel}
        """
        if self.Les_fils == None:
            self.Les_fils = dict()
        self.Les_fils[valeur] = Fils
        # Rem: attention, on ne fait aucun contrôle, la nouvelle association peut
        # écraser une association existante.

    def ajoute_feuille(self, classe):
        """ classe: valeur de la classe
            Ce noeud devient un noeud feuille
        """
        self.classe = classe
        self.Les_fils = None  # normalement, pas obligatoire ici, c'est pour être sûr

    def classifie(self, exemple):
        """ exemple : numpy.array
            rend la classe de l'exemple (pour nous, soit +1, soit -1 en général)
            on rend la valeur 0 si l'exemple ne peut pas être classé (cf. les questions
            posées en fin de ce notebook)
        """
        if self.est_feuille():
            return self.classe
        if exemple[self.attribut] in self.Les_fils:
            # descente récursive dans le noeud associé à la valeur de l'attribut
            # pour cet exemple:
            return self.Les_fils[exemple[self.attribut]].classifie(exemple)
        else:
            # Cas particulier : on ne trouve pas la valeur de l'exemple dans la liste des
            # fils du noeud... Voir la fin de ce notebook pour essayer de résoudre ce mystère...
            print('\t*** Warning: attribut ', self.nom_attribut, ' -> Valeur inconnue: ', exemple[self.attribut])
            return 0

    def to_graph(self, g, prefixe='A'):
        """ construit une représentation de l'arbre pour pouvoir l'afficher graphiquement
            Cette fonction ne nous intéressera pas plus que ça, elle ne sera donc pas expliquée
        """
        if self.est_feuille():
            g.node(prefixe, str(self.classe), shape='box')
        else:
            g.node(prefixe, self.nom_attribut)
            i = 0
            for (valeur, sous_arbre) in self.Les_fils.items():
                sous_arbre.to_graph(g, prefixe + str(i))
                g.edge(prefixe, prefixe + str(i), valeur)
                i = i + 1
        return g


def construit_AD(X, Y, epsilon, LNoms=[]):
    """ X,Y : dataset
        epsilon : seuil d'entropie pour le critère d'arrêt
        LNoms : liste des noms de features (colonnes) de description
    """

    entropie_ens = entropie(Y)

    if (entropie_ens <= epsilon):
        # ARRET : on crée une feuille
        noeud = NoeudCategoriel(-1, "Label")
        noeud.ajoute_feuille(classe_majoritaire(Y))
    else:
        min_entropie = 1.1
        i_best = -1
        Xbest_valeurs = None

        #############
        # COMPLETER CETTE PARTIE : ELLE DOIT PERMETTRE D'OBTENIR DANS
        # i_best : le numéro de l'attribut qui minimise l'entropie
        # min_entropie : la valeur de l'entropie minimale
        # Xbest_valeurs : la liste des valeurs que peut prendre l'attribut i_best
        #
        # Il est donc nécessaire ici de parcourir tous les attributs et de calculer
        # la valeur de l'entropie de la classe pour chaque attribut.
        ############

        for i in range(len(X[0])):  # parcours de tout les attributs
            attribut = X[:, i]  # Recupere la colonne de l'attribut
            diff_val_attribut, nb_fois = np.unique(attribut, return_counts=True)
            diff_val_Y = []

            for j in range(len(diff_val_attribut)):  # parcours des diff valeurs de l'attribut
                p = nb_fois[j] / len(attribut)  # Calcul la proba de cette valeur dans l'attribut
                diff_val_Y.append(
                    -p * entropie(Y[X[:, i] == diff_val_attribut[j]]))  # Calcul de -p(valeur)*H(attribut/valeur)

            entropie_cond = -sum(diff_val_Y)  # Calcul de la somme des entropies

            if (entropie_cond < min_entropie):
                min_entropie = entropie_cond
                i_best = i
                Xbest_valeurs = diff_val_attribut

        if len(LNoms) > 0:  # si on a des noms de features
            noeud = NoeudCategoriel(i_best, LNoms[i_best])
        else:
            noeud = NoeudCategoriel(i_best)
        for v in Xbest_valeurs:
            noeud.ajoute_fils(v, construit_AD(X[X[:, i_best] == v], Y[X[:, i_best] == v], epsilon, LNoms))
    return noeud




def normalisation(X):
    nom = X - X.min(axis=0)
    denom = X.max(axis=0) - X.min(axis=0)
    denom[denom==0] = 1
    return nom/denom

def dist_vect(a,b):
    return np.linalg.norm(a - b)

def centroide(data):
    return data.mean(axis=0)

def inertie_cluster(array):
    cent = centroide(array)
    inertie_ensembe = 0
    for i in array:
        inertie_i = dist_vect(cent,i)**2
        #print("centre: ",cent,"\tExemple: ",i,"\tDistance = ",inertie_i)
        inertie_ensembe += inertie_i
    return inertie_ensembe


import random


def initialisation(k, base):
    """hypothese : k > 1"""
    selec = random.sample(base.tolist(), k)
    print("Sélectionnés: ", selec)
    return np.asarray(selec)


def plus_proche(x,centroides):
    closest = centroides[0]
    i_closest = 0
    for i in range(1,len(centroides)):
        if(dist_vect(x,centroides[i])<dist_vect(x,centroides[i_closest])):
            closest = centroides[i]
            i_closest = i
    return i_closest

def affecte_cluster(base, k):
    dic = {i: [] for i in range(len(k))}
    for i in range(len(base)):
        pproche = plus_proche(base[i,:],k)
        dic[pproche].append(i)
    return dic

def nouveaux_centroides(base, affect):
    newCent = []
    for i in affect:
        mean = sum(base[affect[i]])/len(affect[i])
        newCent.append(mean)
    return np.asarray(newCent)

def inertie_globale(base, affect):
    i_global = 0
    for i in affect:
        i_global+=inertie_cluster(base[affect[i]])
    return i_global


def kmoyennes(k, base, epsilon, iter_max):
    cent = initialisation(k,base)
    for i in range(iter_max):
        affect = affecte_cluster(base, cent)
        old_inert = inertie_globale(base, affect)
        cent = nouveaux_centroides(base, affect)
        affect = affecte_cluster(base, cent)
        new_inert = inertie_globale(base, affect)
        print("iteration ",i,"Inertie :",old_inert,"Difference: ", abs(new_inert - old_inert))
        if (abs(new_inert - old_inert) < epsilon):
            break
    return cent, affect

def affiche_resultat(data, cent, affect):
    plt.scatter(cent[:,0],cent[:,1],color='green')
    for i in range(len(affect)):
        c = (random.random(),random.random(),random.random())
        plt.scatter(data[affect[i]][:,0],data[affect[i]][:,1],color=[c])

import re

def cleanhtml(sentence): #pour nettoyer le mot de toutes les balises html.
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', sentence)
    return cleantext

def cleanpunc(sentence): #pour nettoyer le mot de toute ponctuation ou de tout caractère spécial.
    cleaned = re.sub(r'[?|!|\'|"|#]',r'',sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]',r' ',cleaned)
    return cleaned

from sklearn.feature_extraction.text import TfidfVectorizer
def TFIDF(liste,seuil_min,seuil_max):
    vectorizer = TfidfVectorizer(min_df=seuil_min,max_df=seuil_max)
    vectors = vectorizer.fit_transform(liste)
    feature_names = vectorizer.get_feature_names()
    dense = vectors.todense()
    denselist = dense.tolist()
    return pd.DataFrame(denselist, columns=feature_names)




from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.manifold import MDS
from sklearn.cluster import KMeans
from sklearn.metrics import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics.pairwise import manhattan_distances

def reduction_dimension_pca(df,dim_out=2):
    """
    df : DataFrame
    return : ndarray
    """
    model = PCA(n_components=dim_out)
    return model.fit_transform(df)

def reduction_dimension_tsne(df,dim_out=2,perplexity_=30):
    """
    df : DataFrame
    return : ndarray
    """
    model = TSNE(n_components=dim_out,perplexity=perplexity_)
    return model.fit_transform(df)


def reduction_dimension_mds(df,dim_out=2):
    """
    df : DataFrame -> Matrice carrer des distances euclidiennes, cosaynes ...
    return : ndarray
    """
    seed = np.random.RandomState(seed=3)
    mds = MDS(n_components=dim_out, max_iter=3000, eps=1e-9, random_state=seed, dissimilarity="precomputed", n_jobs=1)
    return mds.fit(df).embedding_

def distance_euclidienne(df):
    return euclidean_distances(df)

def distance_cosine(df):
    return cosine_distances(df)

def distance_manhattan(df):
    return manhattan_distances(df)

def similariter_cosayne(df):
    return cosine_similarity(df)

import seaborn as sns
def afficheHeatmap(matrice_distance,liste_nom):
    """
    matrice_distance -> ndarray : matrice carrer des distances des series
    """
    df = pd.DataFrame(matrice_distance)
    df["apps"] = liste_nom
    df.set_index("apps",inplace=True)
    df.columns = liste_nom
    plt.figure(figsize = (16,12))
    sns.heatmap(df, annot=True,cmap="Blues")

def length(dict):
    s=0
    for i in range(len(dict)):
        s+=len(dict[i])
    return s

import pandas as pd

def interpret_kmeans(affectation,data,words):
    tab=np.full((length(affectation), 1), np.inf)
    for i in range(len(affectation)):
        for j in affectation[i]:
            tab[j]=i
    X_clustered = data
    X_clustered["cluster"] = tab
    #print(X_clustered)
    pd.plotting.parallel_coordinates(X_clustered,class_column="cluster", cols=words)
