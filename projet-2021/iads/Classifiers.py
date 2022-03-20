# -*- coding: utf-8 -*-

"""
Package: iads
File: Classifiers.py
Année: LU3IN026 - semestre 2 - 2020-2021, Sorbonne Université
"""

# Classfieurs implémentés en LU3IN026
# Version de départ : Février 2021

# Import de packages externes
import graphviz as gv
import math
import numpy as np
import pandas as pd
import random
import sys
# ---------------------------
class Classifier:
    """ Classe pour représenter un classifieur
        Attention: cette classe est une classe abstraite, elle ne peut pas être
        instanciée.
    """

    def __init__(self, input_dimension):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension de la description des exemples
            Hypothèse : input_dimension > 0
        """
        raise NotImplementedError("Please Implement this method")

    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        raise NotImplementedError("Please Implement this method")

    def score(self, x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        raise NotImplementedError("Please Implement this method")

    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        raise NotImplementedError("Please Implement this method")

    def accuracy(self, desc_set, label_set):
        """ Permet de calculer la qualité du système sur un dataset donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        # COMPLETER CETTE FONCTION ICI :
        nb_ok = 0.
        for i in range(label_set.size):
            if self.predict(desc_set[i]) == label_set[i]:
                nb_ok += 1
        return (nb_ok / label_set.size) * 100

    def getW(self):
        """ rend le vecteur de poids actuel du perceptron
        """
        return self.w


# ---------------------------
class ClassifierLineaireRandom(Classifier):
    """ Classe pour représenter un classifieur linéaire aléatoire
        Cette classe hérite de la classe Classifier
    """

    def __init__(self, input_dimension):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension de la description des exemples
            Hypothèse : input_dimension > 0
        """
        self.w = np.asarray([np.random.randn() for i in range(0, input_dimension)])
        # raise NotImplementedError("Please Implement this method")

    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        print("Random -> no training")
        # raise NotImplementedError("Please Implement this method")

    def score(self, x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        prediction = 0
        for i in range(0, x.size):
            prediction += np.vdot(self.w[i], x[i])
        return prediction


    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        prediction = self.score(x)
        if (prediction < 0):
            return -1
        else:
            return 1



# ---------------------------

class ClassifierKNN(Classifier):
    """ Classe pour représenter un classifieur par K plus proches voisins.
        Cette classe hérite de la classe Classifier
    """

    # TODO: Classe à Compléter
    def __init__(self, input_dimension, k):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
                - k (int) : nombre de voisins à considérer
            Hypothèse : input_dimension > 0
        """
        self.dim = input_dimension
        self.k = k
        self.desc_set = []
        self.label_set = []

    def score(self, x):
        """ rend la proportion de +1 parmi les k ppv de x (valeur réelle)
            x: une description : un ndarray
        """
        # euclidienne : sqrt( somme(xi - x)**2 )
        distance = []

        for xi in self.desc_set:
            temps = 0
            for i in range(len(xi)):
                temps += (xi[i] - x[i]) ** 2
            distance.append(np.sqrt(temps))

        liste_index = np.argsort(distance)

        nombre_de_1 = 0
        for i in range(self.k):
            index = liste_index[i]
            if self.label_set[index] == 1:
                nombre_de_1 += 1

        return nombre_de_1 / self.k

    def predict(self, x):
        """ rend la prediction sur x (-1 ou +1)
            x: une description : un ndarray
        """
        if self.score(x) >= 0.5:
            return 1
        return -1

    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        self.desc_set = desc_set
        self.label_set = label_set

 # ---------------------------
class ClassifierPerceptron(Classifier):
    """ Perceptron de Rosenblatt
    """

    def __init__(self, input_dimension, learning_rate, history=False):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples
                - learning_rate : epsilon
            Hypothèse : input_dimension > 0
        """
        # raise NotImplementedError("Please Implement this method")
        self.epsilon = learning_rate
        # self.w=np.random.uniform(-1,1,(1,input_dimension))
        self.w = np.zeros(input_dimension)
        self.history = history
        self.allw = []

    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            réalise une itération sur l'ensemble des données prises aléatoirement
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        # raise NotImplementedError("Please Implement this method")
        taille = np.array([i for i in range(len(desc_set))])
        np.random.shuffle(taille)
        for i in taille:
            if self.score(desc_set[i]) * label_set[i] < 1:
                self.w += self.epsilon * desc_set[i] * label_set[i]
                if self.history:
                    self.allw.append((self.w[0], self.w[1]))

    def score(self, x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        # raise NotImplementedError("Please Implement this method")
        return np.vdot(x, self.w)

    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        # raise NotImplementedError("Please Implement this method")
        res = self.score(x)
        if res < 0:
            return -1
        else:
            return +1
 # ---------------------------
class ClassifierPerceptronKernel(Classifier):
    def __init__(self, input_dimension, learning_rate, noyau):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples (espace originel)
                - learning_rate :
                - noyau : Kernel à utiliser
            Hypothèse : input_dimension > 0
        """
        self.noyau = noyau
        self.input_dimension = input_dimension
        self.learning_rate = learning_rate
        self.w = [0 for _ in range(noyau.get_output_dim())]

    def score(self, x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        x = self.noyau.transform(np.array([x]))[0]
        res = 0
        for i in range(len(self.w)):
            res += x[i] * self.w[i]
        return res

    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        if (self.score(x) < 0):
            return -1
        return 1

    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            réalise une itération sur l'ensemble des données prises aléatoirement
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        desc_set = self.noyau.transform(desc_set)
        sur_place = 0
        n = len(desc_set)
        cpt = 0
        while (sur_place < n and cpt < 10 * n):
            i = random.randrange(n)
            xi = desc_set[i]
            yi = label_set[i]
            if (self.score(xi) * yi > 0):
                sur_place += 1
            else:
                sur_place = 0
                for k in range(len(self.w)):
                    self.w[k] += self.learning_rate * xi[k] * yi
            cpt += 1

    def copy(self):
        return ClassifierPerceptronKernel(self.input_dimension, self.learning_rate, self.noyau)
# ------------------------


import copy







class ClassifierADALINE(Classifier):
    """ Perceptron de ADALINE
    """

    # TODO: Classe Ã  ComplÃ©ter
    def __init__(self, input_dimension, history=False, niter_max=1000):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples
                - learning_rate : epsilon
                - history : stockage des poids w en cours d'apprentissage
                - niter_max : borne sur les iterations
            HypothÃ¨se : input_dimension > 0
        """
        self.w = np.zeros(2)
        self.history = history
        self.niter_max = niter_max
        self.allw = []
        self.allw.append(np.copy(self.w))

    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donnÃ©
            rÃ©alise une itÃ©ration sur l'ensemble des donnÃ©es prises alÃ©atoirement
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            HypothÃ¨se: desc_set et label_set ont le mÃªme nombre de lignes
        """
        self.w = np.linalg.solve(np.transpose(desc_set) @ desc_set, np.transpose(desc_set) @ label_set)

    def score(self, x):
        """ rend le score de prÃ©diction sur x (valeur rÃ©elle)
            x: une description
        """
        return x @ self.w

    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        if self.score(x) > 0:
            return 1
        return -1





class ClassifierMultiOAA():
    def __init__(self, classifier):
        self.c = copy.deepcopy(classifier)
        self.classifiers = []
        self.ind_label = dict()  # Dictionnaire {Classe : indice associé dans la liste du classifier}

    def train(self, data_set, label_set):
        # Tout d'abord on crée nos nCl classifiers et on leur assigne un indice
        i = 0
        for l in label_set:
            if l not in self.ind_label:
                self.ind_label[l] = i
                i += 1
                self.classifiers.append(copy.deepcopy(self.c))

        # Pour chaque classe, on transforme le label_set en 1 et -1 et on entraine le classifier
        for classe in self.ind_label:
            ytmp = [1 if k == classe else -1 for k in label_set]
            self.classifiers[self.ind_label[classe]].train(data_set, ytmp)

    def score(self, x):
        res = []
        for c in self.classifiers:
            res.append(c.score(x))
        return res

    def predict(self, x):
        ind = np.argsort(self.score(x))[-1]
        for k in self.ind_label:
            if self.ind_label[k] == ind:
                return k

    def accuracy(self, desc_set, label_set):
        yhat = np.array([self.predict(x) for x in desc_set])
        return np.where(label_set == yhat, 1., 0.).mean()


# code de la classe ADALINE Analytique

class ClassifierADALINE2(Classifier):
    """ Perceptron de ADALINE
    """

    # TODO: Classe à Compléter
    def __init__(self, input_dimension, history=False, niter_max=1000):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples
                - learning_rate : epsilon
                - history : stockage des poids w en cours d'apprentissage
                - niter_max : borne sur les iterations
            Hypothèse : input_dimension > 0
        """
        self.w = np.zeros(2)
        self.history = history
        self.niter_max = niter_max
        self.allw = []
        self.allw.append(np.copy(self.w))

    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            réalise une itération sur l'ensemble des données prises aléatoirement
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        self.w = np.linalg.solve(np.transpose(desc_set) @ desc_set, np.transpose(desc_set) @ label_set)

    def score(self, x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        return x @ self.w

    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        if self.score(x) > 0:
            return 1
        return -1






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


def discretise(desc, labels, col):
    """ array * array * int -> tuple[float, float]
        Hypothèse: les 2 arrays sont de même taille et contiennent au moins 2 éléments
        col est le numéro de colonne à discrétiser
        rend la valeur de coupure qui minimise l'entropie ainsi que son entropie.
    """
    # initialisation: (import sys doit avoir été fait)
    min_entropie = sys.float_info.max  # on met à une valeur max car on veut minimiser
    min_seuil = 0.0

    # trie des valeurs: ind contient les indices dans l'ordre croissant des valeurs pour chaque colonne
    ind = np.argsort(desc, axis=0)

    #  pour voir ce qui se passe, on va sauver les entropies trouvées et les points de coupures:
    liste_entropies = []
    liste_coupures = []

    # Dictionnaire pour compter les valeurs de classes qui restera à voir
    Avenir_nb_class = dict()
    #  et son initialisation:
    for j in range(0, len(desc)):
        if labels[j] in Avenir_nb_class:
            Avenir_nb_class[labels[j]] += 1
        else:
            Avenir_nb_class[labels[j]] = 1

    # Dictionnaire pour compter les valeurs de classes que l'on a déjà vues
    Vues_nb_class = dict()

    # Nombre total d'exemples à traiter:
    nb_total = 0
    for c in Avenir_nb_class:
        nb_total += Avenir_nb_class[c]

    # parcours pour trouver le meilleur seuil:
    for i in range(len(desc) - 1):
        v_ind_i = ind[i]  #  vecteur d'indices de la valeur courante à traiter
        courant = desc[v_ind_i[col]][col]  #  valeur courante de la colonne
        lookahead = desc[ind[i + 1][col]][col]  #  valeur suivante de la valeur courante
        val_seuil = (courant + lookahead) / 2.0;  #  Seuil de coupure: entre les 2 valeurs

        # M-A-J de la distrib. des classes:
        # pour réduire les traitements: on retire un exemple de E2 et on le place
        # dans E1, c'est ainsi que l'on déplace donc le seuil de coupure.
        if labels[v_ind_i[col]] in Vues_nb_class:
            Vues_nb_class[labels[v_ind_i[col]]] += 1

        else:
            Vues_nb_class[labels[v_ind_i[col]]] = 1
        #  on retire de l'avenir:
        Avenir_nb_class[labels[v_ind_i[col]]] -= 1

        #  construction de 2 listes: ordonnées sur les mêmes valeurs de classes
        #  contenant le nb d'éléments de chaque classe
        nb_inf = []
        nb_sup = []
        tot_inf = 0
        tot_sup = 0
        for (c, nb_c) in Avenir_nb_class.items():
            nb_sup.append(nb_c)
            tot_sup += nb_c
            if (c in Vues_nb_class):
                nb_inf.append(Vues_nb_class[c])
                tot_inf += Vues_nb_class[c]
            else:
                nb_inf.append(0)

        # calcul de la distribution des classes de chaque côté du seuil:
        freq_inf = [nb / float(tot_inf) for nb in nb_inf]
        freq_sup = [nb / float(tot_sup) for nb in nb_sup]
        # calcul de l'entropie de la coupure
        val_entropie_inf = shannon(freq_inf)
        val_entropie_sup = shannon(freq_sup)

        val_entropie = (tot_inf / float(tot_inf + tot_sup)) * val_entropie_inf \
                       + (tot_sup / float(tot_inf + tot_sup)) * val_entropie_sup
        # Ajout de la valeur trouvée pour l'historique:
        liste_entropies.append(val_entropie)
        liste_coupures.append(val_seuil)

        # si cette coupure minimise l'entropie, on mémorise ce seuil et son entropie:
        if (min_entropie > val_entropie):
            min_entropie = val_entropie
            min_seuil = val_seuil
    return (min_seuil, min_entropie), (liste_coupures, liste_entropies,)


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
        numeric = False
        #############
        # i_best : le numéro de l'attribut qui minimise l'entropie
        # min_entropie : la valeur de l'entropie minimale
        # Xbest_valeurs : la liste des valeurs que peut prendre l'attribut i_best
        #
        # Il est donc nécessaire ici de parcourir tous les attributs et de calculer
        # la valeur de l'entropie de la classe pour chaque attribut.
        ############

        for i in range(len(X[0, :])):  # parcours de tout les attributs
            if isinstance(X[0, i], float) or isinstance(X[0, i], int):
                resultat, liste_vals = discretise(X, Y, i)  # determine le meilleur seuil
                seuil = resultat[0]
                entropie_cond = resultat[1]

                if (entropie_ens - entropie_cond > entropie_ens - min_entropie):
                    min_entropie = entropie_cond
                    i_best = i
                    Xbest_valeurs = seuil
                    numeric = True

            else:
                attribut = X[:, i]  # Recupere la colonne de l'attribut
                diff_val_attribut, nb_fois = np.unique(attribut, return_counts=True)
                diff_val_Y = []

                for j in range(len(diff_val_attribut)):  # parcours des diff valeurs de l'attribut
                    p = nb_fois[j] / len(attribut)  # Calcul la proba de cette valeur dans l'attribut
                    diff_val_Y.append(
                        -p * entropie(Y[X[:, i] == diff_val_attribut[j]]))  # Calcul de -p(valeur)*H(attribut/valeur)

                entropie_cond = -sum(diff_val_Y)  # Calcul de la somme des entropies

                if (entropie_ens - entropie_cond > entropie_ens - min_entropie):
                    min_entropie = entropie_cond
                    i_best = i
                    Xbest_valeurs = diff_val_attribut
                    numeric = False
        if len(LNoms) > 0:  # si on a des noms de features
            noeud = NoeudCategoriel(i_best, LNoms[i_best])
        else:
            noeud = NoeudCategoriel(i_best)

        if numeric:
            if X[X[:, i_best] < Xbest_valeurs].size != 0:
                noeud.ajoute_fils("< " + str(Xbest_valeurs),
                                  construit_AD(X[X[:, i_best] < Xbest_valeurs], Y[X[:, i_best] < Xbest_valeurs],
                                               epsilon, LNoms))
            if X[X[:, i_best] > Xbest_valeurs].size != 0:
                noeud.ajoute_fils(">= " + str(Xbest_valeurs),
                                  construit_AD(X[X[:, i_best] > Xbest_valeurs], Y[X[:, i_best] > Xbest_valeurs],
                                               epsilon, LNoms))
        else:
            for v in Xbest_valeurs:
                noeud.ajoute_fils(v, construit_AD(X[X[:, i_best] == v], Y[X[:, i_best] == v], epsilon, LNoms))
    return noeud

class CAD(Classifier):
    """ Classe pour représenter un classifieur par arbre de décision
    """

    def __init__(self, input_dimension, epsilon, LNoms=[]):
        """ Constructeur
            Argument:
                - intput_dimension (int) : dimension de la description des exemples
                - epsilon (float) : paramètre de l'algorithme (cf. explications précédentes)
                - LNoms : Liste des noms de dimensions (si connues)
            Hypothèse : input_dimension > 0
        """
        self.dimension = input_dimension
        self.epsilon = epsilon
        self.LNoms = LNoms
        # l'arbre est manipulé par sa racine qui sera un Noeud
        self.racine = None

    def toString(self):
        """  -> str
            rend le nom du classifieur avec ses paramètres
        """
        return 'ClassifierArbreDecision [' + str(self.dimension) + '] eps=' + str(self.epsilon)

    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        self.racine = construit_AD(desc_set, label_set, epsilon=self.epsilon, LNoms=self.LNoms)

    def score(self, x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        # cette méthode ne fait rien dans notre implémentation :
        pass

    def predict(self, x):
        """ x (array): une description d'exemple
            rend la prediction sur x
        """
        return self.racine.classifie(x)

    def affiche(self, GTree):
        """ affichage de l'arbre sous forme graphique
            Cette fonction modifie GTree par effet de bord
        """
        self.racine.to_graph(GTree)





