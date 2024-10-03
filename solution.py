import numpy as np

iris = np.genfromtxt('iris.txt')

######## DO NOT MODIFY THIS FUNCTION ########
def draw_rand_label(x, label_list):
    seed = abs(np.sum(x))
    while seed < 1:
        seed = 10 * seed
    seed = int(1000000 * seed)
    np.random.seed(seed)
    return np.random.choice(label_list)
#############################################


class Q1:
    #Pour toutes les fonctions de Q1, on veut ignorer la dernière col
    #On veut faire un tab contenant la moyenne de chaque cols => tab 1 x 4 (pcq 4 cols)
    def feature_means(self, iris):
        tabMoyCol = np.mean(iris[:, :-1], axis=0)
        return tabMoyCol

    #On veut faire une matrice de la covariance des attr (col) => matrice 4 x 4
    def empirical_covariance(self, iris):
        tabCov = np.cov(iris[:, :-1], rowvar=False)
        return tabCov

    #On veut la moyenne des vals dont dernière col = 1
    def feature_means_class_1(self, iris):
        #On prend toutes les valeurs de la classe 1
        tmp = iris[iris[:, -1] == 1]
        tabMoyClass1 = np.mean(tmp[:, :-1], axis=0)
        return tabMoyClass1

    def empirical_covariance_class_1(self, iris):
        tmp = iris[iris[:, -1] == 1]
        tabCovClass1 = np.cov(tmp[:, :-1], rowvar=False)
        return tabCovClass1


class HardParzen:
    def __init__(self, h):
        self.h = h

    #On arrange les données pour def predict(self, test_data)
    def fit(self, train_inputs, train_labels):
        self.train_inputs = np.array(train_inputs)
        self.train_labels = np.array(train_labels).astype(int)
        #Param de la fonc draw_rand_label
        self.label_list = np.unique(train_labels)

    def predict(self, test_data):
        #array pour cumuler les prédictions
        tmp = []
        
        for pointTest in test_data:
            #1- On calc dist entre les pts d'entrainement et les pts de test
            distM = np.sum(np.abs(self.train_inputs - pointTest), axis=1)

            #2- On trouve les index associés aux voisins
            voisin = np.where(distM <= self.h)[0]
            
            #3- On fait la condition pour trouver le bon label
            #Si aucun voisin, on appel def draw_rand_label(x, label_list)
            if len(voisin) == 0:
                freq = draw_rand_label(pointTest, self.label_list)
            else:
                #Sinon, on va chercher les valeurs aux index voisin
                label = self.train_labels[voisin].astype(int)
                # et on prend celui qui revient le plus frequement
                freq = np.bincount(label).argmax()
            tmp.append(freq)
        
        #On change en numpy array
        predictions = np.array(tmp)
        return predictions


class SoftRBFParzen:
    def __init__(self, sigma):
        self.sigma  = sigma

    def fit(self, train_inputs, train_labels):
        self.train_inputs = np.array(train_inputs)
        self.train_labels = np.array(train_labels).astype(int)
        self.label_list = np.unique(train_labels)

    def predict(self, test_data):
        tmp = []
        
        for pointTest in test_data:
            #On calc la dist
            distM = np.sum(np.abs(self.train_inputs - pointTest), axis=1)
            #On calc le rbf
            rbf = np.exp( -distM**2 / (2*(self.sigma**2)) )
 
            #On fait un dictionnaire pour associer les labels à leur rbf
            dic = {label: np.sum(rbf[label == self.train_labels]) for label in self.label_list}
            
            #On sélectionne le label dans le dictionnaire dont le rbf est le plus grand
            tmp.append( max(dic, key=dic.get) )
            predictions = np.array(tmp)
        return predictions


def split_dataset(iris):
    #Le diviseur
    d = 5
    #Réponses possibles du modulo
    num = [0, 1, 2, 3, 4]
    #Conversion vers non numpy
    data = iris.tolist()
    #tmp pour stocker les résultats
    en012 =[]
    en3 =[]
    en4=[]

    #On passe à travers les données dans iris
    for elem in data:
        #On store l'index
        index = data.index(elem)
        #On compare l'index à tous les résultats possibles, puis on assignes les réponses aux bons endroits
        for i in num:
            if index % d == i:
                if i == 3:
                    en3.append(elem)
                if i == 4:
                    en4.append(elem)
                if i == 0 or i == 1 or i == 2:
                    en012.append(elem)
    
    #On converti en numpy array               
    ensembleEntrainement = np.array(en012)
    ensembleValidation = np.array(en3)
    ensembleTest = np.array(en4)
    return (ensembleEntrainement, ensembleValidation, ensembleTest)


class ErrorRate:
    #def __init__(Donnée entrainement sans derniere col, derniere col données entrainement, Données validation sans derniere col, derniere col données validation)
    def __init__(self, x_train, y_train, x_val, y_val):
        #x_train et x_val sont des matrices d'attributs à 4 colonnes
        #y_train et y_val sont des tableaux contenant les labels
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val

    def hard_parzen(self, h):
        #On veut calc Parzen avec les données d'entrainement
        objHp = HardParzen(h)
        #On entraine algo sur x_train et y_train et l'évalue sur x_val
        objHp.fit(self.x_train, self.y_train)
        tabV = objHp.predict(self.x_val)
        
        """
        #Tx Erreur = Nb prédictions incorrectes / nb tot prédictions
        #Nb prédictions incorrectes = Σ prédictions qui != val réelle
        #Donc, Tx Erreur = Σ prédictions qui != val réelle / nb tot prédictions
        #   Ce qui est équivalent à la moyenne des prédictions
        #prédictions = hardParzen sur x_train et y_train avec x_val comme prédicteur sur les données d'entrainement
        #val réelle = y_val
        """
        #On calc le tx d'erreur => moy des données qui != les données de validation
        errorRate = np.mean(tabV != self.y_val)
        return errorRate

    def soft_parzen(self, sigma):
        #Même logique que pour def hard_parzen(self, h)
        objHp = SoftRBFParzen(sigma)
        objHp.fit(self.x_train, self.y_train)
        tabV = objHp.predict(self.x_val)
        errorRate = np.mean(tabV != self.y_val)
        return errorRate

#TESTS erroRate
ens = split_dataset(iris)
e = ens[0]
v = ens[1]
objER = ErrorRate(e[:, :-1], e[:, -1], v[:, :-1], v[:, -1])
a = [objER.hard_parzen(h) for h in [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 3.0, 10.0, 20.0]]
b = [objER.soft_parzen(sigma) for sigma in [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 3.0, 10.0, 20.0]]
print(a)
print(b)

def get_test_errors(iris):
    # h* est celle parmis choix_h qui minimise l'erreur de Hard Parzen sur l'ensemble de validation
    choix_h = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 3.0, 10.0, 20.0]
    
    # σ* est le paramètre (parmis ceux proposés à la question 5) qui minimise l'erreur de Soft RBF Parzen sur l'ensemble de validation
    choix_sigma = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 3.0, 10.0, 20.0]
    
    #1- On va chercher les ensembles avec split_dataset
    ensembles = split_dataset(iris)
    ensembleEntrainement = ensembles[0]
    ensembleValidation = ensembles[1]
    ensembleTest = ensembles[2]
    
    #2- On veut trouver h*
    #On fait un objet pour trouver les erreurs sur l'ensemble de validation
    objE = ErrorRate(ensembleEntrainement[:, :-1], ensembleEntrainement[:, -1], ensembleValidation[:, :-1], ensembleValidation[:, -1])
    #On fait un dictionnaire pour associer toutes les erreurs avec leur h
    dicH = {h : objE.hard_parzen(h) for h in choix_h}
    #On trouve le h avec la plus petite valeur
    h_etoile = min(dicH, key=dicH.get)
    
    #3- On veut trouver sigma*
    #Même logique qu'à l'étape 2
    dicS = {sigma : objE.soft_parzen(sigma) for sigma in choix_sigma}
    sigma_etoile = min(dicS, key=dicS.get)
    
    #4- On veut le tx d'erreur sur l'ensemble de test
    objE_etoile = ErrorRate(ensembleEntrainement[:, :-1], ensembleEntrainement[:, -1], ensembleTest[:, :-1], ensembleTest[:, -1])
    txE_h_etoile = objE_etoile.hard_parzen(h_etoile)
    txE_sigma_etoile = objE_etoile.soft_parzen(sigma_etoile)
    
    #On récupère les résultats dans un tab
    res = np.array([txE_h_etoile, txE_sigma_etoile])
    
    return res

#Q8 bonus (???)
def random_projections(X, A):
    pass
