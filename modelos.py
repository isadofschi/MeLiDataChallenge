from funciones import unique, flatten, get_words
import operator
import itertools
from tqdm.auto import tqdm
from collections import defaultdict

def constante_1(c): # pickleable
    return 1

class Modelo():
    def predict_title(self,title):
        scores = self.predict_proba(title)
        if len(scores)>0:
            return max(scores.items(), key=operator.itemgetter(1))[0]
        else:
            return self.random_guess
    def predict(self,df):
        tqdm.pandas()
        return df[self.text_column].progress_apply(self.predict_title)

class WordMagic(Modelo):
    def __init__(self, n = 1, lambda_unreliable = 1, word_getter = get_words, class_weights = constante_1, verbose = True, normalizar_predict_proba = True, text_column = "title", category_column = "category", random_guess = None, sacar_repetidos=True ):
        self.n = n
        self.lambda_unreliable = lambda_unreliable
        self.word_getter = word_getter
        self.class_weights = class_weights
        self.verbose=verbose
        self.normalizar_predict_proba = normalizar_predict_proba
        self.text_column = text_column
        self.category_column = category_column
        self.random_guess = random_guess
        self.sacar_repetidos = sacar_repetidos
    def fit(self,datos, row_weights = None):
        self.row_weights = row_weights
        self.cantidad_apariciones = defaultdict(float)
        self.word_code = {}
        self.M = defaultdict(lambda: defaultdict(float))
        if self.verbose:
            print("Extrayendo palabras de los títulos:",flush=True)
        tqdm.pandas()
        list_title_train = list(datos[self.text_column].progress_apply(self.word_getter))
        list_category_train = list(datos[self.category_column])
        self.words = unique(flatten(list_title_train))
        i = 0
        for w in self.words:
            self.word_code[w] = i
            i+=1
        if self.verbose:
            print("Entrenando modelo en ", self.n , "-uplas...", flush=True)
        for i in  tqdm(range(len(list_title_train))):
            c = list_category_train[i]
            weight = self.row_weights[i] if not (self.row_weights is None) else self.class_weights(c)
            if self.sacar_repetidos:
                words = unique([self.word_code[w] for w in list_title_train[i]])
            else:
                words = sorted([self.word_code[w] for w in list_title_train[i]])
            for tupla in itertools.combinations(words,self.n):             
                self.cantidad_apariciones[tupla] += weight
                self.M[tupla][c] += weight
        if self.verbose:
            print("Normalizando...", flush=True)
        for tupla in tqdm(self.M):
            peso_total = self.cantidad_apariciones[tupla]
            for c in self.M[tupla]:
                self.M[tupla][c]/=peso_total
        if self.verbose:
            print("Entrenamiento completo.\n", flush=True)
        del list_title_train,list_category_train
    def predict_proba(self,title):
        words_title = self.word_getter(title)
        scores=defaultdict(float)
        if self.sacar_repetidos:
            words = unique([self.word_code[w] for w in words_title if (w in self.word_code)])
        else:
            words = sorted([self.word_code[w] for w in words_title if (w in self.word_code)])
        for tupla in itertools.combinations(words,self.n):               
            if tupla in self.M: # mejor no sacar este if
                for c in self.M[tupla]:
                    scores[c] += self.M[tupla][c]
        if self.normalizar_predict_proba:
            suma = sum([scores[c] for c in scores])
            for c in scores:
                scores[c] = scores[c]/suma
        return scores

class EnsambleSuma(Modelo):
    "para promediar modelos"
    def __init__(self, modelos, pesos=None, random_guess = None, text_column = "title", category_column = "category"):
        self.modelos = modelos
        self.random_guess = random_guess
        self.text_column = text_column
        self.category_column = category_column
        if pesos is None:
            self.pesos = [1 for m in modelos]
        else:
            assert(len(modelos)==len(pesos))
            self.pesos = pesos
    def predict_proba(self,title):
        d = defaultdict(float)
        for i in range(len(self.modelos)):
            d1 = self.modelos[i].predict_proba(title)
            for c in d1:
                d[c]+=d1[c]*self.pesos[i]
        return d
