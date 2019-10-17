import re

def clean_string(s):
    s=s.lower()
    replacements = [
        [ [ "!","#",'&','%',"(",")","&quot;","+","\\",
            "'",'*','. ',', ',':',';','=','?','@','[',']',
            '^','_','`','{','|','}','~', '¡','¢','£','¤',
            '¥','¦','§','¨','©','«','¬', '®','¯', '´', 'ð',
            'µ','¶','·','¸','¹','»','¿','×','÷', '\x7f',
            '\x81','\x8d','\x90','\x9d','\xa0','\xad'], " "],
        [ ['à','á','â','ã','ä','å','ª'], "a"],
        [ ['è','é','ê','ë'], 'e'],
        [ ['ì','í','î','ï'], "i"],
        [ ['ò','ó','ô','õ','ö',], "o"],
        [ ['ù','ú','û','ü'], "u"],
        [ ['æ'], 'ae'],
        [ ['ß'], 'ss'],
        [ ['ç'], 'c'],
        [ [ '¼'], '1/4'],
        [ [ '½'], '1/2'],
        [ ['¾'], '3/4'],
        [['²'], '2'],
        [['³'], '3'],
        [['ñ'], 'n'],
        [['º'], '°'],
    ] 
    for r in replacements:
        for t in r[0]:
            s=s.replace(t,r[1])
    s = s.replace(".", " ")
    s = re.sub('\$\$+', '$',s)
    return s

def get_words(s,token_length = 0 ):
    if token_length!=0:
        return [w[:token_length] for w in re.split(" |,|\n|-|/",clean_string(s))]
    return re.split(" |,|\n|-|/",clean_string(s))

def get_nontrivial_words(s,token_length = 0 ):
    return [ w for w in get_words(s,token_length) if w!="" ]
###########################

def unique(l):
    return sorted(list(set(l)))
def flatten(l):
    return [item for sublist in l for item in sublist]

################################

from sklearn.metrics import confusion_matrix

def matriz_confusion(y_true,y_pred,categories):
    i=0
    number_cat ={}
    for c in categories:
        number_cat[c]=i
        i+=1
    N = confusion_matrix(y_true,y_pred,labels = categories)
    CM = ( N/N.sum(axis=1)[:,None], number_cat)
    return CM

def submatriz(CM, confusion):
    m = CM[0]
    number_cat = CM[1]
    indices = [number_cat[c] for c in confusion]
    return [[int(100*m[i][j])/100.0 for j in indices] for i in indices]


def categorias_para_mejorar(CM,categories, threshold=0.1):
    cats = [ c for c in categories if CM[0][CM[1][c]][CM[1][c]]<threshold ]
    print("Hay ",len(cats), "categorias para mejorar.")
    return cats

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

def confusiones_principales(CM,categories,threshold=0.05):
    m = CM[0]
    ady = [ [(lambda x : 1 if x>threshold else 0)(x) for x in row] for row in m]
    graph = csr_matrix(ady)
    n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)
    print("Hay ",n_components, " componentes.")
    componentes = [[] for i in range(n_components)]
    for i in range(len(categories)):
        componentes[labels[i]].append(categories[i])
    componentes_complicadas = [c for c in componentes if len(c)>1]
    print("Hay ",len(componentes_complicadas), " componentes complicadas")
    return componentes_complicadas
