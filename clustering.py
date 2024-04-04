from sklearn.mixture import GaussianMixture
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
import warnings
from sklearn.decomposition import PCA

VOWELS = {"a", "e", "i", "o", "u"}

def encode_name(name):
    res = 0
    if "'" in name:
        res += 100
    for char in name:
        if char in VOWELS:
            res += 1
        else:
            res += 10
    return res


'''
lepší zobrazení při printech pandas, nezobrazuje FutureWarning a vypisuje
dataset na celý řádek
'''
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.set_option('display.width', 320)
pd.set_option('display.max_columns', None)

# autorské úpravy datasetu
champions = pd.read_csv("champions.csv").drop(columns=["Unnamed: 0"])
champions["Name_Encoded"] = champions["Name"].apply(encode_name)
'''
Z datasetu první odstraním sloupec Name - je zbytečný, jméno je přepočítané
na číslo, takhle se nebude muset pokaždé používat .drop(columns=["Name"])

Déle odstraním duplicitní řádky - duplicitní hodnoty jsou zbytečné, jen by 
mohly trochu vychýlit algoritmy, odstraňuju tedy všechny řádky se stejným ID,
i kdyby měli jiné hodnoty - beru ID jako rohodující o duplicitě, mohlo by se
stát, že jedna z hodnot byla chybně zadaná. V datasetu zůstane první výskyt
danné ID
'''
champions_dropped = champions.drop_duplicates(subset='ID', keep='first'). \
    drop(columns=["Name"]).reset_index(drop=True)
'''
Pomocí KNNInputeru jsem dopočítal NaN hodnoty, ty je třeba dopočítat, protože
si myslím, že jejich absence by mohla ovlivnit výsledek a protože se většině
modelů NaN hodnota nelíbí a nefungují s ní. Zvolil jsem KNN místo 
původního inputu v kostře, jelikož původní inputer dopočítává podle celého
datasetu, kdežto KNN bere v potaz jen n_neighbors (96 se ukázalo jako nejlepší
hodnota) nejpodobnějších řádků, což vypočítá přesnější
hodnoty - určitě se NaN statistika bude více podobat statistice nějakého
podobného championa, než průměru/mediánu všech šampiónu, po každé úpravě ještě
pro data vytvořím nový dataset
'''
imputer_knn = KNNImputer(n_neighbors=6)
champions_imputed = pd.DataFrame(imputer_knn.fit_transform(champions_dropped),
                                 columns=champions_dropped.columns)
'''
Poté jsem eliminoval anomálie, po průzkumu datasetu jsem zjisitl, že některé 
hodnoty mají více anomálií, takže musím eliminovat v některých sloupíchvíce
anomálií, pro každý sloupec jsem vytvořil hodotnu ve slovníku, aby šlo
rychle měnit kolik % je contamination. Jelikož mi nakonec nejlépe fungoval
model GaussianMixture, který s anomáliemi umí pracovat sám, tak odstraňuju jen
malou část anomálií - ty nejextrémnější, odstranění těchto anomálií silně
zvyšuje accuracy, odstranění až moc anomálií u tohoto modelu accuracy sníží,
(třeba Kmeans je na outliers hodně citlivý, proto by měl vyšší %) při testování
jiných modelů jsem také zkoušel měnit konaminacei, abych ověřil,
jaký u nich bude výsledek.
Na hledání anomálií jsem zvolil Isolation Forest, ten nepříjímá NaN, proto bylo
první třeba imputovat NaN values.
Na inpute dat jsem zvolil SimpleImputer. KNNImputer
mi pro GaussianMixture dával horší accuracy. U některách modelů (např. Kmeans)
dával lepší accuracy, ale za cenu velkého prodoloužení času pro výpočet
a to i pro menší n_neighbours. (ID a name_encoded jsem neřešil jako anomálie - 
u ID to nemá cenu, name_Encoded má výrazně rozdílné hodnoty)
Nejlepší strategie pro imputer byla mean
'''
percentage = {
    'HP': 0.01,
    'HP+': 0.01,
    'HP5': 0.01,
    'HP5+': 0.01,
    'MP': 0.01,
    'MP+': 0.01,
    'MP5': 0.001,
    'MP5+': 0.001,
    'AD': 0.001,
    'AD+': 0.001,
    'AS': 0.001,
    'AS+': 0.001,
    'AR': 0.001,
    'AR+': 0.001,
    'MR': 0.001,
    'MR+': 0.001,
    'MS': 0.001,
    'Range': 0.001
}

for key, value in percentage.items():
    column = champions_imputed[[key]]
    isolation_forest = IsolationForest(contamination=value, random_state=69)
    isolation_forest.fit(column)
    # vrací 1/0 jestli je nebo není outlier
    anomaly_selected = isolation_forest.predict(column) == -1
    champions_imputed[key][anomaly_selected] = np.nan
imputer_simple = SimpleImputer(strategy='mean')
champions_non_anomalous = pd.DataFrame(
    imputer_simple.fit_transform(champions_imputed),
    columns=champions_dropped.columns)

'''
Dále jsem udělal scaling dat - normalizaci, jelikož několik model GMM taková
data preferuje . Scaloval jsem všechny údaje až na ID (nepoužívá se
pro clustering) - Podle accuracy se mi více vyplatilo první hledat anomálie
a pak sclaovat. Vyzkoušel jsem i MinMax scaler a dával horčí accuracy
'''
cluster_data = ['HP', "HP+", 'HP5', "HP5+", "MP", "MP+", "MP5", "MP5+",
                'AD', 'AD+', 'AS', 'AS+', "AR", "AR+", "MR", "MR+",
                "MS", "Range", "Name_Encoded"]
scaler = StandardScaler()
champions_scaled = pd.DataFrame(
    scaler.fit_transform(champions_non_anomalous[cluster_data]),
    columns=champions_dropped.columns[1:])
'''
Poslední pre-proccesing úprave je redukce dimenzí - pomůže zrychlit výpočet
clusterů a pokud nějaká data v datasetu korelují, PCA by je mělo odstranit 
nejvíce se mi vyplatilo 13 dimenzí. Pokud data redukuje do méně dimenzí, můžeme
je taky lépe zobrazit a pozorovat počty anomálií, pro každou dimenzi taky 
vytvořím sloupec v datasetu (i for i in range...)
'''
dimensions = 13
pca = PCA(n_components=dimensions)
df_pca = pd.DataFrame(pca.fit_transform(champions_scaled),
                      columns=[i for i in range(dimensions)])
'''
Z True Labels víme, že clusterů je 5, za zvážení ale stojí počet clusterů pro 
model zadat jako 6, do 6. clusteru by se mohly uklidit některé okrajové hodnoty
a ostatní podobnější hodnoty by byly v prvnách pěti clusterech. S tímto řešením
původně pracoval můj model, ale poslední verze vrací lepší accuracy pro 5 clus-
terů. Rozhodl jsem se převážně používat GMM, jelikož je podobný Kmeans, který
také vracel použitelné accuracy, ale lépe zpracovává anomálie a vrací lepší
accuracy, pro tento model jsem zkoušel potupně měnit a kombinovat všechny
jeho parametry, dobrou představu o tom, jak model vyhodnotil labely šlo získat
pomocí counts
Labels a ID se ukládají do output.csv
'''
print("Pre-proccesing done")
gmm = GaussianMixture(n_components=5,
                      random_state=69,
                      covariance_type="tied",
                      tol=0.50,
                      reg_covar=0.000001,
                      init_params="kmeans")
labels = gmm.fit_predict(df_pca)
print("Model finished running")
# _, label_counts = np.unique(labels, return_counts=True)
# print("label counts:", label_counts)
output = pd.DataFrame({'ID': champions_dropped['ID'], 'Label': labels})
output.to_csv('output.csv', index=False)
print("Dataset with IDs and Labels saved as output.csv")
# Accuracy na mém PC - 58.19%

'''
Tato část kódu zobrazí všechna data v datasetu v závisloti na všech datech v
datasetu (20*20 grafů), z nich jde vypozorovat jak moc jsou současné hodnoty
anomální, závislé na sobě, jelikož byl přístupný dataset se správnými Labels,
tak pomocí nich vybarvuju scatter grafy, kdyby z nich náhodou šla vypozorovat
nějaká závislost, to se ale nepodařilo. Pomocí této funkce jsem zároveň zobrazo
val dataset po PCA, a vybarvoval ho vlastními/reálnými labely, kdyby se ukázala
nějaká souvislost.
Jelikož jsem předtím s pandas a seaborn (lepší řešení pro velký figure oproti
čistě matplotlibu, tak jsem druhou půlku kódu vygeneroval pomocí AI, věřím, že
toto použití AI je v pořádku, jelikož neřeší klíčovou myšlenku programu a ani
není zahrnuta tato část kódu v funkčním řešení)


import matplotlib as plt
import seaborn as sns
true_labels_df = get_my_labels()
true_labels_df.columns = ["Index", "True_Label"]
df_visualization_true_labels = pd.concat([champions, pd.Series(labels,
                                         name='Label')], axis=1)
sns.set(style="ticks")
sns.pairplot(df_visualization_true_labels, hue="Label", palette='viridis',
             markers='o')
plt.show()


Vyzkoušené modely: (u všech modelů jsem měnil jak jejich parametry, tak post
proccesing, některé preferují jiná vstupní data - anomálie, standartizace, PCA)

Gaussian Mixtture - dosahuje nejlepší accuracy - 58.19%

Kmeans - podle kostry - rychlejší výpočet než Gaussian, ale slabší accuracy

AgglomerativeClustering - stejný výsledek jako Kmeans, ale dešlí výpočet

DBSCAN - tvoří hodně nerovnoměrné clustery, myslím si, že budou clustery mít
podobný počet prvků


AffinityPropagation - strašně dlouhé vypočítání - nedostal jsem se ani na konec

Meansift - hodně malých clusterů

SpectralClustering - dlouhý výpočet 

HDBSCAN - má tendenci dát většinu dat do jednoho clusteru

OPTICS - tendence dělat jeden velký cluster

'''
