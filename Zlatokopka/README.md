# Knihovny

V souboru je seznam všech knihoven, které jsou potřeba doinstalovat pro spuštění programu přes PIP. U některých knohoven je uvedena i požadovaná verze. Tuto verzi knihovny prosím dodržte, aby nedošlo k problémům s kompaktibilitou při opravování. To, zda máte nainstalováno vše ve správné verzi můžete ověřit pomocí funkce `testLibsVersions()` v souboru `main.py`.

# Soubory

Ve složce src najdete 4 soubory: `dataset.py`, `downloadUtils.py`, `model.py` a `main.py`. Obsah souborů `dataset.py`, `downloadUtils.py` a `model.py` vás pro vyřešení úlohy nemusí zajímat (ale pokud vás zajímá, jak je model implementovaný, doporučuji se samozřejmě podívat). Tyto soubory obsahují implementaci některých tříd a funkcí, které budete v úloze používat, které jsou poppsané níže. **V těchto souborech nic neměňte, jinak je možné, že odevzdaný model nebude na testech vůbec fungovat!**.

Soubor `main.py` ukazuje několik příkladů, jak pracovat s poskytnutými třídami a hledat tak hyperparametry a učit a používat poskytnutý model.

# Poskytnuté třidy

Většina metod většiny tříd obsahuje spoustu volitelných argumentů, které nemusíte vůbec používat. Typicky ze začátku je doporučené mířit co nejjednodušeji a zjistit, co přesně model dělá a až poté zkusit pracovat s rozšiřujícími parametry na případné dorbné zlepšení modelu. Některé argumenty nebyly použity ani ve vzorovém řešení. 

## Fashion Dataset

Tato třída se importuje ze souboru `dataset.py` příkazem `from dataset import FashionDataset`. Slouží ke stažení datasetu a usnadnění práce s datasetem, abyste nemuseli řešit práci ze soubory, složkami, formátování vstupu, načítání obrázků a podobně. S instancí třídy můžete pracovat podobně jako s listem, tedy zjistit jeho délku a přistupovat na jednotlivá data pomocí indexů v hranatých závorkách. Třída má tyto metody a atributy:

- `init(size, transform, cacheToRAM, maxImagesPerClass, classes, forceDownload, _extracted)`:
  - `size: Tuple[int,int]`, default `(1440, 1080)`: Velikost v pixelech, na kterou budou automaticky všechny obrázky po načtená upraveny.
  - `transform: Optional[callable]`, default `None`: Transofrmace obrázku, která bude aplikována při každém jeho použití, na všechny obrázky v datasetu. Pokud je parametr None, nebude provedena žádná transformace. Jde o funkci, která bere i vrací obrázek jako typ `torch.tensor`. Pokud neumíte s `tensory` pracovat, můžete použít na začátku funkci `FastionDataset.tensorToOpenCVImage(tesnor)`, která převede tensor na `np.array`, jak jste zvyklí z předchozí vlny a na konci funkci `transforms.ToTensor()(tensor)`, která se importuje z knihovny `torchvision` a převede obrázek z `np.array` zpět na `tensor`. Pro již existující transformace se doporučuji podívat [sem](https://pytorch.org/vision/0.9/transforms.html). **Při testování modelu nebudou na data aplikovány žádné transformace.**
  - `cacheToRam: bool`, default `False`: Pokud je `True`, uloží se na začátku všechny obrátky do RAMky, pokud `False`, pak se načítají obrázky jednotlivě z disku až při jejich zavolání. Uložení obrázků do RAM-ky výrazně zrychlí trénovací proces. Ovšem dataset je velký a RAM omezená a celý jej v plném rozlišení do RAM pravděpodobně nedáte. Proto doporučuji zvoliz `True` jen pro menší subsety a pro celý set nastavit parametr na `False`.
  - `maxImagesPerClass: Optional[int]`, default `None`: Pokud je nastaven na číslo `n`, načte se z každé třídy nejvýše `n` obrázků. Pokud je `None`, načtou se obrázky všechny.
  - `classes: Optional[List[str]]`, default `None`: Pokud není `None`, načtou se pouze obrázky z těch tříd, které jsou zde specifikovány. Pokud je `None`, načtou se obrázky ze všech tříd.
  - `forceDownload: bool`, default: `False`: Pokud je `True`, celý dataset se stáhne znova nezávisle na tom, zda je již stažený nebo ne.
  - `_extracted: bool`, default: `False`: Parametr pro interní použití, doporučuji nepoužívat a nechat `False`.

**Pokud dataset není stáhlý (typicky při prvním spuštění), bude zavoláním konstrukotru automaticky stáhnut. Dataset zabírá přibližně 10 GB.**

- `tensorToOpenCVImage(tensor: torch.Tensor) -> np.ndarray`: Statická funkce na převod obrázku ve formátu `torch.tensor` na obrázek ve formátu `np.ndarray`, pro práci s knihovnou OpenCV
- `extractSubset(percentage, balanceClasses, transform) -> 'FashionDataset'`: Extrahuje z detasetu jeho podmnožinu, kterou vrátí jako novou instanci třídy `FashionDataset` a data z této podmnožiny z aktuálního datasetu smaže.
  - `percentage: float`, default `0.1`: Kolik procent dat z datasetu bude extrahováno.
  - `balanceClasses: bool`, default `True`: Pokud je `True`, bude do nového datasetu extrahováno `n` % obrázků z každé třídy v původním setu. Pokud je `False`, bude extrahováno `n` % náhodných obrázků z celého datasetu.
  - `transform: Optional[callable]` default `None`: stejné, jako `transform` v konstruktoru třídy.
- `__len__`: Vrátí počet prvků v datasetu, použití: `len(dataset)`. 
- `__getitem__(self, idx: int) -> dict`: Vrátí záznam na daném indexu, použití: `item = dataset[i]`. Výsledek je slovník, který má 2 záznamy. Záznam s klíčem `"img"` obsahuje samotný obrázek ve formátu `torch.tensor`. Pro přeformátování na OpenCV obrázek můžete použít funkci `tensorToOpenCVImage`. Dále záznam s klíčem `"label"`, který je ve formátu pole (jako `torch.tensor`), který má 1 na pozici třídy, kterou obrázek představuje a 0 kdekoliv jinde. Pro získání názvu třídy můžete použít: `dataset.labels[torch.argmax(dataset[i]["label"]).item()]`
- `labels: List[str]`: Pole všech tříd obrázků.

## activations
  
Jde o enumerate, který popisuje možné hodnoty aktivačních funkcí (jednoho z hyperparametrů), který se importuje ze souboru `model.py` příkazem `from model import activations`. Má 4 hodnoty: `RELU`, `LEAKY_RELU`, `SIGMOID`, `TANH`. Příklad použití: `activationFunction = activations.TANH`.

## HParams

Třída, ukládající všechny hyperparametry popsané v zadání úlohy, která se importuje ze souboru `model.py` příkazem `from model import HParams`.. Příklad použití: `params = HParams(convCount = 1, kernelSize = 1, ...)`. Konstruktor bere všechny možné parametry popsané v zadání. Dále obsahuje metodu `__str__` (použvaná jako `string = str(params)`) a statickou metodu `parseFromString(string: str) -> HParams`. Tyto 2 metody převedou hyperparms na string a ze stringu zpět na objekt.

## FashionModel

Třída, která představuje samotný model, který máte za úkol natrénovat. Je opět v souboru `model.py` a importuje se pomocí příkazu `from model import FashionModel`. Tato třída obsahuje následující metody s následujícími argumenty:

- `init(hparams, classes, imageSize, device)`: Konstruktor třídy
  - `hparams: HParams` **required**: Hyperparametry pro daný model.
  - `classes: List[str]` **required**: Pole tříd, která se v datasetu vyskytují (musí být ve stejném pořadí, jako jsou v datasetu). Předpokládá se, že sem předáte hodnotu `dataset.labels` z vašeho trénovacího datasetu.
  - `imageSize: Tuple[int, int]`, default `(1440, 1080)`: Rozlišení obrázků, které použijete pro trénování a evaluaci modelu. **MUSÍ BÝT STEJNÉ, JAKO ROZLIŠENÍ OBRÁZKŮ V POUŽITÉM DATASETU**
  - `device:  Optional[torch.device]`, default `None`: Zařízení, na kterém mají běžet výpočty (CPU nebo GPU). Pokud je nastavený na `None` (dpoporučená varianta), pak bude automaticky detekováno, zda váš systém podporuje výpočty na GPU a pokud ano, bude GPU automaticky zvoleno. Jinak bude zvoleno CPU.
  
- `predict(image) -> str`: Metoda, která vezme jeden obrázek a predikuje jeho třídu (předpoklá, že je model již naučený, jinak vrátí "náhodný" výsledek)
  - `image: Union[torch.Tensor, np.ndarray]` **required**: Obrázek k predikci ve formátu `ndarray` (default načtený pomocí openCV) nebo `torch.tensor` (formát, ve kterém jsou obrázky uložené v poskytnuté třídě `FashionDataset`)

- `predictBatch(images) -> List[str]`: Stejné, jako predict, akorát bere na vstupu pole obrázků a ohodnotí třídu všech na jednou. Vrátí pak pole tříd.
  - `images: Union[torch.Tensor, List[torch.Tensor], List[np.ndarray]]` **required**: Pole obrázků, které jsou ve stejném formátu jaku v případě funkce `predict`.

- `train(dataset, epochs, showEpochProgressBar, showTotalProgressBar) -> List[float]`: Metoda, která provede jednu nebo více iterací trénování modelu na poskytnutém datasetu. Vrací pole hodnot loss funkce, pro každou provedenou iteraci.
  - `dataset: Dataset` **required**: Dataset obrázků, na kterém bude model natrénován.
  - `epochs: Optional[int] = None`, default `None`: Počet iterací, kolikrát se má trénování provést (stejné jako kdyby byla metoda train zavolána `n`-krát). Pokud je hodnota `None` nebo nižší než 1, bude automaticky nastavená na 1.
  - `showEpochProgressBar: Optional[bool]`, default `True`: Pokud je `True`, zobrazí se pro každou iteraci progress bar do konzole, jinak ne.
  - `showTotalProgressBar: Optional[bool]`, default `True`: Pokud je `True`, a zároveň je počet iterací větší než 1, zobrazí se do konzole progres bar celkového učení, který se updatuje po dokončení každé iterace. Není vhodné kombinovat oba 2 typy progresbarů.

- `evaluate(dataset) -> Tuple[float, float]`: Metoda, která spoučítá loss a accuracy modelu na poskytnutém datasetu. Tyto 2 hodnoty vrací jako tuple ve formátu `(loss, accuracy)`.
  - `dataset: Dataset` **required**: Dataset obrázků, na kterém bude model hodnocen.

- `save(name, path) -> str:` Metoda, která uloží model. (Soubory, vygenerované touto metodou budete na konci odevzdávat). Každý model je uložen do vlastní složky a obsahuje 2 soubory, kde jeden si pamatuje hyperparametry modelu a druhý reálné parametry, které byly naučeny. Odevzdávejte oba 2 soubory přímo do odevzdávátka (nikoliv v zazipované složce). Metoda vrací absolutní cestu ke složce, ve které je model uložen.
  - `name: Optional[str]`, default `None`: Název modelu, pod kterým se uloží. Pokud je `None`, použije se jako název aktuální timestamp. POkud složka již existuje, bude jméno automaticky změněno.
  - `path: Optinonal[str]`, default `None`: Cesta ke složce, do které bude model uložen. Cesta je relativní vůči root složce projektu (tedy té, ve které se nacház např. tento soubor `README`). Pokud je `None`, použije se složka `models`.

- `load(name, path, device) -> FashionModel`: Statická metoda, která načte model uložený pomocí metody `save`.
  - `name: str`, **required**: Název modelu, který se má načíst. Funguje stejně, jako parametr `name` u metody `save`.
  - `path: Optinonal[str]`, default `None`:  Cesta ke složce, ve které je uložený model, který se má načíst. Funguje stejně, jako parametr `path` u metody `save`.
  - `device: Optional[torch.device]`, default `None`: Zařízení, na kterém mají běžet výpočty. Funguje stejně, jako parametr `device` u konstuktoru třídy.
