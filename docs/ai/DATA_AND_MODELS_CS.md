# Jeden dataset a návrh encoderů

Datum: 2026-10-15.

## Potvrzený rozsah

Použijeme pouze Amazon Electronics. Druhý dataset je mimo současný rozsah. Dnešní cíl je získat soubory, načíst malý vzorek a připravit seznam modelů. Dataset ani modelové váhy zatím nejsou stažené.

## Stažení

Použij [Amazon Review Data 2018](https://nijianmo.github.io/amazon/index.html), na který odkazuje [CARec README](https://github.com/ChenMetanoia/CARec). Nezaměňuj jej s vydáním 2014 nebo 2023.

V části „Complete review data“, v řádku Electronics, potřebujeme oba soubory:

- `Electronics.json.gz`, plné recenze, z nichž čteme interakce;
- `meta_Electronics.json.gz`, metadata pro text položek.

Oficiální stránka nyní směruje tyto odkazy na [přístupový formulář](https://forms.gle/UEkkJs69e7Z5A5Ps9). Formulář jsem otevřel pouze pro čtení, nic jsem neodeslal. Uvádí pole Name a Affiliation. Přímé soubory ani jejich velikosti jsem neověřoval.

Návrh lokálního umístění:

```text
data/raw/amazon_2018/Electronics.json.gz
data/raw/amazon_2018/meta_Electronics.json.gz
```

Archivy není potřeba rozbalovat. Při prvním načtení přečteme jen několik JSON řádků přes `gzip`, ne celý dataset do paměti. Zkontrolujeme `reviewerID`, `asin`, `overall`, `unixReviewTime` v recenzích a `asin`, `title`, `brand`, `category` v metadatech. `asin` propojuje obě části.

Nestahuj pro tento protokol menší `Electronics_5.json.gz`. CARec začíná vlastní podvzorkovací procedurou a potom filtruje 5-core. Již filtrovaný vstup by dal jiná data. [Konfigurace CARec](https://github.com/ChenMetanoia/CARec/blob/bbfc7f385e796840f6eb5c967dfc8714284958ce/data/data_process_config_amazon_Electronics.yaml), [preprocessing](https://github.com/ChenMetanoia/CARec/blob/bbfc7f385e796840f6eb5c967dfc8714284958ce/data/utils.py).

## Encodery podporované CARec

| Checkpoint | Nativní rozměr | Vstup v implementaci |
| --- | ---: | --- |
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | Text metadat. |
| `sentence-transformers/all-mpnet-base-v2` | 768 | Text metadat. |
| `google-bert/bert-base-uncased` | 768 | CLS reprezentace. Upstream používá krátké ID `bert-base-uncased`. |
| `BAAI/bge-base-en-v1.5` | 768 | CLS a L2 normalizace. |
| `hkunlp/instructor-xl` | 768 | Instruction `Represent the Amazon title:` před textem. |

Zdroje: [runner](https://github.com/ChenMetanoia/CARec/blob/bbfc7f385e796840f6eb5c967dfc8714284958ce/run_collabcontex.py), [encoder větve](https://github.com/ChenMetanoia/CARec/blob/bbfc7f385e796840f6eb5c967dfc8714284958ce/data/utils.py#L163). Tabulka popisuje checkpointy a implementaci, nikoli nové měření jejich doporučovací kvality.

## Návrh našeho menšího výběru

Toto je doporučení k odsouhlasení, nikoli potvrzená změna finálního shortlistu.

| Model | Rozměr před PCA | Role |
| --- | ---: | --- |
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | Jeden překryv s CARec pro kontrolu vlastní pipeline. |
| `intfloat/e5-base-v2` | 768 | Další general-purpose encoder; pro feature vstupy karta doporučuje `query:`. |
| `Qwen/Qwen3-Embedding-0.6B` | 1024 | Instruction-aware model; doporučený první instruction kandidát. |
| `intfloat/multilingual-e5-large-instruct` | 1024 | Druhý instruction model, pokud projde paměťovým a časovým pilotem. |

Zdroje: [MiniLM](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), [E5-base-v2](https://huggingface.co/intfloat/e5-base-v2), [Qwen3](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B), [E5-instruct](https://huggingface.co/intfloat/multilingual-e5-large-instruct).

Začneme pilotem MiniLM. Nové modely stáhneme a vyzkoušíme postupně, ne všechny naráz. V hlavním experimentu zůstává společná PCA při 64/128/256/384 rozměrech. U Qwenu pro tuto tabulku použijeme nejdřív plný vektor a stejnou PCA, aby jeho vlastní zkracování vektorů nepřidalo další rozdíl. Konkrétní instrukci pro položky musíme zapsat před generováním cache; modelové karty ilustrují převážně retrieval, což samo neurčuje optimální instrukci pro doporučování.

## Jak použít výsledky z článku

Publikované výsledky můžeme citovat a uvést v oddělené tabulce jako výsledky autorů. Nemusíme reprodukovat všechny encodery a všechny jejich baseline.

Samotné označení Amazon Electronics ale nezajišťuje stejný experiment. Přímé srovnání vyžaduje shodné vstupní vydání, sampling, filtrování, konkrétní split, text, kandidáty evaluace a trénink. PCA navíc mění reprezentace i u původních encoderů; chybějící matched-dimension výsledky článek nenahradí.

Doporučení je přepočítat jen MiniLM jako společnou kontrolu. Nové encodery porovnáme s ním na stejných našich datech a rozměrech. Bez vlastního společného baseline běhu můžeme reportovat výsledky nové metody a publikovanou referenci, ale ne spolehlivě tvrdit, že rozdíl způsobila výměna encoderu. Jeden kontrolní běh také sám o sobě nedokazuje přesnou reprodukci autorova splitu.
