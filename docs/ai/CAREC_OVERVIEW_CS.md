# CARec: průvodce paperem a repozitářem

Datum analýzy: 2026-10-15. Pro náš projekt nad [CARec, CIKM 2024](https://doi.org/10.1145/3627673.3679535).

Zdrojový kód jsem četl v commitu [`bbfc7f3`](https://github.com/ChenMetanoia/CARec/tree/bbfc7f385e796840f6eb5c967dfc8714284958ce). Níže rozlišuji chování viditelné v kódu, jeho interpretaci a návrhy pro nás. Model jsem nespouštěl. Snippety jsou samostatné vysvětlující příklady, nejsou převzaté z CARec ani hotové napojení na RecBole.

## 1. Co si odnést jako první

CARec používá textový embedding položky jako vstup do doporučovacího modelu. Z historie interakcí se naučí reprezentace uživatelů a transformaci textových vektorů. Díky textu může skórovat i položku, která nemá trénovací interakce.

Pro nás je užitečné místo před vstupem do doporučovacího modelu. Tam vyměníme encoder a přidáme PCA. Zbytek experimentu musí zůstat při porovnání encoderů stejný. Samotné CARec už více encoderů porovnává. Naším příspěvkem má být kontrolované porovnání při shodných rozměrech a měření ceny v paměti a čase. Rozdíl mezi dvěma modely s odlišným backbone nebo pretrainingem sám o sobě neprokazuje účinek instruction tuningu.

Doporučuji nejdřív rozchodit jednu datovou cestu a jeden malý encoder. Spouštět hned celou tabulku modelů by teď především násobilo problémy s daty a evaluací.

Zdroje: [model](https://github.com/ChenMetanoia/CARec/blob/bbfc7f385e796840f6eb5c967dfc8714284958ce/model/collabcontex.py), [runner](https://github.com/ChenMetanoia/CARec/blob/bbfc7f385e796840f6eb5c967dfc8714284958ce/run_collabcontex.py), naše [metodika](../../third_milestone.md).

## 2. Co už máme lokálně

| Soubor | K čemu nám je |
| --- | --- |
| `assignment.md` | Požadavky předmětu a povinnost rozšířit existující metodu. |
| `topic.md` | Pracovní výzkumná otázka a vybraný paper. |
| `SECOND_MILESTONE.md`, `third_milestone.md` | Related work a návrh experimentu. |
| `6ac65ba1a33e4f01bca06b79/kovarm46.tex` | Rukopis článku; bibliografie je vedle v `.bib`. |
| `src/carec_encoder_comparison.ipynb` | Zatím úvod, importy a výběr zařízení s prioritou MPS. |
| `docs/ai/` | Rozhodnutí, ověřené zdroje a další práce. |

Nemáme tady CARec implementaci, stažená data, embedding cache ani výsledky baseline. Notebook není implementovaný experiment. To je důležité pro plánování dalšího kroku.

## 3. Mapa upstream repozitáře

| Soubor nebo složka | Co skutečně dělá | Co si v něm přečíst |
| --- | --- | --- |
| `run_collabcontex.py` | Sestaví konfiguraci, načte RecBole dataset, spustí dvě fáze a evaluaci. | `set_up_config`, `run_evaluation_phases`. |
| `model/collabcontex.py` | Vlastní model, loss, přepínání trénovatelných parametrů, grafová propagace a skórování. | `CollabContex`, `LGCNEncoder`, `MLP`. |
| `data/process_data.py` | Vstupní skript preprocessingu. Výchozí dataset je Electronics. | Načtení YAML a `preprocess`. |
| `data/utils.py` | Čtení dat, filtrování, cold split, texty, encodery, mapování ID a export. | `RawDataProcessor`, `split_data`. |
| `data/data_process_config_amazon_Electronics.yaml` | Volba surových dat, vzorkování, encoderu a splitu. | `inter_sample_ratio`, `text_feature_list`, `cold_ratio`, `eval_args`. |
| `properties/train.yaml` | Batch, learning rate, počet epoch, early stopping a metriky. | Zejména `valid_metric`. |
| `properties/data.yaml` | RecBole formát vstupu, soubory splitů a full-sort evaluace. | `benchmark_filename`, `load_col`. |
| `utils/util.py` | Dataloadery, samplery a vlastní cold evaluator. | `ColdEvaluator.get_mask`, `eval_func`. |
| `recbole_gnn/` | Grafové vrstvy a další modely. Pro hlavní cestu CARec je podstatný `LightGCNConv`. | `model/layers.py`. |

Názvy `CollabContex` a `CollabFusion` v cestách jsou názvy v kódu. Neznamenají, že jsme omylem otevřeli jiný paper.

Odkazy: [strom souborů](https://github.com/ChenMetanoia/CARec/tree/bbfc7f385e796840f6eb5c967dfc8714284958ce), [grafová vrstva](https://github.com/ChenMetanoia/CARec/blob/bbfc7f385e796840f6eb5c967dfc8714284958ce/recbole_gnn/model/layers.py).

## 4. Jak projde jedna položka celou cestou

```text
Amazon metadata                     Amazon reviews
název + značka + kategorie           uživatel, položka, čas
         |                                  |
   textový encoder                 filtrování a pevný split
         |                                  |
   E_raw [položky, D]                trénovací graf interakcí
         |
   naše PCA, fit jen na train položkách
         |
   E [položky, d] ---------------------------+
         |                                  |
         +---- CARec: uživatelé + MLP + LightGCN
                             |
                       skóre a top-k
```

PCA je náš plánovaný zásah. Upstream ji v této cestě nemá. Textové embeddingy se v doporučovacím tréninku nepočítají znovu a encoder se zde netrénuje.

Pozor na slovo „title“. Konfigurace Electronics má `text_feature_list: [title, brand, category]` a preprocessing tato pole skutečně spojí. Instruction sice říká `Represent the Amazon title:`, vstup je ale bohatší. Pro reprodukci bych zachoval metadata. Varianta pouze s názvem musí být samostatně popsaná změna protokolu.

Zdroje: [preprocessing, řádky 131–235 a 334–361](https://github.com/ChenMetanoia/CARec/blob/bbfc7f385e796840f6eb5c967dfc8714284958ce/data/utils.py#L131), [Electronics YAML](https://github.com/ChenMetanoia/CARec/blob/bbfc7f385e796840f6eb5c967dfc8714284958ce/data/data_process_config_amazon_Electronics.yaml).

## 5. Co se model učí

Máme matici textových vektorů `E` s tvarem `[n_items, d]`, tabulku uživatelů `U` s tvarem `[n_users, d]` a MLP s převodem `d -> d`.

| Fáze | Co se učí | Co je zmrazené | Vstup položek do grafu |
| --- | --- | --- | --- |
| Item tutoring | Uživatelská tabulka `U`. | Textové vektory a MLP. | Původní `E`, MLP se obchází. |
| User tutoring | Parametry MLP. | Uživatelská tabulka a textové vektory. | `MLP(E)`. |

„Uživatelská tabulka je zmrazená“ neznamená, že se při druhé fázi nemůže změnit výsledný vektor uživatele po průchodu grafem. Sousední položky nyní procházejí MLP, takže se mění jejich příspěvek do agregace.

LightGCN posílá vektory podél hran uživatel–položka. Hrana má váhu podle stupňů obou uzlů. Model průměruje reprezentaci před propagací a po každé vrstvě. Výchozí runner používá dvě vrstvy.

Loss přibližuje normalizovaný vektor uživatele a položky z pozorované interakce. Současně uniformity člen rozprostírá reprezentace, aby se všechny nesešly v jednom bodě. V první fázi se počítá uniformity uživatelů, ve druhé položek. Tento model tedy nemá obvyklou BPR loss s pozitivní a negativní položkou.

Schematicky:

```text
alignment = průměr ||normalize(u) - normalize(i)||²
uniformity = log průměr exp(-2 * párová_vzdálenost²)
loss = alignment + uniformity aktivní strany
```

Tohle je vysvětlení cíle. Při reprodukci musíme zachovat i detaily, například odstranění duplicit uživatelů nebo položek před uniformity členem.

Zdroj: [model, řádky 32–104 a 123–199](https://github.com/ChenMetanoia/CARec/blob/bbfc7f385e796840f6eb5c967dfc8714284958ce/model/collabcontex.py#L32).

## 6. Data a split, které nesmíme měnit mezi encodery

V kontrolovaném commitu dělá Electronics preprocessing následující kroky:

1. Vezme 35 % interakcí, seed je 2023.
2. Iterativně filtruje uživatele a položky na alespoň pět interakcí.
3. Omezí interakce na dostupná metadata.
4. Náhodně vybere 5 % položek a přesune všechny jejich interakce do cold části.
5. Znovu provede 5-core filtrování warm části a v cold části ponechá jen uživatele, kteří přežili ve warm datech.
6. Warm interakce rozdělí náhodně po uživatelích v poměru 80/10/10. Počty train a valid zaokrouhluje dolů, zbytek připadne testu.
7. Exportuje `.item`, `.train.inter`, `.valid.inter`, `.test.inter`, `.cold.inter`.

Finální podíl cold položek proto nemusí být přesně 5 %. Poměr se zadává před dalším filtrováním. Uživatel s pěti warm interakcemi může mít při tomto zaokrouhlení prázdnou valid část.

Tohle není časový split. Čas je v datech, ale konfigurace používá random order. Seed samotný nestačí jako důkaz shodných dat. Uložíme jednou výsledné soubory, mapování ID a jejich hashe, potom měníme jen embeddingy.

Zdroje: [konfigurace](https://github.com/ChenMetanoia/CARec/blob/bbfc7f385e796840f6eb5c967dfc8714284958ce/data/data_process_config_amazon_Electronics.yaml), [filtrování](https://github.com/ChenMetanoia/CARec/blob/bbfc7f385e796840f6eb5c967dfc8714284958ce/data/utils.py#L237), [split a export](https://github.com/ChenMetanoia/CARec/blob/bbfc7f385e796840f6eb5c967dfc8714284958ce/data/utils.py#L447).

## 7. Jak číst výsledky

Warm evaluace používá full-sort cestu s MLP. Cold větev runneru MLP vypne. Vlastní evaluator potom maskuje všechny kandidáty kromě položek v cold datech. Cold Recall@10 tedy měří hledání mezi cold kandidáty. Není to totéž jako žebříček nad společným katalogem warm a cold položek.

Recall@10 měří, jaký podíl relevantních testovacích položek se dostal do první desítky. NDCG@10 navíc zvýhodní relevantní položky na začátku žebříčku. Pro přesnou reprodukci převezmeme i pravidla průměrování metrik a maskování historie z použité verze evaluatoru.

Výchozí trénovací YAML má batch 1024, learning rate 0.001, limit 200 epoch a early stopping 30 validačních kroků. Vybírá checkpoint podle NDCG@10, reportuje Recall a NDCG pro k=10 a 50. Jsou to konfigurační hodnoty, ne naměřená doba ani záruka, že je vhodné použít je pro každý encoder.

Zdroje: [cold evaluator](https://github.com/ChenMetanoia/CARec/blob/bbfc7f385e796840f6eb5c967dfc8714284958ce/utils/util.py#L393), [runner](https://github.com/ChenMetanoia/CARec/blob/bbfc7f385e796840f6eb5c967dfc8714284958ce/run_collabcontex.py#L86), [trénovací konfigurace](https://github.com/ChenMetanoia/CARec/blob/bbfc7f385e796840f6eb5c967dfc8714284958ce/properties/train.yaml).

## 8. Místa, která potřebují ověřit před baseline

Tohle jsou závěry ze čtení zdrojů. Runtime chyby zatím nejsou reprodukované.

| Nález | Praktický dopad |
| --- | --- |
| Encoder větev MiniLM/MPNet/INSTRUCTOR má natvrdo `device='cuda:2'`. Pomocná volba zařízení zná CPU a CUDA. | Na M4 Pro nestačí přepnout zařízení v našem notebooku. |
| Runner odvozuje rozměr uživatelů i MLP z krátkého seznamu encoderů. | Nové encodery a PCA vyžadují samostatný parametr rozměru. |
| INSTRUCTOR větev zřetězí instruction s textem. | Ověřit očekávané rozhraní přesné verze knihovny. Tento zápis nepřebírat jako obecný vzor pro INSTRUCTOR. |
| `predict()` odkazuje na embeddingy na hlavním modelu, zatímco konstruktor je ukládá v `encoder`. | Full-sort cesta používá jinou metodu. Samostatná predikce potřebuje ověření. |
| Cold evaluator provede `scores[mask]` před reshape na `[batch, items]`. | Model vrací ploché skóre; pro batch větší než jedna vypadá délka masky nekompatibilně. Ověřit na malém příkladu. |
| Vypnutí MLP vede k novému výpočtu grafových vektorů obou stran. | Ověřit shodu cold inference s popisem paperu; nestačí předpokládat pevný uživatelský vektor a raw cold položku. |
| Přechod mezi fázemi resetuje `cur_step`, používá stejný Trainer. | Zkontrolovat best score, epochy a optimizer v RecBole 1.1.1. Nevíme zatím, zda druhá fáze dostane nezávislý checkpoint a očekávaný počet epoch. |
| Celý strom neobsahuje zjevný licenční soubor. | Podle dosavadního pravidla projektu zatím zdroj pouze analyzujeme; převzetí a úpravy zůstávají otevřené. |

README uvádí starý stack s Pythonem 3.8.5, PyTorchem 1.12.1 a RecBole 1.1.1. Model navíc importuje PyTorch Geometric. Instalaci na současný Mac musíme ověřit pilotem. README není ověřený postup pro MPS.

Zdroje: [preprocessing encoderů](https://github.com/ChenMetanoia/CARec/blob/bbfc7f385e796840f6eb5c967dfc8714284958ce/data/utils.py#L163), [model](https://github.com/ChenMetanoia/CARec/blob/bbfc7f385e796840f6eb5c967dfc8714284958ce/model/collabcontex.py#L106), [evaluator](https://github.com/ChenMetanoia/CARec/blob/bbfc7f385e796840f6eb5c967dfc8714284958ce/utils/util.py#L465), [README](https://github.com/ChenMetanoia/CARec/blob/bbfc7f385e796840f6eb5c967dfc8714284958ce/README.md).

## 9. Snippety pro pochopení našeho rozšíření

Tyto příklady budeme později přizpůsobovat skutečným datům. Nedefinované proměnné představují výstupy předchozích kroků. Kód zde neslibuje kompatibilitu s CARec ani nenahrazuje jeho trénink.

### A. Vektor patří ke konkrétnímu ID

```python
import numpy as np

# item_ids mají stejné pořadí jako řádky raw_vectors.
assert len(item_ids) == len(set(item_ids))
assert raw_vectors.ndim == 2
assert raw_vectors.shape[0] == len(item_ids)
assert np.isfinite(raw_vectors).all()

row_for_id = {item_id: row for row, item_id in enumerate(item_ids)}
train_rows = np.array([row_for_id[i] for i in sorted(set(train_item_ids))])
assert set(train_item_ids).isdisjoint(cold_item_ids)
```

Správný tvar nestačí. Když zaměníme pořadí položek, model dostane platné vektory přiřazené špatným produktům. RecBole navíc tokeny znovu mapuje a pracuje s padding ID. Načtení `.item` musí respektovat jeho mapování, ne naše pořadí ze CSV.

### B. PCA se učí jen z trénovacích položek

```python
from sklearn.decomposition import PCA

# Každý encoder má vlastní PCA fit, ale stejné nastavení a train ID.
assert target_dim <= min(len(train_rows), raw_vectors.shape[1])
pca = PCA(n_components=target_dim, svd_solver="full")
pca.fit(raw_vectors[train_rows])
item_vectors = pca.transform(raw_vectors).astype(np.float32)
```

PCA může transformovat i cold položky. Nesmí se z nich učit projekci. Uložíme také PCA parametry. Tento příklad používá centrovanou PCA bez standardizace a bez následné L2 normalizace. Zda přidat standardizaci a normalizaci, musíme před experimentem zapsat jako společný protokol. PCA na nativním rozměru stále centruje a rotuje data; není automaticky totožná s původním embeddingem.

### C. Skórování a maskování na malém příkladu

```python
# user_vectors: [batch, d], item_vectors: [items, d]
scores = user_vectors @ item_vectors.T
assert scores.shape == (len(user_vectors), len(item_vectors))
scores[:, ~candidate_mask] = -np.inf
ranking = np.argsort(-scores, axis=1)[:, :10]
```

`candidate_mask` je boolean vektor přes katalog. Při cold-only evaluaci obsahuje jen cold položky. Při společném katalogu je jiný. V reálné evaluaci navíc odmaskujeme padding a známou historii a stanovíme zacházení se shodnými skóre. Tento příklad ukazuje maticové skórování, ne celý CARec evaluator ani definici jeho výsledných uživatelských vektorů.

## 10. Doporučené pořadí společné implementace

1. Připravit konfiguraci experimentu a malé datové rozhraní. Jedna položka má stabilní ID a text, jedna interakce uživatele a item ID. Předem určit textová pole.
2. Na malé sadě textů ověřit MiniLM na dostupném zařízení. Změřit čas a paměť, uložit vektory spolu s ID a revizí modelu. Je to technický pilot, ne baseline výsledek.
3. Připravit jednou Electronics split, audit počtů a cold/train disjunktnosti. Zmrazit soubory a hashe.
4. Přidat PCA fit na train ID a cache pro jeden rozměr, například 128.
5. Vyřešit cestu doporučovače a ověřit dvě fáze i obě evaluace. Zaznamenat každou odchylku od upstreamu. Menší vlastní model může být baseline, ale nesmíme ho označit za reprodukci CARec.
6. Teprve po funkční baseline přidat instruction encoder a sweep 64/128/256/384. Více seedů až po změření doby jednoho běhu.

První společný krok bych držel malý: definovat záznam položky a kontrakt cache. Hotovo bude, až dokážeme pro několik textů uložit a znovu načíst vektory bez ztráty vazby na ID. Tím získáme základ, který využije libovolný encoder i zvolený doporučovač.

Každý výsledek musí později nést alespoň identifikátor splitu, encoder a jeho revizi, instruction, textová pole, PCA nastavení a fit ID, rozměr, seed, trénovací konfiguraci a kandidátní množinu evaluace. Bez toho nepůjde poznat, co způsobilo rozdíl v metrikách.
