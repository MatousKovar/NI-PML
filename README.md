# NI-PML benchmarky

Experimenty s textovými encodery a rozměrem item embeddingů v CARec. Jediným datasetem je Amazon Electronics 2018. Zadání je v [assignment.md](assignment.md), metodika v [third_milestone.md](third_milestone.md).

## Prostředí

Z kořene repozitáře spusť:

```bash
bash scripts/setup.sh
source .venv/bin/activate
```

Pro nové prostředí potřebuješ nainstalovaný Python 3.11, dostupný jako `python3.11`. Setup vytvoří `.venv`, nainstaluje knihovny pomocí `pip install -r requirements.txt`, připojí CARec na commitu `bbfc7f385e796840f6eb5c967dfc8714284958ce` a zaregistruje kernel `NI-PML benchmarks` do uživatelského Jupyteru. Kernel používá Python z projektového `.venv`, ale najde ho i Jupyter spuštěný z jiného prostředí. Existující funkční `.venv` použije znovu. Systémový Python neupravuje. Pro jinou cestu k Pythonu použij `PYTHON_BIN=/cesta/k/python3.11 bash scripts/setup.sh`.

V editoru vyber interpreter `.venv/bin/python`. Notebooky mají oddělené úlohy:

- [src/data_exploration.ipynb](src/data_exploration.ipynb) obsahuje náhled dat, původní přípravu textů CARec, sampling, split, ukládání a ověření.
- [src/carec_encoder_comparison.ipynb](src/carec_encoder_comparison.ipynb) načítá uložený katalog, splity a mapování ID. Tady pokračují experimenty s encodery a doporučováním.

Projekt má RecBole 1.1.1 a PyTorch 2.6.0. Jde o prostředí upravené pro současný Mac, nikoli přesnou kopii původního CUDA prostředí autorů.

Pokud VS Code kernel nenabízí, spusť `Developer: Reload Window`. V notebooku pak vyber `Select Another Kernel…` → `Jupyter Kernels…` → `NI-PML benchmarks`. Interpreter pro Python soubory a kernel notebooku se vybírají samostatně.

Samotnou instalaci knihoven do aktivovaného prostředí provedeš také přímo:

```bash
python -m pip install -r requirements.txt
```

## Data

První příprava v `src/data_exploration.ipynb` používá rozbalené soubory:

```text
data/Electronics.json
data/meta_Electronics.json
```

Jsou to JSONL soubory s jedním záznamem na řádek. Mají dohromady přibližně 22,5 GB. Sekce `Prepare and save the CARec sample` v datovém notebooku vybírá původních 35 % interakcí se seedem 2023 a volá původní CARec filtrování a rozdělení. Při opakovaném spuštění načítá uložený vzorek. Experimentální notebook pouze načte hotové soubory a zobrazí jejich počty a náhled.

Ověřená data jsou v `data/processed/electronics_carec_035_seed2023/`:

- `sampled_interactions.tsv.gz` a `sampled_metadata.jsonl.gz` uchovávají vstupní vzorek a potřebná metadata pro opakování přípravy.
- `train.tsv.gz`, `valid.tsv.gz`, `test.tsv.gz` a `cold.tsv.gz` obsahují pevné rozdělení s původními ID a čísly zdrojových řádků.
- `items.jsonl.gz` obsahuje katalog, metadata a text připravený původní metodou CARec.
- `user_ids.tsv.gz` a `item_ids.tsv.gz` uchovávají mapování od nuly. RecBole bude potřebovat vlastní přemapování včetně paddingu.
- `manifest.json` obsahuje konfiguraci, kontrolní součty, počty a výsledek ověření.

Po kontrole součtů, opětovného načtení a shodného přepočítání ze vzorku má manifest stav `verified`. Potom lze oba původní JSONL soubory odstranit; náhled v notebooku i kontrola prostředí umějí načíst uložený vzorek. Nový výběr z celé původní populace by vyžadoval opětovné stažení dat. Uložené soubory zachovávají potřebná pole, nikoli kompletní recenzní texty a všechna původní metadata.

Data, cache, checkout CARec a lokální prostředí ignoruje Git. `requirements.txt` obsahuje přesné verze hlavních knihoven i jejich závislostí z ověřeného prostředí na Macu s Pythonem 3.11.

## Kontrola a importy

```bash
.venv/bin/python -m benchmarks.check_environment
```

Kontrola vypíše verze a commit, načte deset řádků obou původních souborů nebo uloženého ověřeného vzorku a zkusí grafovou vrstvu. Na malé syntetické grafové ukázce ověří také loss obou fází CARec a tvar skóre na CPU. Dostupné MPS vyzkouší pro samotnou grafovou vrstvu. Nepočítá metriky, negeneruje embeddingy a nespouští preprocessing celého datasetu. Bez dat lze použít `--skip-data`, pro jiné umístění dat `--data-dir`.

Ve vlastním kódu spuštěném z kořene repozitáře:

```python
from benchmarks.carec import load_carec
from recbole.config import Config
from recbole.trainer import Trainer

carec = load_carec()
CollabContex = carec.CollabContex
ColdEvaluator = carec.ColdEvaluator
data_preparation = carec.data_preparation
```

Wrapper importuje originální zdroje z `external/CARec`, kontroluje commit i změny sledovaných souborů. CARec není balíček pro `pip install`; jeho interní importy používají obecné názvy `model`, `utils` a `recbole_gnn`. Pokud se střetnou s dříve importovaným modulem, restartuj kernel. Importy neprovádějí trénink.

## Co zbývá před benchmarkem

Pevný split, produktové texty a mapování ID jsou uložené a ověřené. Zbývají cache encoderů a redukce rozměru. Datový notebook volá datové metody CARec bez jeho automatického generování embeddingů, které má natvrdo CUDA zařízení. RecBole má vlastní volbu CPU/CUDA, takže notebookové `DEVICE` samo nenastaví MPS pro jeho Trainer. Funkční import ani malý výpočet nejsou důkazem hotové MPS podpory celého CARec tréninku.

Audit původního protokolu našel 3 844 opakovaných dvojic uživatel–produkt mezi train a test, 7 warm produktů bez trénovací hrany a 69 134 uživatelů bez validačních interakcí kvůli zaokrouhlování splitu. Záznamy jsme zatím zachovali stejně jako CARec. Před vyhodnocením musíme uzavřít zacházení s těmito případy.

Cold evaluator a přechod mezi dvěma trénovacími fázemi potřebují ověřit před měřením. Přehled známých míst je v [průvodci CARec](docs/ai/CAREC_OVERVIEW_CS.md). U PyTorche 2.6 je také třeba ověřit načítání checkpointů starého RecBole. Dosavadní kontrola checkpointy nepoužívá.

Upstream je v samostatném lokálním checkoutu bez našich úprav. [Veřejný repozitář CARec](https://github.com/ChenMetanoia/CARec) v kontrolovaném commitu nemá zjevný licenční soubor. Použití v tomto pracovním prostředí uživatel výslovně zadal 2026-10-17; podmínky dalšího zveřejnění převzatého kódu zůstávají neověřené.
