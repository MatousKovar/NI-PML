# NI-PML benchmarky

Experimenty s textovými encodery a rozměrem item embeddingů v CARec. Jediným datasetem je Amazon Electronics 2018. Zadání je v [assignment.md](assignment.md), metodika v [THIRD_MILESTONE.md](THIRD_MILESTONE.md).

## Prostředí

Z kořene repozitáře spusť:

```bash
bash scripts/setup.sh
source .venv/bin/activate
```

Pro nové prostředí potřebuješ nainstalovaný Python 3.11, dostupný jako `python3.11`. Setup vytvoří `.venv`, nainstaluje knihovny pomocí `pip install -r requirements.txt`, připojí CARec na commitu `bbfc7f385e796840f6eb5c967dfc8714284958ce` a zaregistruje kernel uvnitř `.venv`. Existující funkční `.venv` použije znovu. Systémový Python neupravuje. Pro jinou cestu k Pythonu použij `PYTHON_BIN=/cesta/k/python3.11 bash scripts/setup.sh`.

V editoru vyber interpreter `.venv/bin/python`. Notebook je [src/carec_encoder_comparison.ipynb](src/carec_encoder_comparison.ipynb). Projekt má RecBole 1.1.1 a PyTorch 2.6.0. Jde o prostředí upravené pro současný Mac, nikoli přesnou kopii původního CUDA prostředí autorů.

Samotnou instalaci knihoven do aktivovaného prostředí provedeš také přímo:

```bash
python -m pip install -r requirements.txt
```

## Data

Lokálně očekáváme již rozbalené soubory:

```text
data/Electronics.json
data/meta_Electronics.json
```

Jsou to JSONL soubory s jedním záznamem na řádek. Mají dohromady přibližně 22,5 GB. Data, cache, checkout CARec a lokální prostředí ignoruje Git. Soubor `requirements.txt` naopak patří do verzování. Obsahuje přesné verze hlavních knihoven i jejich závislostí z ověřeného prostředí na Macu s Pythonem 3.11.

## Kontrola a importy

```bash
.venv/bin/python -m benchmarks.check_environment
```

Kontrola vypíše verze a commit, načte deset řádků každého datového souboru a zkusí grafovou vrstvu. Na malé syntetické grafové ukázce ověří také loss obou fází CARec a tvar skóre na CPU. Dostupné MPS vyzkouší pro samotnou grafovou vrstvu. Nepočítá metriky, negeneruje embeddingy a nespouští preprocessing celého datasetu. Bez dat lze použít `--skip-data`, pro jiné umístění dat `--data-dir`.

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

Musíme připravit pevný split, mapování ID, cache encoderů a PCA. Původní preprocessing má natvrdo CUDA zařízení a očekává gzip vstupy; zatím ho automaticky nespouštíme. RecBole má vlastní volbu CPU/CUDA, takže notebookové `DEVICE` samo nenastaví MPS pro jeho Trainer. Funkční import ani malý výpočet nejsou důkazem hotové MPS podpory celého CARec tréninku.

Cold evaluator a přechod mezi dvěma trénovacími fázemi potřebují ověřit před měřením. Přehled známých míst je v [průvodci CARec](docs/ai/CAREC_OVERVIEW_CS.md). U PyTorche 2.6 je také třeba ověřit načítání checkpointů starého RecBole. Dosavadní kontrola checkpointy nepoužívá.

Upstream je v samostatném lokálním checkoutu bez našich úprav. [Veřejný repozitář CARec](https://github.com/ChenMetanoia/CARec) v kontrolovaném commitu nemá zjevný licenční soubor. Použití v tomto pracovním prostředí uživatel výslovně zadal 2026-10-17; podmínky dalšího zveřejnění převzatého kódu zůstávají neověřené.
