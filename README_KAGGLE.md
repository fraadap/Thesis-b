# Piano B su Kaggle — istruzioni passo-passo

Questa guida è pensata per chi non ha mai usato Kaggle. Le voci marcate **[DA VERIFICARE]** dipendono dall'interfaccia o dalle impostazioni del tuo account e vanno controllate la prima volta.

Panoramica:

1. **Sul tuo PC (una volta):** scarichi FACED processed, lo impacchetti, scarichi i pesi CBraMod e costruisci due zip.
2. **Su Kaggle (una volta):** carichi i due zip come Dataset privati e li condividi con gli altri due account.
3. **Su ogni account:** importi il notebook, aggiungi i due Dataset come input, scegli la GPU e lanci *Save & Run All*.
4. **A fine run:** copi in chat il riepilogo stampato e scarichi `results_*.zip`.
5. **Se la coda non è finita:** rilanci collegando l'output della versione precedente. Il codice riprende da dove si era fermato.

---

## A. Preparazione dei dati (sul tuo PC, una volta)

Servono Python 3.10 o superiore e i pacchetti `numpy scipy torch nbformat`. Basta la CPU.

```bash
pip install numpy scipy torch nbformat synapseclient
```

### A1. Scarica FACED processed da Synapse

1. Crea un account su <https://www.synapse.org>. Accetta le condizioni d'uso del dataset FACED (progetto `syn50614194`).
2. Crea un *Personal Access Token*: profilo → *Account Settings* → *Personal Access Tokens*, con permessi di lettura (view, download).
3. Scarica la cartella `Processed_data` (123 file `sub000.pkl` … `sub122.pkl`, circa 3 GB). Sul web: entra nella cartella e usa *Download Options*. Da riga di comando:

```bash
synapse login --authToken IL_TUO_TOKEN
```

```bash
synapse get -r syn50615881 --downloadLocation ./FACED/Processed_data
```

`syn50615881` è l'ID della cartella `Processed_data` indicato dalla documentazione di TorchEEG **[DA VERIFICARE]**. Se non funziona, apri il progetto `syn50614194` sul web e copia l'ID della cartella `Processed_data`.

### A2. Impacchetta (ricampiona a 200 Hz, salva in float16)

```bash
python tools/pack_faced.py --src ./FACED/Processed_data --out dist/faced
```

Dura qualche minuto e produce `dist/faced/faced_200hz_f16.npy` (circa 1.3 GB), `meta.json` e `checksums.json`. Lo script si ferma con un errore chiaro se un file non ha shape `(28, 32, 7500)`. Se manca un soggetto lo scrive in `meta.json`. Il ricampionamento di default è lo stesso dello script ufficiale di CBraMod (FFT); con `--resample poly` usi `resample_poly(4, 5)`.

### A3. Scarica i pesi CBraMod

```bash
python tools/fetch_cbramod_weights.py --out dist/weights
```

Se il download automatico fallisce, scarica a mano `pretrained_weights.pth` da <https://huggingface.co/weighting666/CBraMod> (il nome del file è **[DA VERIFICARE]**) e lancia:

```bash
python tools/fetch_cbramod_weights.py --file ./pretrained_weights.pth --out dist/weights
```

Lo script carica i pesi nel modello in modo *strict* e salva lo sha256 in `cbramod_pretrained.sha256.json`.

### A4. Costruisci gli zip

```bash
python tools/make_zips.py --data dist/faced --weights dist/weights
```

Ottieni:

- `dist/pianob_faced_data.zip` (circa 1.3 GB): dati e pesi. Si carica **una volta sola**.
- `dist/pianob_code.zip` (meno di 1 MB): codice, notebook e code dei job. Si ricarica a ogni modifica del codice.

Quando cambio il codice, ti basta rigenerare il secondo:

```bash
python tools/make_zips.py --code-only
```

L'opzione `--single` produce un unico `pianob_all.zip`. È sconsigliata, perché a ogni correzione del codice dovresti ricaricare 1.3 GB.

---

## B. Su Kaggle (una volta): i due Dataset

### B1. Crea i Dataset privati (con l'account 1)

1. <https://www.kaggle.com> → *Create* → *New Dataset*.
2. Trascina `pianob_faced_data.zip`. Titolo: `pianob-faced-data`. Visibilità: **Private**. → *Create*.
3. Ripeti con `pianob_code.zip` e titolo `pianob-code`.

Kaggle estrae automaticamente gli zip caricati **[DA VERIFICARE]**: controlla nella pagina del Dataset che compaiano le cartelle `faced/` e `weights/`. Il codice cerca i file in modo ricorsivo dentro `/kaggle/input/`, quindi una cartella in più o in meno non è un problema.

### B2. Rendi i Dataset disponibili agli altri due account

Hai due possibilità.

- **Condivisione (consigliata):** nella pagina del Dataset → *Settings* (o l'icona di condivisione) → *Sharing* → aggiungi gli username degli account 2 e 3 come collaboratori in lettura. Gli altri account li trovano in *Add Input* → *Datasets* → *Shared with you* (o *Your Datasets*) **[DA VERIFICARE la voce esatta del menu]**.
- **Caricamento ripetuto:** ripeti il passo B1 su ciascun account. È più lento (1.3 GB per account), ma non dipende dalla condivisione.

### B3. Aggiornare il codice

Pagina del Dataset `pianob-code` → *New Version* → carica il nuovo `pianob_code.zip` → *Create*. I notebook usano sempre l'ultima versione del Dataset collegato. Se un notebook continua a vedere la versione vecchia, nel pannello *Input* del notebook clicca i tre puntini sul Dataset e scegli *Update to latest version*.

---

## C. Creare ed eseguire un notebook

| Notebook | Account | Contenuto |
|---|---|---|
| `nb0_smoke.ipynb` | uno qualsiasi, va bene la CPU | Verifica di dati, pesi e split, poi 1 epoca su 3 soggetti per modello (meno di 15 min) |
| `nbA_baselines_tsception.ipynb` | Account 1 | Baseline su feature, Optuna di TSception ed EEGNet, TSception ed EEGNet, BN-α, ablazione EMG, permutazioni |
| `nbB_cbramod_gate.ipynb` | Account 2 | CBraMod con la ricetta ufficiale (**Gate 1**), CBraMod da zero, Optuna di CBraMod LoRA |
| `nbC_cbramod_main.ipynb` | Account 3 | Solo **dopo il Gate 1** e con l'Optuna di LoRA disponibile: CBraMod su leave-clip-out, EA, EMG, sonda, permutazioni |
| `nbD_lnso10.ipynb` | a rotazione | 10 fold sui soggetti per le config finali |
| `nbE_eegnet.ipynb` | a rotazione | EEGNet e TSception con la metodologia originale, bande |
| `nbF_ricetta_pretraining.ipynb` | Sessione 02 | Optuna della ricetta a due fasi di CBraMod, controllo di sanità, CBraMod a due fasi (pre-addestrato e da zero), pretraining sulle clip non viste. Coda `F` (circa 1-2 sessioni) |
| `nbG_reve.ipynb` | Sessione 02, account dedicato | REVE-Base: Gate R poi tutti i protocolli di CBraMod. Coda `G` (2-3 sessioni). **Internet ON** e Secret `HF_TOKEN` (sezione C2) |

La coda `F_bis` (ablazione della testa) è solo preparata: non c'è un notebook. Si lancia solo se l'aggregatore
scrive "LANCIARE la coda F_bis" (regola A.4); in quel caso basta cambiare `"F"` in `"F_bis"` nel notebook F.

I file `.ipynb` sono dentro `pianob_code.zip`, nella cartella `notebooks/`. Estrai lo zip sul PC per averli a portata di mano.

Per ogni account:

1. *Create* → *New Notebook* → menu *File* → *Import Notebook* → carica il `.ipynb` giusto.
2. Pannello a destra, *Input* → *Add Input* → aggiungi **`pianob-faced-data`** e **`pianob-code`**.
3. Pannello a destra, *Session options* (o *Settings*):
   - *Accelerator*: **GPU T4 x2**. Per `nb0_smoke` va bene anche *None*.
   - *Internet*: off. Non serve.
   - *Persistence*: se l'opzione esiste, *Files only* **[DA VERIFICARE]**.
4. Per l'esecuzione in background: in alto a destra **Save Version** → *Save & Run All (Commit)* → *Save*. La run continua anche a browser chiuso. La trovi in *View Active Events* o nella pagina del notebook, alla voce *Versions*.

Non usare *Run All* in modalità interattiva: la sessione si interrompe se chiudi il browser.

### C2. REVE: token Hugging Face e Internet (solo notebook G)

I pesi di `brain-bzh/reve-base` sono pubblicati con condizioni d'uso (niente redistribuzione, niente
re-identificazione). Per questo non stanno nello zip dei dati: il notebook G li scarica a ogni sessione.
Se il repo è ad accesso controllato (bisogna accettare le condizioni sulla pagina del modello), il download
richiede un token, con il nostro codice come con `AutoModel.from_pretrained`. Senza Secret il notebook prova
comunque il download anonimo e, se Hugging Face risponde 401/403, si ferma con un messaggio chiaro.

1. Su huggingface.co, con il tuo account, apri `brain-bzh/reve-base` e accetta le condizioni d'uso.
2. *Settings* → *Access Tokens* → crea un token di tipo **Read**.
3. Nel notebook G su Kaggle: menu *Add-ons* → *Secrets* → *Add a new secret*, nome **`HF_TOKEN`**, valore = il
   token. Attiva l'interruttore accanto al secret per questo notebook.
4. *Session options* → *Internet*: **on** (serve per il download; se l'opzione è bloccata va verificato il
   telefono dell'account Kaggle).

Il notebook scarica i pesi in `/tmp` (fuori da `/kaggle/working`, quindi non finiscono nell'output che poi
ricarichi come Dataset per la ripresa) e stampa la revisione scaricata (commit). Per fissare una revisione:
`os.environ["PIANOB_REVE_REVISION"] = "<commit>"` prima della cella di download. Prima di partire esegue i
controlli obbligatori (conteggi 6720/1680/1932, forme, 1.807.881 parametri allenabili in LP, pesi strict).

### Limiti di Kaggle

- **Sessione massima di 12 h.** Il codice si ferma da solo verso 11.3 h, stampa il riepilogo e salva tutto. Se una run superasse le 12 h, Kaggle potrebbe non salvare l'output: per questo la guardia del tempo è tarata più bassa.
- **Quota GPU settimanale:** circa 30 h per account **[DA VERIFICARE]** in *Settings* → *Quotas*. T4 x2 conta come una sessione GPU. Il codice usa entrambe le GPU con due worker in parallelo.
- **Output persistente:** `/kaggle/working`, circa 20 GB. I checkpoint e i risultati vanno lì; le cache ricalcolabili vanno in `/tmp`.

### Prima cosa da fare: lo smoke test

Esegui `nb0_smoke` (anche solo con CPU) e incollami l'output. Prima di spendere GPU conferma che shape, canali, ampiezze, mappa delle label e pesi siano a posto.

---

## D. Riprendere una coda interrotta

Quando la guardia del tempo ferma la coda, il notebook stampa `Da fare (N): ...`. Per continuare:

1. Apri il notebook (in modifica) → *Add Input* → scheda *Notebooks* / *Your Work* → seleziona **lo stesso notebook**. Viene collegato l'output della sua ultima versione e compare in `/kaggle/input/<nome-notebook>/`.
2. **Save Version** → *Save & Run All*.

All'avvio il codice copia dall'input i risultati, i job già completati, i checkpoint e i database Optuna (`.db`). Salta i job finiti, riprende quello interrotto dal suo checkpoint e riprende gli studi Optuna da dove erano.

Per riprendere da una sessione più vecchia, nel pannello *Input* scegli la versione dell'output che vuoi usare.

### Passare gli iperparametri Optuna da un account all'altro

Lo studio di CBraMod LoRA gira sull'account 2 (notebook B) ma serve all'account 3 (notebook C). Hai due strade:

- **Collegamento dell'output:** sull'account 3, *Add Input* → output del notebook B dell'account 2. Funziona solo se il notebook B è condiviso con l'account 3: nel notebook B → *Share* → aggiungi l'account 3.
- **Tramite me:** il riepilogo del notebook B contiene `hpo_best`. Incollamelo in chat: lo salvo in `pianob/hpo/best/cbramod_lora_r2.json` e rigeneri `pianob_code.zip`.

Il notebook C si rifiuta di partire se non trova questi iperparametri: non usa valori di default in silenzio.

---

## E. Dove trovare i risultati

- **Riepilogo da incollare in chat:** apri la versione completata del notebook (pagina del notebook → *Versions* → la versione → *Notebook*) e scorri fino all'ultima cella. Copia:
  1. la tabella markdown sotto `## Riepilogo risultati`, insieme alle righe Gate 1, Δstim e ΔEMG;
  2. **tutto** il blocco tra `PIANOB_JSON_BEGIN` e `PIANOB_JSON_END`.
- **File completi:** scheda *Output* della versione → scarica `results_<account>.zip`. Contiene risultati per soggetto, record dei job, studi Optuna, predizioni e log. Se me lo carichi, lo analizzo con `tools/aggregate.py`:

```bash
python tools/aggregate.py results_account1_nbA.zip results_account2_nbB.zip --out analysis/
```

---

## F. Risoluzione dei problemi

| Problema | Cosa fare |
|---|---|
| `CUDA out of memory` | Riguarda soprattutto CBraMod con la testa ufficiale (circa 128M parametri). Dimmelo: riduco `batch_size` (64 → 32) nel registro e rigeneri il codice. Su T4 x2 ogni worker usa una sola GPU da 15 GB. |
| Ti è stata assegnata una P100 invece della T4 | Funziona lo stesso, ma è più lenta in fp16 (niente tensor core) e ha una sola GPU, quindi un solo worker. Il micro-benchmark iniziale ricalcola i tempi e la guardia adatta la coda. |
| `Pacchetto FACED non trovato` / `Codice non trovato` | Manca un input: *Add Input* → `pianob-faced-data` / `pianob-code`. Controlla che dentro ci siano `faced/meta.json` e `pianob/__init__.py` (anche annidati). |
| `pesi CBraMod non trovati` o `I pesi non corrispondono` | Nel Dataset dati deve esserci `weights/cbramod_pretrained.pth` del repo ufficiale, non il re-hosting di braindecode. |
| La sessione si è interrotta | Vedi la sezione D. Se Kaggle ha ucciso la sessione oltre le 12 h senza salvare l'output, si perde solo il job in corso. I job completati sono già nei file della versione precedente, se la versione è stata salvata. |
| Un job è `FALLITO` | Il riepilogo stampa l'errore. Incollamelo: correggo il codice e rigeneri `pianob_code.zip`. Alla sessione successiva i job falliti vengono riprovati. |
| `optuna non disponibile` | Attiva Internet nel notebook e aggiungi una cella `!pip install optuna` in cima. Di solito l'immagine Kaggle lo include già. |

## G. Prova locale senza dati veri

```bash
python tools/build_notebooks.py --flat local_work/flat
```

```bash
PIANOB_SYNTHETIC=1 python local_work/flat/nb0_smoke_flat.py
```

```bash
python -m pytest -q
```
