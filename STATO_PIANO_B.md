# Stato del Piano B

Ultimo aggiornamento: 2026-11-20 (sessione 01: pipeline completa su FACED per Kaggle, Proposta 1)

## Obiettivo

Benchmark del riconoscimento delle emozioni da EEG su FACED (123 soggetti, 28 clip, 9 classi) con hold-out congiunto di soggetti e stimoli, foundation model (CBraMod) e allineamento per soggetto.

Contributo atteso:
- il gap residuo dei foundation model sotto il leave-clip-out (metodo di Xu & Plataniotis, che non valutano i foundation model);
- la dipendenza dall'EMG, misurata con un controllo appaiato.

## Passo corrente

**Codice pronto e testato su dati sintetici.** Prossima azione dell'utente:
1. scaricare FACED processed;
2. packing e pesi;
3. zip e upload;
4. `nb0_smoke` su Kaggle, con l'output incollato in chat.

## Prossimi passi

1. Smoke test su Kaggle con i dati veri. Controllare canali, ampiezze, mappa delle label e pesi.
2. In parallelo:
   - account 1 → notebook A (baseline e TSception);
   - account 2 → notebook B (**Gate 1**).
3. Gate 1 passato → account 3 → notebook C (serve l'Optuna di LoRA dal notebook B).
4. Gate 2: tabelle complete su `split_fm` e `split_fm_lco` con `tools/aggregate.py`.
5. Scegliere le config finali → rigenerare `queues/nbD_lnso10.json` → `lnso10` / `lnso10_lco` → gate di potenza.
6. Sessione successiva:
   - preprocessing dai raw (Picard + ICLabel) per l'analisi EMG estesa e per le clip intere;
   - Proposta 2 (ISC/MIL);
   - Proposta 3 (prior di normalizzazione).

## Gate

| Gate | Criterio | Esito |
|---|---|---|
| Gate 1, riproduzione | `cbramod_ft_official` su `split_fm`, 3 seed, BA pooled a livello di finestra entro circa 2 punti da 0.5509 | da eseguire |
| Gate 2, pipeline | tutte le config (tranne REVE) su `split_fm` e `split_fm_lco`, con tabelle e indici | da eseguire |
| Gate di potenza | dopo `lnso10` per 2 config: effetto minimo rilevabile ≤ 2 punti? | da eseguire |

Se il Gate 1 fallisce, controllare in quest'ordine:
- scala /100;
- 200 Hz;
- finestre da 10 s non sovrapposte;
- ultimi 30 s;
- mappa delle label;
- `multi_lr`;
- testa `all_patch_reps`.

Poi provare AMP disattivato (`hparams.amp=False`): la run ufficiale è in fp32.

## Decisioni prese (con motivazione)

**Dati e preprocessing**
- **Ricampionamento 250 → 200 Hz con FFT** (`scipy.signal.resample` sui 30 s), identico allo script ufficiale di CBraMod, per il Gate 1. `resample_poly(4, 5)` resta disponibile come opzione (`--resample poly`). Il prompt chiedeva `resample_poly`: scelto l'ufficiale per non introdurre differenze nella riproduzione.
- **Scala d'ampiezza** ×0.01 nel loader (CBraMod `data/100`, verificato). Il file packed resta in µV. Le feature spettrali si calcolano in µV.
- **Canali:** si usano tutti e 32, mastoidi A1/A2 inclusi, come fa CBraMod (che passa tutti i 32 canali del `.pkl`).

**Modelli**
- **TSception:**
  - usa 28 canali in ordine emisferico (14 coppie simmetriche, A1/A2 inclusi, linea mediana esclusa), come nel codice originale;
  - `num_T` e `num_S` = numero di kernel (spazio Optuna {9, 15, 21} e {6, 9, 15});
  - fs = 200 Hz, kernel temporali 0.5 / 0.25 / 0.125 s.
- **CBraMod:**
  - codice vendorizzato senza modifiche (commit `b9e9610`);
  - teste riscritte senza einops (`flatten` equivalente a `Rearrange`, testato).
- **LoRA:**
  - via `torch.nn.utils.parametrize`, anche su `in_proj_weight` di `nn.MultiheadAttention` (Q|K|V a blocchi) e su `out_proj`, `linear1`, `linear2`;
  - dropout LoRA applicato ai fattori A (variante weight-space);
  - aggiunta allo spazio Optuna la scelta della testa (`all_patch_reps` ufficiale, circa 128M parametri, oppure `pooled_mlp` leggera). Con la testa ufficiale LoRA non è "efficiente".

**Allineamento**
- **EA:**
  - i soggetti di training lo stimano sulle sole clip di training;
  - i soggetti di test sull'intera sessione (trasduttivo, da dichiarare);
  - per CBraMod, dopo EA o z-score, l'RMS viene riportato a quello dei dati grezzi (`align_rescale`), perché il modello pre-addestrato si aspetta la scala /100.
- **BN-α:**
  - le statistiche del soggetto si stimano con BN in train mode, media cumulativa per batch;
  - in streaming ogni finestra è stimata da sola e poi mediata;
  - n0 scelto in validation tra {8, 32, 128} (riga `n0sel`); n0 = 0 = AdaBN;
  - online: statistiche dalle prime n finestre e valutazione su tutte, più la variante causale `stream`.

**Training e selezione**
- **Selezione del modello:**
  - ricetta ufficiale: kappa pooled sulla validation, 50 epoche senza early stopping (come `finetune_trainer.py`);
  - resto: BA media per soggetto sulla validation, patience 10.
- **Loss:** CE pesata (1/n_c) con label smoothing 0.1, tranne la ricetta ufficiale (CE non pesata, LS 0.1).
- **AMP fp16** con GradScaler su T4/P100.

**Controlli e indici**
- **Controllo EMG:**
  - si rimuovono 4 coppie simmetriche non di bordo estratte a caso (8 canali) da {F3/F4, FC1/FC2, C3/C4, CP1/CP2, CP5/CP6, P3/P4, PO3/PO4, O1/O2};
  - A1/A2 e P7/P8 sono esclusi dal pool perché laterali;
  - stessa struttura del set di bordo, compatibile con TSception.
  - L'ipotesi primaria H4 confronta direttamente BA(4–30 Hz, senza 8 casuali) con BA(4–30 Hz, senza bordo).
- **Δstim:** Δcorr = Δraw − Δperm per soggetto, come da prompt. Δperm viene da riaddestramenti con la mappa clip → classe permutata (conteggi per classe preservati).
- **Holm:** solo sulle 4 ipotesi primarie (`tools/aggregate.py`); il resto è esplorativo.

**Budget**
- **`lnso10_lco` di CBraMod con 1 seed** (30 job) per restare nel budget; 3 seed su `lnso10`.

## [DA VERIFICARE] sui dati veri / sulle fonti

1. **Ordine A1/A2** (canali 31–32). Non è documentato; ininfluente per CBraMod, rilevante solo per la coppia TSception. I primi 30 canali seguono l'ordine di TorchEEG, che non è stato verificato sul file (i `.pkl` non hanno nomi di canale).
2. **Unità dei `.pkl`** (attese in µV). `pack_faced.py` stampa la std mediana e la scrive in `meta.json`.
3. **ID Synapse della cartella** `Processed_data` (`syn50615881` secondo TorchEEG).
4. **Partizione a 10 fold di Xu & Plataniotis.** Implementata in modo sequenziale (9 × 12 + 15) come nel codice CLISA degli autori FACED (THU-EP), ma non verificata per FACED.
5. **Forma esatta della correzione per permutazione di Xu & Plataniotis.** Implementato Δraw − Δperm come da prompt. Va verificato sul paper che sia la stessa definizione, perché con forte memorizzazione Δperm può superare Δraw.
6. **Nome del file dei pesi** su Hugging Face (`weighting666/CBraMod/pretrained_weights.pth`). Il re-hosting di braindecode probabilmente ha chiavi diverse e il loader lo rifiuta.
7. **Interfaccia Kaggle:** estrazione automatica degli zip nei Dataset, voci di menu per la condivisione, quota GPU settimanale dell'account.
8. **Self-report** (valence, arousal, liking, familiarity): non sono nei `.pkl` processed. Andranno presi dai raw.
9. **Issue #22 di CBraMod:** l'init casuale su FACED dà risultati diversi dal paper. Riportare `cbramod_scratch_ft` con la sua ricetta esatta (identica a quella ufficiale).

## Problemi aperti

- **Artefatti di grande ampiezza nel processed ufficiale** (packing del 2026-11-21, std mediana 9.1 µV):
  - sub023: canale A2 rotto in quasi tutte le clip (fino a ~327 000 µV);
  - sub030: OZ e PO4 in molte clip (std del soggetto ~5270 µV);
  - sub004: picco breve su OZ nella clip 17;
  - sub120 (soggetto di test): picco fino a ~6900 µV, sotto la soglia del float16.

  Oltre ±65 000 µV i campioni sono saturati (limite del float16) e registrati in `meta.json`. Riguarda solo soggetti di training dello split dei foundation model, quindi l'effetto sul Gate 1 dovrebbe essere trascurabile.
  Non escludiamo soggetti, per restare confrontabili con la letteratura. Da valutare una variante di sensibilità con clipping robusto per canale (per esempio ±k·MAD) nel loader.

- REVE-Base rinviato. C'è solo lo stub: 69M parametri, codice remoto, checkpoint da verificare.
- Tuning annidato per fold (facoltativo, solo TSception) non ancora in coda.
- Le stime dei tempi sono a priori; il micro-benchmark le ricalibra alla prima run.

## Registro delle sessioni

- **Sessione 01 (2026-11-20).** Pipeline completa:
  - packing, split con asserzioni anti-leakage;
  - baseline su feature, EEGNet, TSception, CBraMod (ufficiale, da zero, LoRA, EA/z-score);
  - BN-α, Optuna (SQLite riprendibile, 2 shard), coda con dipendenze e 2 worker;
  - guardia del tempo, checkpoint e ripresa;
  - Δstim, ΔEMG, sonda della clip, statistica, aggregatore;
  - notebook e zip.

  30 test pytest passano su CPU (leakage, LoRA, EA, metriche, end-to-end TSception/CBraMod/feature, Optuna, ripresa).
