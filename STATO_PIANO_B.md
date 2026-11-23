# Stato del Piano B

Ultimo aggiornamento: 2026-11-20 (sessione 01: pipeline completa su FACED per Kaggle, Proposta 1)

## Obiettivo

**Perimetro attuale (decisione 2026-11-21):** l'unico dataset è FACED e l'unico foundation model è CBraMod.
- MUSIN-G, NMED-T, EREMUS, SEED e REVE sono un'eventuale espansione successiva.
- Nel codice restano solo come stub e hook dormienti, che nessuna coda usa: `DatasetAdapter`, `stimulus_holdout_plans`, stub REVE.

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

## Risultati parziali (notebook A, prima sessione, 2026-11-21)

BA a livello di finestra, media su 23 soggetti di test, 9 classi; caso = 0.111.

| Modello | split FM | LCO viste | LCO non viste |
|---|---|---|---|
| log-power + LR | 0.200 | 0.213 | 0.131 |
| DE + asimmetrie + LR | 0.185 | 0.196 | 0.120 |
| DE + asimmetrie + z-score per soggetto | 0.232 | 0.270 | 0.145 |
| EEGNet + EA (5 seed) | 0.229 | 0.260 | 0.121 |
| EEGNet senza EA (1 seed) | 0.145 | – | – |
| TSception + EA (5 seed) | 0.212 | 0.247 | 0.130 |
| TSception + EA + BN-α sessione (n0 da validation) | 0.235 | 0.267 | 0.139 |

Letture:
- Sulle clip non viste tutti i modelli sono vicini al caso (0.12–0.145): quasi tutto il vantaggio sopra il caso nello split FM dipende dalla clip.
- **Correzione del Δstim (decisione 2026-11-21).** La formula del prompt, Δraw − Δperm, è distorta. Con label permutate (3 clip per classe), una clip non vista prende la label delle clip viste simili con probabilità 2/27 = 0.074 < 1/9: nei dati BA_unseen^perm = 0.076–0.079. Per questo il Δcorr del prompt è negativo per costruzione (da −0.03 a −0.04).
  - Nuova correzione principale, `corr_chance`: Δraw − (BA_seen^perm − 1/K). In più si riportano la memorizzazione (BA_seen^perm − 1/K) e il guadagno specifico dell'emozione sulle clip viste (BA_seen − BA_seen^perm).
  - Feature: `corr_chance` ≈ 0 (da −0.008 a 0.000). Il divario viste/non viste è interamente spiegato dalla memorizzazione della clip (0.076–0.130).
  - Il guadagno specifico dell'emozione sulle viste è +0.008/+0.012 (n.s.) e +0.029 con z-score per soggetto (p = 0.004).
  - Da confrontare con la definizione esatta di Xu & Plataniotis.
- Clip non viste contro il caso: log-power +0.020 (p 0.02), DE +0.009 (n.s.), DE con z-score +0.034 (p 0.006), EEGNet+EA +0.010 (n.s.), TSception+EA +0.018 (p 0.01).
- BN-α (sessione) contro base: TSception +0.022 (p 1e-5), sulle non viste +0.009 (p 0.004); EEGNet +0.011 (p 0.001), sulle non viste +0.001 (n.s.).
- EA: +8 punti su EEGNet (da confermare con 5 seed). BN-α: +2.3 punti su TSception, +1.1 su EEGNet; n0 scelto quasi sempre 8.
- Le reti compatte (0.21–0.23) restano molto sotto l'EEGNet riprodotto da CBraMod (0.409) sullo stesso split. Il Gate 1 dirà se il problema sono i dati o la ricetta.
- Optuna: TSception con finestre da 6 s e batch 32 (val 0.239); EEGNet con finestre da 4 s (val 0.273).

Coda: 63/156 job in 11 h (Optuna circa 6 h per worker). Ultima sessione: TSception circa 7 min/job, EEGNet circa 8 min/job. Le stime del micro-benchmark erano 2–3 volte troppo alte; ora la guardia del tempo usa i minuti effettivi dei job completati.

Mancano: TSception senza EA, EEGNet senza EA (4 seed), ablazione EMG, permutazioni del Δstim per TSception, sonda della clip, valenza binaria con TSception.

## Risultati notebook A, seconda sessione (2026-11-22)

Coda: 178/208 job; **completata (208/208) con la ripresa del 2026-11-23**.
- **Bug trovato e corretto.** La cache filtrata 4–30 Hz conteneva `inf`: il filtraggio dei campioni già saturati di sub004/023/030 superava il range del float16.
  - I 5 job "4–30 Hz" sono finiti con loss NaN e sono stati invalidati (`INVALIDATED` in `experiments/queue.py`): si rieseguono e le loro righe vengono scartate.
  - I 30 job "4–30 Hz + rimozione canali" sono falliti in modo esplicito.
  - Ora i valori filtrati sono saturati prima del cast; dati o loss non finiti fanno fallire il job.
- **Allineamento (H2):** TSception +0.084 con EA (p 5e-7); EEGNet +0.098. Senza EA entrambi sono vicini al caso (0.128–0.131).
- **Scomposizione** (split FM, media sui 3 fold di clip, 23 soggetti):

  | Modello | Totale | Quantità di dati | Δstim^size | Δstim^size corretto | Indice appaiato corretto | Differenza size − appaiato |
  |---|---|---|---|---|---|---|
  | TSception + EA | +0.084 | +0.016 (p 0.001) | +0.068 | −0.019 (p 0.016) | −0.025 | +0.006 (n.s.) |
  | DE + z-score | +0.090 | +0.001 (n.s.) | +0.088 | −0.015 (n.s.) | −0.005 | −0.010 (n.s.) |

  - Circa l'80–100% del calo dipende dalla novità dello stimolo, non dalla quantità di dati.
  - Il Δstim^size corretto è ≤ 0. Sulle clip T_k viste da altri soggetti, il modello con label vere fa quanto quello con label permutate (TSception 0.197 contro 0.197): il vantaggio di aver visto lo stimolo è interamente memorizzazione dell'identità della clip. Il valore negativo equivale a −(BA_unseen − 1/K), cioè alla piccola generalizzazione genuina sulle clip non viste.
  - **L'ipotesi pre-registrata H3 ("> 0") non regge: il segno è opposto. Va interpretata così, non riformulata a posteriori.**
  - Indice appaiato e stima a parità di dati concordano: la diversità degli stimoli in training non conta.
- **EMG (completo dopo la ripresa del 2026-11-23, TSception + EA, 5 seed):** BA piena 0.212.
  - 4–30 Hz: 0.211;
  - 4–30 Hz senza canali di bordo: 0.210 (ΔEMG = +0.003);
  - 4–30 Hz senza 8 canali casuali: 0.214 (controllo −0.002);
  - H4 non significativa (+0.004, p 0.27).

  Per classe: con 4–30 Hz e senza bordo il recall di amusement scende da 0.32 a 0.27, quello di disgust sale da 0.29 a 0.37. Nessuna dipendenza globale dall'EMG per TSception; resta da verificare su CBraMod.
- **Sonda della clip (TSception):** accuratezza sulla clip 0.105 (caso 0.036); la BA della categoria dedotta dalla clip (0.209) è quasi uguale a quella del modello (0.217).
- **Valenza binaria:** TSception 0.556, DE 0.543 (caso 0.5).

## Valenza binaria su stimoli nuovi (notebook A, 2026-11-24)

58 job, nessun fallimento. Caso = 0.5; 23 soggetti di test.

| | Clip condivise | LCO viste | LCO non viste (finestra / trial) | A parità di dati, su T_k |
|---|---|---|---|---|
| TSception + EA (3 seed) | 0.556 (p 2e-4) | 0.586 | **0.515 (p 0.27) / 0.499** | 0.567 |
| DE + asimmetrie | 0.543 (p 0.007) | 0.553 | **0.509 (p 0.60) / 0.511** | 0.542 |

- **Su clip non viste la valenza binaria è al caso** per entrambi i modelli. Per TSception per fold di clip: 0.543 / 0.480 / 0.521. Solo il fold 0 è sopra il caso nel test di permutazione.
- Scomposizione:
  - TSception: totale +0.041 (p 0.004) = quantità di dati −0.011 (n.s.) + Δstim^size +0.052 (p 5e-4); Δstim^size corretto +0.022 (n.s.); indice appaiato corretto +0.008 (n.s.).
  - DE: totale +0.034 = dati +0.001 + Δstim^size +0.033; corretto −0.011 (n.s.).
- Memorizzazione della clip (permutate viste − caso): TSception 0.063, DE 0.053. Spiega il vantaggio sopra il caso sulle clip condivise.
- **Conclusione:** anche per la valenza binaria il risultato "sopra il caso" con clip condivise è memorizzazione dello stimolo; sulle clip nuove non c'è segnale rilevabile con 23 soggetti.

## Risultati notebook B (2026-11-23)

- **Gate 1 passato** con AMP fp16 sulle T4: BA media per soggetto 0.558 (trial 0.751).
  - Rispetto a TSception+EA: +0.346 (p 2e-7, nessun soggetto peggiora).
  - BA su T_k (clip condivise): 0.536 / 0.523 / 0.607.
- **Init casuale con la ricetta esatta: collassa** in tutti e 3 i seed, predicendo sempre "neutral" (BA 0.111).
  - Nella prima epoca la loss media è 10–17 (il caso è ln 9 ≈ 2.2), poi si assesta su 2.2: il training diverge e la rete diventa costante.
  - Su CPU: con l'init kaiming(fan_out) del codice ufficiale le feature del backbone hanno deviazione standard circa 5, contro 1.5 con l'init di default di PyTorch; i logit iniziali sono piccoli in entrambi i casi.
  - Coerente con la issue #22 del repo.
  - **H1 contro questo controllo sarebbe banale**: serve un "da zero" che impari. Diagnostica in coda sul notebook B (1 seed per variante, una modifica alla volta): init di default; warm-up lineare delle LR per 3 epoche; entrambe. La variante che impara si porta a 3 seed per H1. Si riportano sia la ricetta esatta (collasso) sia la variante, dichiarando la deviazione.
- **Optuna CBraMod con LoRA completato:** 13 trial completi + 18 potati. Migliore BA di validation 0.600 con rango 4, lr testa 8.2e-4, lr LoRA 3.1e-4, dropout LoRA 0.27, label smoothing 0.2, warm-up della testa 3 epoche, testa ufficiale. Salvato in `pianob/hpo/best/cbramod_lora_r2.json`: il notebook C lo trova senza collegare l'output del notebook B.

## Checklist delle run (aggiornata il 2026-11-23)

Legenda: [x] fatto · [ ] da fare · (opz.) facoltativo.

**Account 1 (notebook A, TSception ed EEGNet)**
- [x] Baseline su feature, Optuna TSception ed EEGNet, TSception ed EEGNet su split FM e leave-clip-out, BN-α, ablazioni senza EA, ablazione EMG, permutazioni Δstim, sonda della clip, controllo della quantità di dati (9 classi), valenza su split FM. 208 job.
- [x] **Valenza binaria su stimoli nuovi** (58 job, completati il 2026-11-24):
  - TSception su leave-clip-out (3 seed) e a parità di dati (3 seed × 2 estrazioni);
  - permutazioni per entrambe le correzioni;
  - DE su leave-clip-out e a parità di dati, più la rivalutazione su T_k.

**Account 2 (notebook B, CBraMod)**
- [x] Gate 1: ricetta ufficiale su split FM, 3 seed → **passato** (0.558).
- [x] Init casuale, stessa ricetta, 3 seed → collasso (vedi sopra).
- [x] Optuna CBraMod con LoRA, 30 trial (31 finiti in una sessione).
- [ ] **Diagnostica dell'init casuale** (3 job, circa 1 h), poi 3 seed della variante che impara (H1).
- (opz.) Optuna sulla ricetta ufficiale (lr e `multi_lr`, 10 trial) se il Gate 1 non passa per poco.

**Account 3 (notebook C)**, 88 job, circa 2–3 sessioni. **Può partire ora**: Gate 1 passato, iperparametri LoRA nel codice.
- [ ] LoRA su split FM (3 seed) e leave-clip-out (5 seed); ricetta ufficiale su leave-clip-out (5 seed).
- [ ] LoRA con EA (split FM e leave-clip-out, 3 seed); LoRA con z-score (3 seed).
- [ ] Controllo della quantità di dati con CBraMod (3 seed + 2 permutati) **prima** dell'ablazione EMG.
- [ ] Ablazione EMG con CBraMod (3 seed, 3 estrazioni di controllo); sonda della clip; permutazioni Δstim (2).

**A rotazione (notebook D, 10 fold sui soggetti)**, dopo la scelta delle config finali:
- [ ] Config finali su `lnso10` (3 seed) e `lnso10_lco` (TSception 3 seed, CBraMod 1 seed, DE 1).
- [ ] `lnso10_sizematched` per le 2 config finali (1 seed × 1 estrazione).
- [ ] Gate di potenza dopo le prime 2 config.
- (opz.) Anticipare la parte TSception e DE sull'account 1 con una coda dedicata (non dipende da CBraMod).

**Facoltativi / condizionati**
- (opz.) EEGNet nel controllo della quantità di dati (circa 3 GPU-h).
- (opz.) EEGNet con la ricetta CBraMod (finestre da 10 s, niente EA, 50 epoche, selezione sul kappa), per il confronto con lo 0.409 riportato da CBraMod. Ha senso se il Gate 1 passa.
- (opz.) Variante di sensibilità con clipping robusto delle ampiezze (sub003/004/023/030/056/064/120).
- (opz.) Tuning annidato per fold di TSception (15 trial per fold esterno).
- (opz.) Arousal binario: servono i self-report dai dati grezzi (file da individuare su Synapse).
- Fuori perimetro per ora: altri dataset, REVE, Proposte 2–3 e preprocessing dai raw (sessioni successive).

## Prossimi passi

1. Smoke test su Kaggle con i dati veri. Controllare canali, ampiezze, mappa delle label e pesi.
2. In parallelo:
   - account 1 → notebook A (baseline e TSception);
   - account 2 → notebook B (**Gate 1**).
3. Gate 1 passato → account 3 → notebook C (serve l'Optuna di LoRA dal notebook B).
4. Gate 2: tabelle complete su `split_fm`, `split_fm_lco` e `split_fm_sizematched` con `tools/aggregate.py`, inclusa la scomposizione quantità di dati / novità dello stimolo.
5. Scegliere le config finali → rigenerare `queues/nbD_lnso10.json` → `lnso10` / `lnso10_lco` → gate di potenza.
6. Sessione successiva:
   - preprocessing dai raw (Picard + ICLabel) per l'analisi EMG estesa e per le clip intere;
   - Proposta 2 (ISC/MIL);
   - Proposta 3 (prior di normalizzazione).

## Gate

| Gate | Criterio | Esito |
|---|---|---|
| Gate 1, riproduzione | `cbramod_ft_official` su `split_fm`, 3 seed, BA pooled a livello di finestra entro circa 2 punti da 0.5509 | **PASSATO (2026-11-23)**: BA pooled 0.5579 (seed 0.5613 / 0.5546 / 0.5578), κ 0.500, F1w 0.563 (paper 0.5509 / 0.5041 / 0.5618) |
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
- **Controllo della quantità di dati per il leave-clip-out (decisione 2026-11-21).**
  - *Motivo.* Il leave-clip-out addestra su meno trial (18–19 clip su 28 per soggetto; su MUSIN-G sarebbero 9 brani su 12). Una parte del calo rispetto alle clip condivise può quindi dipendere dalla riduzione dei dati e non dalla novità dello stimolo.
  - *Protocollo `split_fm_sizematched`* (`lnso10_sizematched` solo per le 2 config finali). Per ogni fold di clip k:
    - stessi soggetti di training, validation e test del leave-clip-out;
    - ogni soggetto di training usa un sottoinsieme casuale di clip, diverso per soggetto, con la stessa numerosità e composizione per classe del fold k: 2 clip per emozione + le neutre previste;
    - le clip T_k sono viste da circa 2/3 dei soggetti di training;
    - la validation è identica al leave-clip-out (clip di training del fold, senza T_k);
    - il test è sugli stessi esempi della condizione "unseen": stessi soggetti, clip T_k, stesso allineamento (EA stimato sull'intera sessione).
  - *Asserzioni:*
    - finestre di training identiche al fold leave-clip-out;
    - composizione per classe identica per ogni soggetto;
    - ogni clip di T_k vista da almeno un soggetto.
  - *Estrazioni:* seed di sottocampionamento registrato (`draw`); almeno 2 estrazioni per TSception.
  - *Scomposizione per soggetto, con IC bootstrap.* BA_full è il modello split_fm valutato su T_k (condizioni `tk0..2`, ricavate anche dalle predizioni salvate dei job già completati).
    - BA_full − BA_unseen = (BA_full − BA_size) [quantità di dati] + (BA_size − BA_unseen) [Δstim^size];
    - Δstim^size è corretto per permutazione con la stessa correzione "rispetto al caso" dell'indice appaiato: Δstim^size − (BA_size^perm − 1/K);
    - si riporta anche la differenza Δstim^size corretto − indice appaiato corretto, che misura l'effetto della diversità degli stimoli in training. Va riportata anche se le due stime divergono.
  - *Ipotesi H3 di Holm:* diventa "Δstim^size > 0, corretto per permutazione". L'indice appaiato resta come stima di robustezza.
  - *Budget:* circa 10 GPU-h su T4.
    - TSception: 5 seed × 2 estrazioni × 3 fold, più 5 modelli permutati × 3 fold (circa 5 h);
    - CBraMod LoRA: 3 seed × 1 estrazione × 3 fold, più 2 permutati × 3 fold (circa 5 h);
    - DE con z-score: 2 estrazioni con 50 permutazioni interne, su CPU.

    BN-α non si calcola sui job sizematched.
  - *Dataset musicali:* stessa logica predisposta in `DatasetAdapter.stimulus_holdout_plans` (MUSIN-G 9/12, NMED-T 8/10), valutata sugli stessi brani tenuti fuori.

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
  - sub120 (soggetto di test): picco fino a ~6900 µV, sotto la soglia del float16;
  - non saturati ma oltre 2000 µV: sub003 (~10 900), sub056 (~24 200), sub064 (~43 100), tutti di training.

  Nessun soggetto di validation (sub080-099) supera 2000 µV.

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
