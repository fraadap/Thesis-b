# Stato del Piano B

Ultimo aggiornamento: 2026-11-26 (notebook D quasi completo; Sessione 02 pronta: ricetta a due fasi di CBraMod, pretraining su clip non viste, REVE-Base)

## Obiettivo

**Perimetro attuale:** l'unico dataset è FACED. Foundation model: CBraMod e, dalla Sessione 02 (2026-11-25), REVE-Base.
- MUSIN-G, NMED-T, EREMUS e SEED sono un'eventuale espansione successiva.
- Nel codice restano solo come stub e hook dormienti, che nessuna coda usa: `DatasetAdapter`, `stimulus_holdout_plans`.

Benchmark del riconoscimento delle emozioni da EEG su FACED (123 soggetti, 28 clip, 9 classi) con hold-out congiunto di soggetti e stimoli, foundation model (CBraMod) e allineamento per soggetto.

Contributo atteso:
- il gap residuo dei foundation model sotto il leave-clip-out (metodo di Xu & Plataniotis, che non valutano i foundation model);
- la dipendenza dall'EMG, misurata con un controllo appaiato.

## Passo corrente

**Sessione 02 in corso** (dettagli nella sezione "Sessione 02" più sotto).
Prossime azioni dell'utente:
1. ricostruire lo zip del codice;
2. notebook F: rilanciarlo con in input l'output della prima sessione (`results_nbF_ricetta_pretraining`).
   Partono solo i 43 job della riselezione e della Parte A ripetuta (circa 6 ore con 2 worker);
3. notebook G (coda `G`) su un account dedicato, dopo aver creato il Secret `HF_TOKEN` e attivato Internet
   (README_KAGGLE.md, sezione C2);
4. notebook D: ✅ completato il 2026-11-27 (360/360).

## Sessione 02 (2026-11-25): ricetta a due fasi, pretraining su clip non viste, REVE-Base

**Notebook F, prima sessione (2026-11-27, codice `d780163`, 50/50 job).**
- **Optuna (31 trial, 16 completati):** vince la testa appiattita (trial 27, validation 0.607). Le teste a pooling restano basse: per canale con standardizzazione per sessione 0.296, globale 0.227.
- **La configurazione vincente collassa.** Con il trial 27 la fase LP (solo testa, lr 2.3e-3) non impara mai: validation 1/9 in tutte le epoche LP. Il fine-tuning a volte recupera, a volte resta su una classe costante. Job collassati: 1/3 su split FM, 4/9 sul leave-clip-out, 2/6 permutati, 12/12 da zero (da zero la loss della LP parte da 14.7). Il trial 27 era l'unico con LP piatta in Optuna e ha vinto con un seed fortunato. I trial 21, 16 e 19 imparano già nella LP (validation 0.23–0.33 alla prima epoca).
- **Rimedio (scelta solo in validation):** riselezione tra i 4 migliori trial con 3 seed ciascuno su split FM; vince la media della BA di validation, e un collasso conta come caso. Poi si ripete la Parte A con la configurazione scelta (job `select__cbramod_twophase` nella coda F). I job a due fasi del primo giro sono invalidati (`INVALIDATED`, con orario di taglio) e l'aggregatore segnala i job collassati.
- Sui job a due fasi **non collassati**, le clip nuove restano al caso (circa 0.110). La regola A.4 provvisoria dice di non lanciare F_bis, da confermare dopo la ripetizione.
- **Parte B (valida, 3 seed per fold, 23 soggetti), leave-clip-out:**
  - ricetta ufficiale pre-addestrata (0.588 / 0.111) contro da zero con warm-up (0.570 / **0.123**): viste +1.9 (p 0.02), nuove −1.2 (IC95 −2.4/−0.0, p 0.065), interazione +3.1 (p 0.003). **S1: p Holm 0.13, non significativa; il segno è contrario al pretraining sulle clip nuove**;
  - stessa ricetta con warm-up, pre-addestrato (0.532 / 0.121) contro da zero (0.570 / 0.123): viste −3.8 (p 7e-5), nuove −0.2 (n.s.);
  - da zero con warm-up, clip nuove contro il caso: +1.2 (p 0.015); pre-addestrato con warm-up: +1.0 (p 0.07).
  - **Lettura:** il pretraining non porta nulla sulle clip nuove. Dove aiuta, aiuta solo sulle clip viste.
- Tempi: la ricetta a due fasi pre-addestrata richiede 13.6 min a job (LP 0.7 min, 3 epoche; FT circa 27 epoche); i job collassati si fermano dopo 5–8 minuti.

Prompt: `PROMPT_PIANOB-02_LPFT_pretraining.md`. Le quattro ipotesi primarie restano invariate. Le nuove analisi
formano una **seconda famiglia** (aggiunta successiva, Holm solo su queste 4); tutto il resto è esplorativo:

| | Ipotesi | Test |
|---|---|---|
| S1 | Il pretraining aiuta sulle clip **non viste** (ricette scelte in validation, regola di H1: `cbramod_ft_official` contro `cbramod_scratch_ft_warmup`) | Wilcoxon appaiato, bilaterale come H1 |
| S2 | CBraMod a due fasi sulle clip non viste supera il caso | Wilcoxon unilaterale contro 1/9 |
| S3 | REVE sulle clip non viste supera il caso | Wilcoxon unilaterale contro 1/9 |
| S4 | REVE dipende dal contenuto fuori 4–30 Hz (banda piena − 4–30 Hz > 0, split FM) | Wilcoxon unilaterale |

**Inventario della Parte B (righe valide negli zip, 23 soggetti):**

| Config | split FM | T_k | LCO viste/non viste | parità di dati |
|---|---|---|---|---|
| `cbramod_ft_official` | 3 seed | sì | 5 seed × 3 fold | no |
| `cbramod_ft_official_warmup` | 3 seed | sì | **no** | no |
| `cbramod_lora_r2` | 3 seed | sì | 5 seed × 3 fold | 3 seed |
| `cbramod_scratch_ft` (collassa) | 3 seed | sì (dalle predizioni) | no | no |
| `cbramod_scratch_ft_warmup` | 3 seed | sì (dalle predizioni) | **no** | no |
| `cbramod_scratch_ft_definit(_warmup)` | 1 seed | sì | no | no |

Confermato: nessuna variante da zero ha righe di leave-clip-out; la coda F le aggiunge (3 seed ciascuna, insieme
al pre-addestrato con warm-up).

**Parte A — CBraMod a due fasi (LP poi FT, Kumar et al. 2022):**
- Teste: `flat_official` (= la testa ufficiale), `pool_channel` (media sui patch per canale → 6400 → 256 → K),
  `pool_global` (media su patch e canali → 200 → 256 → K).
- Standardizzazione per soggetto degli embedding (solo teste a pooling): `session` (tutte le finestre del
  soggetto, senza etichette, trasduttiva come EA), `heldout` (per i soggetti di test solo le clip fuori dal fold di
  test, con asserzione). Statistiche: una volta in LP, all'inizio di ogni epoca FT (train) e prima di ogni
  validation, poi sulla rete finale prima del test.
- Ricetta `twophase`: AdamW wd 0.05, one-cycle per fase, label smoothing e warm-up da Optuna, batch 64, clipping
  1.0 in FT, AMP + GradScaler; selezione sulla BA media per soggetto in validation tra le epoche FT, patience 10.
- Optuna `cbramod_twophase`: 30 trial, spazio del prompt, Hyperband fino a 60 epoche globali (LP + FT).
  `subj_embed_norm` è un parametro condizionale (solo teste a pooling).
- Controllo di sanità nella coda: il primo job su split FM deve avere **BA di validation ≥ 0.20**; altrimenti
  fallisce e i job della Parte A non partono. Si usa la validation, non il test, per non prendere decisioni sul test.
- Ablazione A.4 (coda `F_bis`, 36 job): solo preparata. L'aggregatore stampa la decisione secondo la regola del
  prompt (Wilcoxon unilaterale per soggetto contro 1/9, p < 0.05). Se Optuna sceglie già `flat_official`, la coda
  usa la migliore configurazione con testa a pooling (`best_by_head` nel file dell'Optuna) e salta le due varianti
  della standardizzazione. La coda si rigenera dopo aver copiato `hpo_best/cbramod_twophase.json` in `pianob/hpo/best/`.
- Predizione per trial anche dalla media delle **log-probabilità**: nuova colonna `ba_trial_lp`; `ba_trial`
  (media delle probabilità) resta invariata per non cambiare i risultati esistenti.

**Parte C — REVE-Base dentro `pianob`:**
- Codice vendorizzato dal repo ufficiale (commit `06a7059a`, MIT): encoder, backbone, `ReveClassifier` con
  pooling "no", StableAdamW. Verificato su CPU: uscita **identica** al codice ufficiale con gli stessi pesi
  (differenza 0 in eval, 1e-6 con il rumore sulle posizioni); StableAdamW identico dopo 25 passi.
- Controlli §R10 superati con pesi casuali: 352 token, ingresso della testa 180.736, **1.807.881** parametri
  allenabili in LP, encoder 69.2 M, conteggi 6720/1680/1932. Con i pesi veri li ripete il notebook G prima della coda.
- Ricetta `reve_official`: porting di `dt.py` (LP 20 epoche lr 5e-3, FT 200 epoche lr 1e-4 dagli ultimi pesi di
  LP, warm-up esponenziale 3/5 epoche, ReduceLROnPlateau sull'accuracy, selezione ed early stopping sulla BA pooled
  di validation con patience > 10, mixup, clipping 2.0, fp16). Varianti di §R8: dropout 0.0, GradScaler spento,
  niente LoRA (A, fedeli al codice); le varianti B (`reve_base_gradscaler`, `reve_base_yamldropout`) esistono ma
  non sono in coda: si usano solo se il training è instabile, scegliendo in validation.
- Scala: il loader dà dati/100; il modello moltiplica per 0.1, quindi REVE vede dati/1000 come nel codice.
- **Ordine dei canali:** il codice di REVE (e quello di CBraMod) chiama le ultime due righe `A2, A1`. Per REVE si
  usa quest'ordine (posizioni 3D). `faced_meta.CHANNELS` resta `A1, A2` per non cambiare TSception.
- Rimozione dei canali: per REVE i canali si tolgono davvero, insieme alle posizioni. Anche CBraMod li toglie davvero
  (la testa è dimensionata sul numero di canali), quindi la variante "stesso meccanismo di CBraMod" coincide.
- Pesi: scaricati a ogni sessione con il Secret `HF_TOKEN` in `/tmp` (**deviazione**: il prompt diceva
  `/kaggle/working`; lì finirebbero nell'output che ricarichi come Dataset, cioè una copia redistribuibile).
  Revisione: risolta in un commit al download e annotata nei risultati; da qui Hugging Face non è raggiungibile,
  quindi non l'ho potuta fissare prima. Caricamento strict; pesi mancanti = errore (§R8.9).
- Riferimenti per i confronti: la scomposizione a parità di dati e la memorizzazione con permutati esistono solo
  per CBraMod **LoRA**, quindi lì REVE si confronta con LoRA; altrove con la ricetta ufficiale.
- Differenza nota dai dati del paper: i campioni oltre ±65 000 µV (saturazione del float16 in sub003/004/023/030,
  soggetti di training) sono tagliati a 65 000.

**Code e stime:**

| Coda | Job | Contenuto, in ordine | Stima |
|---|---|---|---|
| `F` | 50 | Optuna (2 shard) → sanità → Parte A (18 + 12) → Parte B (18) | 1–2 sessioni |
| `F_bis` | 36 | ablazione A.4, da lanciare solo se la regola lo dice | circa 1 sessione |
| `G` | 119 | Gate R1 + R2 (6) → LCO (9) → permutati (6) → parità di dati (15) → bande (9) → canali (21) → valenza REVE e CBraMod (24) → R3 (5) → opzionali da zero e LoRA (24) | 3–4 sessioni; blocchi 1–4 nella prima |

**[DA VERIFICARE] al primo job di REVE:** memoria GPU con batch 64 su T4 (su CPU il container ha esaurito la
memoria con batch 64) e minuti per epoca (stime a priori: 45 min per job su split FM, 35 per fold LCO).



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
- **Diagnostica dell'init casuale (2026-11-24, seed 3407).** Una modifica alla volta rispetto alla ricetta esatta:

  | Variante | BA pooled | κ | F1w | Loss 1ª epoca |
  |---|---|---|---|---|
  | Init di default | 0.111 (collasso) | – | – | 5.2 |
  | **Warm-up delle LR (3 epoche)** | **0.5345** | 0.474 | 0.539 | 2.3 |
  | Init di default + warm-up | 0.5246 | 0.464 | 0.530 | 2.2 |

  - La causa del collasso è la learning rate piena dalla prima iterazione, non l'inizializzazione.
  - Il valore 0.534 è coerente con quello del paper senza pretraining (circa 0.523).
  - **Deviazione dichiarata per H1:** confronto a ricetta identica, pre-addestrato con warm-up contro da zero con warm-up (3 seed ciascuno, in coda sul notebook B). In mancanza, pre-addestrato ufficiale contro da zero con warm-up. Il confronto con il collasso resta solo come riferimento.
  - Stima preliminare (3 seed contro 1): +0.023. **Superata dai risultati completi qui sotto.**
- **H1 completo (2026-11-24, 3 seed per braccio, BA media per soggetto su split FM):**

  | Modello | BA finestra | BA trial | κ val (selezione) |
  |---|---|---|---|
  | Pre-addestrato, ricetta ufficiale | 0.558 | 0.751 | 0.562 |
  | Pre-addestrato + warm-up | 0.521 | 0.725 | 0.526 |
  | Da zero + warm-up | 0.550 | 0.734 | 0.528 |
  | Da zero, ricetta ufficiale | 0.111 (collasso) | – | 0.000 |

  - Il warm-up **peggiora** il pre-addestrato: −0.037, p 8e-6, peggiora il 91% dei soggetti. Con il warm-up il fine-tuning si ferma prima (epoche migliori 20–31 contro 19–31). Il "confronto a ricetta identica" penalizzerebbe quindi il pre-addestrato (−0.029, p 0.007), così come la ricetta ufficiale penalizza il da-zero (collasso).
  - **Regola adottata per H1:** per ciascun modello la ricetta con la migliore metrica di selezione in validation (mai il test). Pre-addestrato → ufficiale; da zero → warm-up. Implementata in `tools/aggregate.py`.
  - **H1: +0.008 (IC95 da −0.006 a +0.022, p 0.21, p Holm 0.42).** Il pretraining non dà un vantaggio significativo su FACED quando il training da zero è stabile. Il +2.8 del paper dipende probabilmente dalla ricetta usata per il da-zero. La sensibilità alla ricetta (±3–4 punti) supera l'effetto del pretraining.
  - Famiglia primaria (Holm, 4 ipotesi):
    - H1 pretraining n.s. (p Holm 0.42);
    - H2 allineamento confermata (p Holm 2e-6);
    - H3 Δstim^size corretto: segno opposto (−0.019, p Holm 0.049);
    - H4 EMG n.s. (p Holm 0.42).
- **Optuna CBraMod con LoRA completato:** 13 trial completi + 18 potati. Migliore BA di validation 0.600 con rango 4, lr testa 8.2e-4, lr LoRA 3.1e-4, dropout LoRA 0.27, label smoothing 0.2, warm-up della testa 3 epoche, testa ufficiale. Salvato in `pianob/hpo/best/cbramod_lora_r2.json`: il notebook C lo trova senza collegare l'output del notebook B.

Vista d'insieme (atteso contro ottenuto, piano contro deviazioni): `CHECKLIST_PIANO_B.md`.

## Checklist delle run (aggiornata il 2026-11-24)

Legenda: [x] fatto · [ ] da fare · (opz.) facoltativo. Tutte le BA sono a livello di finestra, media per
soggetto sui 23 soggetti di test di split FM (caso: 0.111 a 9 classi, 0.5 binario), salvo dove indicato.

### Sessioni completate

| # | Data | Account / notebook | Job | Esito principale |
|---|---|---|---|---|
| 1 | 21/11 | qualsiasi / smoke test | 8 | Pipeline funzionante sui dati veri; pesi CBraMod caricati, checksum OK. |
| 2 | 21/11 | 1 / A (prima sessione) | 63/156 | Optuna TSception (val 0.239, finestre da 6 s) ed EEGNet (val 0.273). TSception+EA 0.212, EEGNet+EA 0.229, DE 0.185, DE+z-score 0.232. LCO: clip non viste 0.12–0.145. BN-α +2.2 (TSception). |
| 3 | 21–22/11 | 1 / A (ripresa) | 178/208 | Allineamento: +8.4 (TSception), +9.8 (EEGNet); senza EA quasi al caso (0.128–0.131). Scomposizione: dati +1.6, stimolo +6.8, corretto −1.9. Sonda della clip: 10.5% (caso 3.6%). **Bug**: inf in float16 nella cache 4–30 Hz (5 job invalidati, 30 falliti). |
| 4 | 22/11 | 1 / A (codice non aggiornato) | 0 | Run inutile: zip del codice non ricostruito. |
| 5 | 23/11 | 1 / A (ripresa) | 208/208 | EMG: 4–30 Hz senza bordo 0.210 contro piena 0.212; controllo con canali casuali 0.214; H4 n.s. (p 0.27). |
| 6 | 23/11 | 2 / B (senza GPU) | 0 | Sessione partita su CPU, 11 h sprecate. Aggiunto lo stop immediato se manca la GPU. |
| 7 | 23/11 | 2 / B | 8/8 | **Gate 1 passato**: 0.5579 pooled (paper 0.5509), BA per soggetto 0.558, trial 0.751. Da zero con ricetta esatta: collasso (0.111). Optuna LoRA: val 0.600. |
| 8 | 24/11 | 1 / A (valenza) | 266/266 | Valenza binaria: clip condivise 0.556 (TSception), 0.543 (DE); **clip non viste 0.515 e 0.509 (n.s.)**; vantaggio da stimolo visto interamente spiegato da memorizzazione. |
| 9 | 24/11 | 2 / B (diagnostica) | 11/11 | Da zero: impara solo con il warm-up delle LR (0.534). |
| 10 | 24/11 | 2 / B (H1) | 16/16 | Da zero + warm-up 0.550 (3 seed); pre-addestrato + warm-up 0.521. **H1 (ricette scelte in validation): +0.008, p 0.21, n.s.** |

| 11 | 24/11 | 3 / C (2 sessioni) | 88/88 | **CBraMod su clip non viste al caso**: ufficiale 0.111, LoRA 0.112 (viste 0.588 / 0.558). LoRA = ufficiale su split FM (0.558). Memorizzazione della clip 0.445 sopra il caso. EMG: 4–30 Hz −37.6 punti; senza bordo −4.5. **Varianti EA/z-score di CBraMod non valide** (riscalatura dominata dagli artefatti), invalidate e in ripetizione. |
| 12 | 24/11 | 2 / C (ripetizione, con l'output dell'account 3) | 15/15 | **Allineamento su CBraMod ripetuto con la scala robusta: ora impara in tutti i 15 job, ma non aiuta.** EA 0.397 (−16.0, tutti i 23 soggetti peggiorano); z-score 0.540 (−1.7, p 0.048). Clip non viste con EA 0.123 (+1.2 sul caso, p 0.07, n.s.). |
| 13 | 24/11 | 3 / E | 33/33 | **EEGNet originale (Lawhern 2018, finestre da 10 s) + EA: 0.455**, il doppio del nostro EEGNet+EA (0.229) e sopra lo 0.409 riportato da CBraMod. Senza EA 0.257; con il framework di fine-tuning CBraMod 0.222. Senza EA, clip non viste 0.118 (n.s.). |
| 14 | 24–25/11 | 2 / B (ripresa) | 21/21 | **Il crollo di CBraMod viene tutto da sotto 4 Hz**: passa-alto 4 Hz 0.177 (−38.0), passa-basso 30 Hz 0.567 (+1.0, n.s.). Non è EMG. Bordo frontale −1.0, temporale −1.5, casuali −0.1. |
| 15 | 24–25/11 | 3 / E (ripresa) | 57/57 | **EEGNet originale + EA su clip nuove: 0.115 (al caso)**, viste 0.513, memorizzazione 38.0. Finestre da 4 s 0.264: il salto viene dalla finestra. 400 epoche: nessun cambiamento. TSception originale: 0.214 con EA (= nostro), clip nuove **0.143 (+3.2, p 0.001)**. |
| 16 | 25/11 | 2 / E (ripresa) | 15/15 | **EEGNet+EA con solo 0–4 Hz: 0.450 (= banda piena 0.455)**; con 4–47 Hz 0.273 (−18.3, come le finestre da 4 s). Leave-clip-out a 4–47 Hz: clip nuove **0.136 (+2.5 sul caso, p 0.007; +2.1 sulla banda piena, p 0.03)**. |
| 17 | 25/11 | 4 / B (ripresa) | 18/18 | **CBraMod con solo 0–4 Hz: 0.576 (≥ banda piena 0.555, p 0.05)**. Passa-alto 1 Hz 0.259, 2 Hz 0.201, 4 Hz 0.177: **circa 3/4 della caduta sta sotto 1 Hz**. Leave-clip-out a 4–47 Hz: clip nuove 0.119 (+0.8, n.s.). |
| 18 | 25–26/11 | 3 / D (ripresa) | 211/330 | **Gate di potenza passato**: effetto minimo rilevabile 1.5 punti su 123 soggetti. TSception sui 10 fold 0.259; **clip nuove 0.145 (+3.4 sul caso, p 1e-17)**. Scomposizione su 123 soggetti: dati +2.4, stimolo visto corretto −0.4 (IC95 −1.5/+0.7). CBraMod LoRA sui 10 fold 0.670; clip nuove (60 soggetti, 1 seed) 0.122. |
| 19 | 26/11 | 4 / E (ripresa) | 6/6 | Permutati di EEGNet + EA a 4–47 Hz: memorizzazione 22.3 (contro 38.0 a banda piena); divario viste/non viste interamente memorizzazione (corretto −3.7); le clip nuove a 0.136 (+2.5) restano una generalizzazione reale. |
| 20 | 26/11 | rotazione / D (ripresa) | 345/360 | **DE + z-score sulle clip nuove: 0.166 (+5.5 sul caso), meglio di TSception (+2.1, p 4e-6) e di CBraMod (+5.8 su 60 soggetti)**. Permutati di TSception: memorizzazione 15.7, corretto −2.1. DE + z-score: scomposizione corretta −1.7. ΔEMG TSception su 123 soggetti +1.4 (p 0.001). Restano 15 fold di CBraMod (codice vecchio). |
| 21 | 27/11 | S edge 2 / D (ripresa) | 15/15 | **CBraMod LoRA sulle clip nuove con 123 soggetti: viste 0.696, nuove 0.124 (+1.3 sul caso, p 1e-4)**: con piena potenza un residuo minimo sopra il caso, sotto TSception (−2.1) e DE + z-score (−4.2). Solo 2 fold su 30 superano il test di permutazione interno. Notebook D completo. |
| 22 | 26–27/11 | Z edge 3 / F | 50/50 | Optuna a due fasi: vince la testa appiattita (trial 27, 0.607), ma **collassa a classe costante** in 19 job su 33 a due fasi (tutti i 12 da zero). Riselezione su 3 seed in coda. Parte B: pretraining sulle clip nuove −1.2 (p 0.065; S1 p Holm 0.13); da zero con warm-up 0.123 sulle clip nuove. |
| 23 | 27/11 | S edge 2 / G | ~40 job, persi | **Disco di /kaggle/working pieno dopo circa 6 ore**: ogni job REVE salvava il modello finale (284 MB) e i checkpoint (circa 1.1 GB per worker). Poi tutti i job sono falliti e il notebook non ha scritto l'output. Correzione: modello finale salvato solo sotto 10 M parametri; guardia sullo spazio libero (il worker si ferma sotto 4 GB); la ripresa non copia checkpoint parziali né modelli grandi. |

**Risultati del notebook C (2026-11-24), 23 soggetti, caso 0.111:**
- **Clip non viste (risultato centrale):**
  - CBraMod ricetta ufficiale: 0.111 (−0.0005, p 0.94; trial 0.114);
  - CBraMod LoRA: 0.112 (p 0.94);
  - clip viste: 0.588 e 0.558;
  - CBraMod è **sotto** TSception sulle clip non viste (−1.9 punti, p 0.035).
- **Indice appaiato (LoRA, 3 modelli permutati):** con label permutate le clip viste restano a 0.556 (contro 0.558) → memorizzazione 0.445 sopra il caso; guadagno specifico dell'emozione sulle viste +0.003; corretto +0.001 (n.s.).
- **Scomposizione a parità di dati (LoRA, 3 seed + 2 permutati):** BA_full 0.555, BA_size 0.484, BA_unseen 0.112, BA_size^perm 0.461.
  - Quantità di dati +7.1 punti (p 4e-5). È più grande che per TSception, probabilmente perché nel sottocampionamento ogni clip è vista da circa 2/3 dei soggetti e la memorizzazione ne risente.
  - Δstim^size +37.2 punti.
  - **Corretto +2.2 (IC95 da +0.8 a +3.7, p 0.008)**: con il sottocampionamento le label vere danno 2.3 punti in più di quelle permutate sulle clip T_k. Questo vantaggio **non si trasferisce** a clip nuove (BA_unseen = caso): sono feature condivise tra le clip di training della stessa categoria, non emozione generalizzabile.
  - Differenza con l'indice appaiato: +2.1 (p 0.08).
- **LoRA contro ufficiale:** identici su split FM (−0.001); sulle clip viste del leave-clip-out LoRA −3.0 (p 9e-5).
- **EMG (LoRA, 3 seed):**
  - piena 0.555;
  - 4–30 Hz 0.179 (**−37.6**, tutti i soggetti peggiorano), con crollo uniforme su tutte le classi (fear da 0.72 a 0.16, sadness da 0.38 a 0.09…), non specifico di amusement/disgust;
  - senza bordo 0.511 (−4.5, p 2e-5); senza 8 canali casuali −0.3 (n.s.);
  - H4: 4–30 Hz senza casuali contro senza bordo **+2.0 (p 0.007)**.
  - **Interpretazione aperta:** il crollo a 4–30 Hz può venire da < 4 Hz (residui oculari, potenziali lenti agganciati allo stimolo, cioè l'"impronta" della clip) o da > 30 Hz (EMG). Il bordo comprende Fp1/Fp2 (oculari). In coda sul notebook B: solo passa-alto 4 Hz, solo passa-basso 30 Hz, bordo frontale contro temporale.
- **Sonda della clip (embedding medio da 200 dimensioni):** accuratezza 8.4% (caso 3.6%), BA dalla clip 0.156 contro modello 0.555. La media su canali e patch perde l'identità che la testa da 64 000 dimensioni usa: la sonda sottostima. È un limite.
- **CBraMod con EA / z-score: non validi.** In 13 job su 15 la loss resta a 2.22 (caso). Causa probabile: la riscalatura all'RMS dei primi 5 soggetti include sub003/sub004 (artefatti fino a 87 000 µV) e gonfia la scala degli ingressi. Corretto con una scala robusta (mediana delle deviazioni standard per canale su 20 soggetti); job invalidati (`INVALIDATED`) e ripetuti sul notebook B.

**Ripetizione EA / z-score su CBraMod LoRA (sessione 12, scala robusta, 3 seed):**
- Tutti i 15 job imparano (validation 0.45–0.59); il problema di scala è risolto.
- **EA: 0.397 contro 0.557 senza allineamento (−16.0 punti, p 2e-7, peggiorano tutti i 23 soggetti).**
- **z-score per soggetto: 0.540 (−1.7, p 0.048, peggiorano 17 soggetti su 23).**
- Leave-clip-out con EA: viste 0.450, non viste 0.123 (+1.2 sul caso, p 0.07, n.s.; il segnale viene quasi tutto dal fold 0: 0.14 contro 0.12 e 0.11). Rispetto a LoRA senza allineamento sulle non viste: +1.1, p 0.11, n.s.
- **Lettura:** l'allineamento che regge i modelli compatti (+8.4 TSception, +9.8 EEGNet) danneggia il foundation model. EA ruota e mescola i canali: le patch non hanno più la statistica spaziale e spettrale vista nel pre-training, e la testa all_patch_reps lavora canale per canale. Lo z-score toglie solo la scala per canale e costa poco. L'eterogeneità tra soggetti, che EA compensa nei modelli compatti, CBraMod la assorbe già nella rappresentazione. Nessuna delle due varianti cambia il risultato centrale: sulle clip nuove si resta intorno al caso.
**Notebook E: EEGNet con la metodologia originale (sessione 13, 3 seed, 23 soggetti):**

| Config | Finestra | Allineamento | Ricetta | Split FM | Viste / non viste (leave-clip-out) |
|---|---|---|---|---|---|
| EEGNet nostro (Optuna) | 4 s | EA | nostra | 0.229 | 0.260 / 0.121 |
| EEGNet nostro | 4 s | nessuno | nostra | 0.131 | — |
| EEGNet originale | 10 s | nessuno | Lawhern | 0.257 | 0.242 / 0.118 |
| EEGNet, framework CBraMod | 10 s | nessuno | fine-tuning CBraMod | 0.222 | 0.219 / 0.119 |
| **EEGNet originale + EA** | 10 s | EA | Lawhern | **0.455** | ⏳ |
| CBraMod LoRA | 10 s | nessuno | — | 0.557 | 0.558 / 0.112 |

- **Il divario con lo 0.409 non era della ricetta CBraMod:** quella ricetta dà 0.222, meno della ricetta di Lawhern (0.257, +3.6, p 5e-6). Senza allineamento non arriviamo a 0.409; con EA lo superiamo.
- **EA vale +19.8 punti** su EEGNet originale (tutti i soggetti migliorano, p 2e-7), contro i +9.8 del nostro EEGNet. Su CBraMod invece costa 16 punti.
- **EEGNet originale + EA contro il nostro EEGNet+EA: +22.6 (p 2e-7).** Contro TSception+EA: +24.3. Il nostro Optuna cercava finestre solo tra 2, 4 e 6 s (TSception) o 2 e 4 s (EEGNet): **le baseline compatte del benchmark erano sotto-ottimizzate.** Il divario CBraMod − miglior modello compatto scende da circa 34 a **10 punti** (LoRA +10.1, p 5e-7; ricetta ufficiale +10.3, p 3e-5).
- **Il training da 200 epoche non è arrivato a convergenza:** 2 seed su 3 hanno la validation migliore all'ultima epoca (199 e 191).
- **Clip non viste, senza EA:** EEGNet originale 0.118 (+0.7, p 0.27), framework CBraMod 0.119 (+0.8, p 0.22). Divario viste/non viste 12.4 punti.
- **Indice appaiato (EEGNet originale, 2 modelli permutati):** clip viste con label permutate 0.200 contro 0.242 vere → memorizzazione +8.9 sopra il caso (p 3e-5); guadagno specifico dell'emozione sulle viste +4.2; corretto +3.5 (p 3e-4). È lo stesso schema di CBraMod: un vantaggio emotivo reale sulle clip viste che sulle clip nuove non si vede.
- **Domanda aperta:** lo 0.455 con EA regge su clip nuove? Se sì, è il primo modello che generalizza agli stimoli; se no, l'EA amplifica la memorizzazione. In coda sul notebook E (36 job aggiunti).

**Notebook E ripreso (sessione 15) e notebook B ripreso (sessione 14), 3 seed, 23 soggetti:**
- **EEGNet originale + EA non generalizza:** leave-clip-out viste 0.513, non viste 0.115 (+0.4, p 0.32). Con label permutate le viste restano a 0.491 → memorizzazione 38.0 punti; corretto +1.8 (p 0.04). A parità di dati: quantità di dati +3.4, Δstim^size corretto +1.1 (n.s.).
- **Il salto da 0.229 a 0.455 viene dalla finestra:** stessa ricetta con finestre da 4 s 0.264 (−19.1, tutti i soggetti peggiorano); la ricetta di Lawhern vale +3.5 sul nostro EEGNet a 4 s. 400 epoche: 0.455, identico.
- **TSception originale (Ding 2022, 10 s):** senza EA 0.138 (un seed collassa al caso); con EA 0.214, uguale al nostro TSception (0.212). Su clip nuove **0.143, +3.2 sul caso (p 0.001)**, meglio del nostro TSception (+1.3, p 0.02). Viste 0.234, permutate viste 0.231: il divario viste/non viste è memorizzazione (corretto −2.9).
- **Bande, CBraMod LoRA:** solo passa-alto 4 Hz 0.177 (−38.0, come 4–30 Hz); solo passa-basso 30 Hz 0.567 (+1.0, n.s.). **Tutta l'informazione sta sotto 4 Hz; sopra 30 Hz (EMG) non serve.** TSception: −1.7 e +1.4, entrambi n.s.
- **Bordo, CBraMod LoRA:** frontale (Fp1, Fp2, F7, F8) −1.0 (n.s.), temporale (FC5, FC6, T7, T8) −1.5 (p 0.005), 4 canali casuali −0.1. Nessun contributo oculare specifico; i −4.5 senza bordo sono la somma di due effetti piccoli.
- **Lettura d'insieme:** più un modello è accurato sulle clip condivise, meno generalizza a quelle nuove (CBraMod 0.557 → 0.112; EEGNet+EA 0.455 → 0.115; TSception 0.214 → 0.143; DE+z-score 0.232 → 0.145). L'accuratezza alta viene dal contenuto lento (sotto 4 Hz, visibile in finestre lunghe) che identifica la clip. È un'ipotesi da verificare: code aggiunte sotto.

**Notebook E, bande (sessione 16, EEGNet originale + EA, 3 seed):**
- Solo 0–4 Hz: 0.450 contro 0.455 a banda piena (n.s.). Tutta l'accuratezza sulle clip condivise sta sotto 4 Hz, come per CBraMod.
- Solo 4–47 Hz: 0.273 (−18.3, peggiorano 22 soggetti su 23), uguale alle finestre da 4 s (0.264, differenza n.s.). **Il vantaggio delle finestre lunghe è proprio il contenuto lento.**
- Leave-clip-out a 4–47 Hz: viste 0.322, **non viste 0.136 (+2.5 sul caso, p 0.007)**, contro 0.115 a banda piena (+2.1, p 0.03). Tolta l'impronta lenta della clip, la generalizzazione a clip nuove compare, piccola, al livello dei modelli compatti (TSception 0.130, differenza n.s.).
- Resta un divario viste/non viste di 18.6 punti. **Permutati a 4–47 Hz (sessione 19, 2 per fold):** con le label permutate le clip viste arrivano a 0.334 (più delle label vere, 0.322) e le nuove a 0.094. Memorizzazione **22.3 punti** sopra il caso (contro 38.0 a banda piena: togliere il segnale lento ne elimina circa il 40%); corretto rispetto al caso **−3.7 (p 6e-4)**: il divario viste/non viste è tutto memorizzazione. Il +2.5 sulle clip nuove non è toccato da questa correzione (le clip nuove non si possono memorizzare) ed è la sola parte che generalizza.
- Corretto un errore nell'indice: con una variante di banda, `delta_stim` prendeva i modelli permutati a banda piena invece di quelli della stessa banda. Ora prende i permutati della stessa variante; l'aggregatore li riporta per ogni variante che li ha.

**Notebook B, bande (sessione 17, CBraMod LoRA, 3 seed):**
- Solo 0–4 Hz: **0.576**, anzi leggermente sopra la banda piena (+1.9, p 0.05). Tutto quello che CBraMod usa sta sotto 4 Hz.
- Soglia del passa-alto: 1 Hz 0.259 (−29.7), 2 Hz 0.201 (−35.6), 4 Hz 0.177 (−38.0). **Circa il 78% della caduta viene dalla banda 0.05–1 Hz** (i dati FACED processed partono da 0.05 Hz, con ICA oculare e riferimento medio).
- Leave-clip-out a 4–47 Hz: viste 0.186, **non viste 0.119 (+0.8, n.s.)**. A differenza di EEGNet (0.136, +2.5), CBraMod senza contenuto lento non generalizza; la differenza tra i due (−1.6) però non è significativa (p 0.14).
- **Lettura:** l'"impronta" della clip è un segnale molto lento (sotto 1 Hz) dentro la finestra da 10 s: derive, potenziali lenti, residui oculari sfuggiti all'ICA. Ordine di presentazione (verificato sul paper FACED): blocchi di 4 clip della stessa valenza, con l'ordine delle clip nei blocchi e dei 7 blocchi randomizzato tra i soggetti; 5 s di croce di fissazione prima di ogni clip, almeno 30 s di riposo dopo le valutazioni. Una deriva legata alla posizione nella sessione non può quindi identificare la stessa clip in soggetti diversi: il segnale lento che i modelli usano è **agganciato al contenuto della clip** (risposte lente evocate dal video, movimenti oculari residui). Decisione: non si approfondisce oltre.

**Notebook D, 10 fold sui soggetti (sessione 18, 123 soggetti; 211 job su 330, nessun errore):**

| Modello | 10 fold (`lnso10`) | Stessi soggetti di split FM (100–122) | split FM |
|---|---|---|---|
| Classe maggioritaria | 0.111 | — | 0.111 |
| DE + asimmetrie | 0.208 | — | 0.185 |
| DE + z-score per soggetto | 0.287 | — | 0.232 |
| TSception + EA (3 seed) | 0.259 | 0.236 | 0.212 |
| CBraMod LoRA (3 seed, prima sessione) | **0.670** | 0.576 | 0.557 |

- **Gate di potenza passato:** deviazione standard delle differenze appaiate TSception − DE 0.059 → effetto minimo rilevabile **1.5 punti** (soglia 2). Nadeau–Bengio sui 10 fold: +5.1 punti, p 3e-5.
- Con 111 soggetti di training invece di 80 tutti i modelli salgono (TSception +2.3, CBraMod +1.9 sugli stessi 23 soggetti di test). I fold con i soggetti 0–59 danno BA più alte di quelli con i soggetti 84–122 (CBraMod 0.69–0.82 contro 0.57–0.58): l'eterogeneità tra soggetti è grande.
- Limite noto confermato innocuo: i fold 7–9 contengono i soggetti di validation di Optuna (080–099) ma hanno le BA più basse, quindi nessun ottimismo visibile.
- **Clip nuove, TSception, 123 soggetti:** viste 0.281, **non viste 0.145 (+3.4 sul caso, IC95 +2.9/+4.0, p 1e-17, peggiorano 15% dei soggetti)**. La piccola generalizzazione a stimoli nuovi vista su split FM (+1.8, al limite) è reale.
- **Scomposizione a parità di dati, TSception, 123 soggetti:** BA_full 0.259, BA_size 0.235, BA_unseen 0.145, BA_size permutato 0.205. Quantità di dati +2.4 (p 2e-8); stimolo visto +9.0; **corretto per permutazione −0.4 (IC95 −1.5/+0.7, p 0.60)**. Con piena potenza il vantaggio delle clip viste, a parità di dati, è tutto memorizzazione (H3 per TSception: zero con intervallo stretto).
- **CBraMod LoRA sulle clip nuove (60 soggetti, 15 fold su 30, 1 seed, dalla prima sessione con codice vecchio):** viste 0.757, non viste 0.122 (+1.1, p 0.04 unilaterale). Sugli stessi 60 soggetti TSception fa meglio sulle clip nuove (+2.8, p 7e-5). Aggiunti alla coda D i 15 fold mancanti, per avere CBraMod su tutti i 123 soggetti.
- BN-α (TSception): +2.8 sui 10 fold, +0.9 sulle clip nuove. DE + z-score contro DE: +7.9.
**Notebook D, seconda sessione (sessione 20, codice `6bc17f4`, coda terminata: 345 job su 360, nessun errore).** I 15 fold mancanti di CBraMod sul leave-clip-out sono arrivati nella sessione 21 (codice `d780163`): notebook D completo, 360/360.

Clip nuove con 123 soggetti (BA per finestra, media per soggetto; CBraMod su 60 soggetti):

| Modello | Clip viste | Clip nuove | Sopra il caso | Memorizzazione |
|---|---|---|---|---|
| DE + z-score per soggetto (regressione logistica) | 0.323 | **0.166** | **+5.5** (IC95 +4.7/+6.3, p 9e-19) | 17.0 |
| TSception + EA + BN-α (sessione) | 0.310 | 0.154 | +4.3 (p 6e-20) | — |
| TSception + EA (3 seed) | 0.281 | 0.145 | +3.4 (p 1e-17) | 15.7 |
| CBraMod LoRA (1 seed, 123 soggetti) | 0.696 | 0.124 | +1.3 (IC95 +0.7/+2.0, p 1e-4; peggiorano 40% dei soggetti) | — |

- **Su stimoli nuovi la classifica si rovescia.** Il modello più semplice generalizza di più. DE + z-score batte TSception sulle clip nuove (+2.1, IC95 +1.3/+2.9, p 4e-6; +1.2 anche contro TSception con BN-α) e CBraMod (+4.2, IC95 +3.2/+5.1, p 3e-12, 123 soggetti). Anche TSception batte CBraMod sulle clip nuove (+2.1, p 6e-6). Le feature DE partono da 1 Hz (bande 1–4 … 30–45 Hz, varianza su segmenti da 1 s), quindi non vedono l'impronta lenta sotto 1 Hz che CBraMod usa.
- Il permutation test interno di DE + z-score (20 permutazioni per fold) è al minimo possibile (p 0.048) in 25 fold su 30. Il z-score del soggetto di test usa tutte le sue finestre senza label: è trasduttivo come l'EA, senza fughe di label.
- **Il vantaggio delle clip viste resta tutto memorizzazione, per entrambe le famiglie di modelli.** Indice appaiato corretto: TSception −2.1 (p 2e-5), DE + z-score −1.25 (p 6e-4). Scomposizione a parità di dati: TSception −0.3 (IC95 −1.5/+0.8, n.s.), DE + z-score −1.7 (IC95 −2.5/−0.9). La quantità di dati pesa +2.4 per TSception e +0.6 (n.s.) per DE + z-score. Il segno negativo indica che memorizzazione ed emozione non si sommano: con label permutate TSception riconosce già le clip viste a 0.268, contro 0.281 con le label vere.
- **EMG con TSception su 123 soggetti (1 seed, 2 estrazioni casuali):**
  - 4–30 Hz senza 8 canali casuali 0.259, senza bordo 0.246;
  - l'analogo di H4 è +1.4 (IC95 +0.5/+2.3, p 0.001 unilaterale), +1.7 con BN-α;
  - banda piena contro 4–30 Hz senza canali casuali: 0.0, quindi TSception non usa nulla fuori 4–30 Hz;
  - su split FM l'effetto era +0.4 (n.s.): con piena potenza emerge un piccolo contributo dei canali di bordo nella banda 4–30 Hz (esplorativo).

- Correzione al riepilogo: il JSON compatto poteva riportare i test di permutazione dei job invalidati (i record per job non avevano l'orario). Ora ogni record ha l'orario e i vecchi lo prendono dal file `done`; le tabelle per soggetto erano già corrette.

**Ipotesi primarie con i dati attuali (Holm su 4).** H3 e H4 sono sul modello migliore pre-registrato, CBraMod LoRA (`--best auto`):
- H1 pretraining: +0.008, p Holm 0.21, n.s.;
- H2 allineamento (TSception): +0.084, p Holm 2e-6, confermata;
- **H3 Δstim^size corretto > 0 (CBraMod): +0.022, p Holm 0.022, confermata in senso stretto**, ma il vantaggio non si trasferisce a stimoli nuovi (BA_unseen al caso). Con TSception: −0.019;
- **H4 EMG oltre il controllo (CBraMod): +0.020, p Holm 0.022, confermata**, ma con il bordo che include i canali oculari e il crollo a 4–30 Hz da attribuire (verifiche in coda). Con TSception: n.s.

**Ipotesi primarie aggiornate al 2026-11-23 (prima del notebook C):**
- H1 pretraining: +0.008 (p Holm 0.42), n.s.;
- H2 allineamento: +0.084 (p Holm 2e-6), confermata;
- H3 Δstim^size corretto > 0: −0.019 (p Holm 0.049), segno opposto, cioè vantaggio interamente compatibile con memorizzazione della clip;
- H4 EMG oltre il controllo: +0.004 (p Holm 0.42), n.s.

H3 e H4 per ora valgono per TSception; per CBraMod servono i notebook C e D.

### Prossime run

**Account 3, notebook C (CBraMod, 88 job)**: completato il 2026-11-24 (vedi sessione 11).

**Account 2, notebook B (aggiunte del 2026-11-24, 21 job rimasti).** EA/z-score già fatti nella sessione 12. Allegare in input gli output del notebook B e del notebook C: il ripristino unisce gli stati e salta i 15 job completati.

| Blocco | Cosa verificare | Ipotesi testata |
|---|---|---|
| CBraMod LoRA e TSception: solo passa-alto 4 Hz, solo passa-basso 30 Hz (3 seed) | Quale dei due spiega il crollo a 4–30 Hz (0.555 → 0.179)? | L'impronta della clip sta sotto 4 Hz (oculare/lenta) o sopra 30 Hz (EMG)? |
| CBraMod LoRA senza bordo frontale (Fp1, Fp2, F7, F8) / temporale (FC5, FC6, T7, T8) / 4 canali casuali | Quale metà del bordo spiega i −4.5 punti? | Il contributo del bordo è oculare o muscolare? |

| Blocco | Cosa verificare nei risultati | Risultato / ipotesi testata |
|---|---|---|
| LoRA su split FM (3 seed) | BA vicina alla ricetta ufficiale (0.558); parametri allenabili molto inferiori | Una variante efficiente è equivalente al full fine-tuning? (esplorativo) |
| LoRA (5 seed) e ricetta ufficiale (5 seed) su leave-clip-out | BA sulle clip non viste rispetto al caso (0.111) e al TSception (0.130); divario viste/non viste | **Il vantaggio di CBraMod (0.558 contro 0.212) regge su stimoli nuovi?** È il risultato centrale del lavoro. |
| Controllo a parità di dati con CBraMod (3 seed + 2 permutati) | BA_full, BA_size, BA_unseen su T_k; BA_size con label permutate | H3 per il foundation model: il vantaggio da stimolo visto è memorizzazione anche per CBraMod? |
| LoRA con EA e con z-score per soggetto (3 seed) | Differenza appaiata contro LoRA senza allineamento, su clip condivise e non viste | L'allineamento aiuta anche un foundation model? (H2 esteso, esplorativo) |
| Ablazione EMG con CBraMod (3 seed, 3 estrazioni di controllo) | 4–30 Hz senza bordo contro 4–30 Hz senza canali casuali; recall di amusement e disgust | H4 per il foundation model: CBraMod sfrutta l'EMG (alte frequenze, canali di bordo)? |
| Sonda della clip e permutazioni Δstim | Accuratezza della clip dagli embedding; BA ottenuta dalla clip predetta contro BA del modello | Le rappresentazioni di CBraMod codificano l'identità dello stimolo? |

**Account 1, notebook D (10 fold sui soggetti, modelli compatti e feature, 330 job, circa 2 sessioni)**, da lanciare ora.

| Blocco | Cosa verificare | Risultato / ipotesi testata |
|---|---|---|
| `lnso10`: TSception (3 seed), DE, DE+z-score, classe maggioritaria | BA su 123 soggetti; deviazione standard delle differenze appaiate → effetto minimo rilevabile (`power_gate` in `tools/aggregate.py`) | **Gate di potenza:** l'effetto minimo rilevabile scende sotto 2 punti? I risultati di split FM dipendono dai 23 soggetti scelti? |
| `lnso10_lco`: TSception (3 seed) + 1 permutato per fold | BA sulle clip non viste contro il caso, con 123 soggetti | Esiste una generalizzazione a stimoli nuovi, piccola ma reale? Su split FM: +1.8 punti, al limite della rilevabilità. |
| `lnso10_sizematched`: TSception (1 seed) + 1 permutato per fold | Scomposizione per soggetto su 123 soggetti | H3 con piena potenza: la quantità di dati conta davvero poco? Il Δstim^size corretto resta ≤ 0? |
| DE + z-score su leave-clip-out e a parità di dati (20 permutazioni interne) | Stessa scomposizione per la baseline su feature | Il quadro è lo stesso con feature classiche? |
| (opz., in fondo) EMG: 4–30 Hz senza bordo e senza 8 canali casuali (2 estrazioni), 1 seed | ΔEMG contro controllo, con 123 soggetti | H4 con piena potenza: l'effetto minimo rilevabile scende da 1.5 a circa 0.6 punti. |

Limite noto: gli iperparametri vengono da Optuna su split FM, che usa i soggetti 080–099 come validation. Nei fold 7–9 di `lnso10` quegli stessi soggetti sono di test, quindi la stima sui 10 fold è leggermente ottimistica. Rimedio facoltativo: tuning annidato per fold di TSception.

**Aggiunte del 2026-11-25 (dopo le sessioni 14 e 15).** Ricostruire lo zip del codice; in input l'output dell'ultima sessione dello stesso notebook.

| Dove | Blocco | Cosa verificare | Ipotesi testata |
|---|---|---|---|
| Account 2, notebook B (18 job, circa 3.5 h) | CBraMod LoRA su leave-clip-out a 4–47 Hz (3 seed) | Clip non viste contro il caso, viste contro 0.558 | Tolta l'impronta lenta della clip, emerge una generalizzazione? |
| Account 2, notebook B | CBraMod LoRA con solo 0–4 Hz, passa-alto 1 Hz e 2 Hz (3 seed) | BA con solo la banda lenta; dove cade la soglia | Quanta accuratezza sta in 0–4 Hz, e in quale parte |
| Account 3, notebook E (15 job, circa 2 h) | EEGNet originale + EA a 4–47 Hz e 0–4 Hz (3 seed); leave-clip-out a 4–47 Hz (3 seed) | BA contro 0.455; clip non viste | Il +19 delle finestre da 10 s è la stessa impronta lenta? |

**Account 3, notebook E (ripresa, 57 job aggiunti, circa 9 h): completato (sessione 15).** Ricostruire lo zip del codice e allegare in input l'output della prima sessione: i 33 job fatti vengono saltati.

| Blocco | Cosa verificare | Ipotesi testata |
|---|---|---|
| EEGNet originale + EA su leave-clip-out (3 seed) + 2 permutati per fold | BA sulle clip non viste contro il caso; indice appaiato corretto | Lo 0.455 è emozione che si trasferisce o memorizzazione della clip amplificata dall'EA? |
| TSception con la metodologia originale (Ding 2022), finestre da 10 s, con e senza EA (3 seed) | BA contro 0.212 (nostro TSception, Optuna, 6 s) e contro EEGNet originale + EA | Anche TSception era sotto-ottimizzato? |
| TSception originale + EA su leave-clip-out (3 seed) + 2 permutati per fold | Clip non viste contro il caso | Il compatto migliorato generalizza a stimoli nuovi? |
| EEGNet originale + EA a parità di dati (3 seed + 2 permutati) | Scomposizione dati / stimolo; Δstim^size corretto | H3 sul miglior modello compatto |
| EEGNet originale + EA con finestre da 4 s | BA contro 0.455 (10 s) e 0.229 (nostra ricetta, 4 s) | Il salto viene dalla finestra o dalla ricetta? |
| EEGNet originale + EA, 400 epoche | BA contro 0.455 | Quanto si perde a 200 epoche? |

**Account 2: libero.** Il notebook C sta per finire sull'account 3 (decisione 2026-11-24: non si divide). Dopo i suoi risultati: CBraMod sui 10 fold con la configurazione finale scelta.

| Blocco | Cosa verificare | Risultato / ipotesi testata |
|---|---|---|
| CBraMod (config finale) su `lnso10` (3 seed), `lnso10_lco` e `lnso10_sizematched` (1 seed) | Come per TSception | Le conclusioni del foundation model valgono su tutti i 123 soggetti? (coda da definire dopo il notebook C) |

**Facoltativi / condizionati**
- (opz.) EEGNet con la ricetta CBraMod (finestre da 10 s, niente EA, 50 epoche): spiega il divario con lo 0.409 riportato da CBraMod per EEGNet?
- (opz.) EEGNet nel controllo a parità di dati (circa 3 GPU-h).
- (opz.) Clipping robusto delle ampiezze anomale (sub003/004/023/030/056/064/120): le conclusioni sono sensibili agli artefatti?
- (opz.) Tuning annidato per fold di TSception: quanto è ottimistica la stima sui 10 fold?
- (opz.) Arousal binario: servono i self-report dai dati grezzi.
- Fuori perimetro per ora: altri dataset, REVE, Proposte 2–3, preprocessing dai raw.

## Prossimi passi

0. Sessione 02: notebook F (coda `F`), notebook G (coda `G`), poi `tools/aggregate.py` (sezione "Sessione 02":
   seconda famiglia, regola A.4, REVE contro CBraMod, Gate R, tempi).
1. Account 3 → notebook C (risultato centrale: CBraMod su stimoli nuovi).
2. Account 1 → notebook D (10 fold, gate di potenza).
3. Account 2 → dopo i risultati del notebook C, CBraMod sui 10 fold (coda da preparare).
4. Gate 2 con `tools/aggregate.py` su tutti gli zip; poi stesura dei risultati.
5. Sessione successiva: preprocessing dai raw, Proposte 2–3.

## Gate

| Gate | Criterio | Esito |
|---|---|---|
| Gate 1, riproduzione | `cbramod_ft_official` su `split_fm`, 3 seed, BA pooled a livello di finestra entro circa 2 punti da 0.5509 | **PASSATO (2026-11-23)**: BA pooled 0.5579 (seed 0.5613 / 0.5546 / 0.5578), κ 0.500, F1w 0.563 (paper 0.5509 / 0.5041 / 0.5618) |
| Gate 2, pipeline | tutte le config (tranne REVE) su `split_fm` e `split_fm_lco`, con tabelle e indici | da eseguire |
| Gate di potenza | dopo `lnso10` per 2 config: effetto minimo rilevabile ≤ 2 punti? | **PASSATO (2026-11-26)**: 1.5 punti (TSception contro DE, 123 soggetti) |

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

1. **Ordine A1/A2** (canali 31–32). Non è documentato nel dataset; il codice ufficiale di REVE (e quello di CBraMod da cui deriva) usa `A2, A1` (aggiornato il 2026-11-25). Ininfluente per CBraMod; per REVE si usa l'ordine del suo codice; per TSception resta la nostra etichettatura (cambia solo il verso della coppia). I primi 30 canali seguono l'ordine di TorchEEG, che non è stato verificato sul file (i `.pkl` non hanno nomi di canale).
2. **Unità dei `.pkl`** (attese in µV). `pack_faced.py` stampa la std mediana e la scrive in `meta.json`.
3. **ID Synapse della cartella** `Processed_data` (`syn50615881` secondo TorchEEG).
4. **Partizione a 10 fold di Xu & Plataniotis.** Implementata in modo sequenziale (9 × 12 + 15) come nel codice CLISA degli autori FACED (THU-EP), ma non verificata per FACED.
5. **Forma esatta della correzione per permutazione di Xu & Plataniotis.** Implementato Δraw − Δperm come da prompt. Va verificato sul paper che sia la stessa definizione, perché con forte memorizzazione Δperm può superare Δraw.
6. **Nome del file dei pesi** su Hugging Face (`weighting666/CBraMod/pretrained_weights.pth`). Il re-hosting di braindecode probabilmente ha chiavi diverse e il loader lo rifiuta.
7. **Interfaccia Kaggle:** estrazione automatica degli zip nei Dataset, voci di menu per la condivisione, quota GPU settimanale dell'account.
8. **Self-report** (valence, arousal, liking, familiarity): non sono nei `.pkl` processed. Andranno presi dai raw.
9. **Issue #22 di CBraMod:** l'init casuale su FACED dà risultati diversi dal paper. Riportare `cbramod_scratch_ft` con la sua ricetta esatta (identica a quella ufficiale).

## Idee archiviate

In `IDEE_ARCHIVIATE.md`: da riprendere solo a progetto finito, anche dopo l'eventuale espansione a nuovi dataset e modelli. Non si pianificano run.

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
