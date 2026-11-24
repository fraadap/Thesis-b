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

**Account 2, notebook B (aggiunte del 2026-11-24, 36 job, circa 7 h)**

| Blocco | Cosa verificare | Ipotesi testata |
|---|---|---|
| CBraMod LoRA con EA (split FM + LCO, 3 seed) e con z-score (3 seed), riscalatura robusta | La loss scende sotto 2.2? BA contro LoRA senza allineamento | L'allineamento aiuta anche un foundation model? (H2 esteso) |
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
