# Checklist del Piano B: atteso contro ottenuto, piano contro deviazioni

Aggiornata il 2026-11-24, dopo il notebook C. Fonte dei numeri: `tools/aggregate.py` sugli zip degli account 1–3.

**Convenzioni:**
- BA = balanced accuracy a livello di finestra, media per soggetto, 23 soggetti di test di split FM;
- caso = 0.111 (9 classi) e 0.5 (valenza binaria);
- "clip non viste" = condizione `unseen` del leave-clip-out;
- p = Wilcoxon appaiato per soggetto.

**Legenda dello stato:**
- ✅ fatto come previsto;
- 🔀 fatto con una deviazione (motivata);
- 🆕 aggiunto in corso d'opera;
- ⏳ in coda;
- ⛔ non valido, da ripetere;
- 💤 rinviato o archiviato.

---

## 1. Gate di avanzamento

| Gate | Criterio previsto | Ottenuto | Stato |
|---|---|---|---|
| Gate 1: riproduzione CBraMod | BA pooled entro circa 2 punti da 0.5509 (3 seed) | **0.5579** (0.561 / 0.555 / 0.558); κ 0.500 (paper 0.504); F1w 0.563 (paper 0.562) | ✅ passato |
| Gate 2: pipeline completa su split FM + leave-clip-out | Tutte le config tranne REVE, con tabelle e indici | Tutte eseguite; CBraMod con EA / z-score ripetuto con la scala robusta | ✅ completo |
| Gate di potenza | Dopo `lnso10` per 2 config: effetto minimo rilevabile ≤ 2 punti | Stima preliminare su 23 soggetti: 1.1–3.7 punti; proiezione su 123: 0.5–1.5 punti | ⏳ notebook D |

---

## 2. Dati e preprocessing

| Voce | Piano iniziale | Cosa è successo | Stato |
|---|---|---|---|
| Ricampionamento 250 → 200 Hz | `resample_poly(4, 5)` | Scelto il ricampionamento FFT dello script ufficiale CBraMod, per non introdurre differenze nel Gate 1; `poly` resta un'opzione | 🔀 |
| Scala d'ampiezza | Da verificare | /100 nel loader (verificato sul codice ufficiale) | ✅ |
| Ordine canali e mappa delle label | Da verificare | Verificati su CBraMod e TorchEEG; l'ordine A1/A2 resta non documentato | ✅ (A1/A2 da verificare) |
| Artefatti di ampiezza | Non previsti | sub004/023/030 oltre il limite del float16 (fino a 332 000 µV) → saturazione registrata; altri 4 soggetti oltre 2000 µV | 🆕 |
| Bug float16 nella cache filtrata | — | Filtrando i campioni saturati si superava il float16 → `inf` → 5 job a 4–30 Hz con loss NaN, 30 falliti. Corretto, job invalidati e ripetuti; ora dati o loss non finiti fanno fallire il job | 🆕 corretto |
| Preprocessing dai raw (Picard + ICLabel) | Sessione successiva | Non iniziato | 💤 |

---

## 3. Protocolli di valutazione

| Protocollo | Piano | Stato e note |
|---|---|---|
| `split_fm` (sub000–079 / 080–099 / 100–122) | Confronto con la letteratura | ✅ |
| `split_fm_lco` (3 fold di clip, clip viste / non viste) | Indice di scorciatoia dello stimolo | ✅ |
| `lnso10`, `lnso10_lco` | Claim principali con 123 soggetti | ⏳ notebook D (modelli compatti); CBraMod da decidere |
| **`split_fm_sizematched` / `lnso10_sizematched`** | Non previsti | 🆕 su richiesta: separano quantità di dati e novità dello stimolo, valutati sugli stessi esempi del leave-clip-out |
| Condizioni `tk0..2` su split FM | Non previste | 🆕 BA con clip condivise sulle clip tenute fuori, necessaria alla scomposizione (ricavata anche dalle predizioni salvate dei job vecchi) |
| Correzione per permutazione del Δstim | Δcorr = Δraw − Δperm | 🔀 **Formula distorta.** Con label permutate le clip non viste scendono sotto il caso per costruzione (2/27 = 0.074). Adottata la correzione rispetto al caso: Δraw − (BA_viste^perm − 1/K); il valore originale resta riportato per trasparenza |

---

## 4. Modelli: atteso contro ottenuto (split FM, 9 classi)

| Modello | Atteso (letteratura o piano) | Ottenuto | Commento |
|---|---|---|---|
| Classe maggioritaria | 0.111 | 0.111 | ✅ |
| Log-power + LR | baseline minima | 0.200 | ✅ |
| DE + asimmetrie + LR | FACED paper: 35.2% (protocollo e metrica diversi) | 0.185 | non confrontabile direttamente |
| DE + z-score per soggetto | — | 0.232 | ✅ |
| EEGNet + EA (nostra ricetta, 4 s) | 0.409 (EEGNet riprodotto da CBraMod) | **0.229** | 🔀 sotto-ottimizzato (finestre di Optuna ≤ 6 s) |
| 🆕 EEGNet originale (Lawhern, 10 s), senza EA / framework CBraMod / + EA | 0.409 | 0.257 / 0.222 / **0.455** | 🆕 con EA supera lo 0.409; 400 epoche non cambiano nulla (−0.04) |
| 🆕 EEGNet originale + EA con finestre da 4 s | — | 0.264 | 🆕 **la finestra vale +19.1 punti, la ricetta solo +3.5** |
| 🆕 TSception originale (Ding 2022, 10 s), senza EA / + EA | — | 0.138 / 0.214 | 🆕 nessun guadagno sul nostro TSception (0.212); un seed senza EA collassa |
| TSception + EA | "competitivo" con i foundation model | **0.212** | ✅ non era sotto-ottimizzato: la versione originale a 10 s dà lo stesso (0.214) |
| CBraMod, ricetta ufficiale | 0.5509 | **0.558** (trial 0.751) | ✅ Gate 1 |
| CBraMod LoRA | variante efficiente | 0.558 | ✅ uguale al full fine-tuning |
| CBraMod da zero, ricetta ufficiale | "con cautela" (issue #22); paper 0.523 | **0.111 (collasso)** | 🔀 vedi §5 |
| CBraMod da zero + warm-up delle LR | — | 0.550 | 🆕 |
| REVE-Base | stub | — | 💤 fuori perimetro |
| EEGNet / TSception senza EA | — | 0.131 / 0.128 | 🆕 per H2 |

**Iperparametri (Optuna):**

| Modello | Trial previsti | Trial eseguiti (completi + potati) | Scelte e BA di validation |
|---|---|---|---|
| TSception | 60 | 61 (36 + 25) | finestre da 6 s, batch 32; val 0.239 |
| EEGNet | 40 | 41 (20 + 21) | finestre da 4 s; val 0.273 |
| CBraMod LoRA | 30 | 31 (13 + 18) | rango 4, testa ufficiale; val 0.600 |

Deviazione 🔀: la scelta della testa (ufficiale contro leggera) è stata aggiunta allo spazio di ricerca di LoRA. L'Optuna facoltativo sulla ricetta ufficiale non è servito.

---

## 5. Deviazioni nate dai risultati

| Problema emerso | Strada prevista | Strada presa | Esito |
|---|---|---|---|
| CBraMod da zero collassa sulla classe "neutral" (loss 10–17 nella prima epoca) | Riportare la ricetta esatta | 🆕 Diagnostica una modifica alla volta: init di default (collassa ancora), **warm-up delle LR (impara: 0.534)**, entrambe (0.525) | La causa è la LR piena dalla prima iterazione |
| Il warm-up peggiora il pre-addestrato (0.521 contro 0.558) | H1 a ricetta identica | 🔀 **H1 con la ricetta scelta in validation per ciascun modello**: ufficiale per il pre-addestrato, warm-up per il da-zero. Mai scelta sul test | H1: +0.008, n.s. |
| Sessione dell'account 2 senza GPU (11 h perse) | — | 🆕 La coda si ferma subito se manca la GPU | Risolto |
| Zip del codice non aggiornato (run inutile) | — | 🆕 Versione del codice nella riga di avvio del log | Risolto |
| Stime dei tempi 2–3 volte troppo alte | Micro-benchmark | 🔀 Stime dai minuti effettivi dei job già completati | Risolto |
| CBraMod con EA / z-score fermo al caso (13/15 job) | — | 🆕 Riscalatura resa robusta (prima era dominata dagli artefatti di sub003/004); job invalidati e ripetuti | ✅ ora impara; EA −16.0, z-score −1.7 |
| CBraMod perde 37.6 punti a 4–30 Hz in modo uniforme su tutte le classi | Ablazione EMG = 4–30 Hz + bordo | 🆕 Scomposizione: solo passa-alto 4 Hz, solo passa-basso 30 Hz, bordo frontale contro temporale | ✅ **tutto sotto 4 Hz**: passa-alto 4 Hz −38.0, passa-basso 30 Hz +1.0 (n.s.); non è EMG. ⏳ localizzazione (0–4, 1, 2 Hz) e leave-clip-out a 4–47 Hz |
| EEGNet originale + EA a 0.455 contro 0.229 del nostro | Baseline compatte con iperparametri da Optuna | 🆕 Lo spazio di Optuna limitava le finestre a 2–6 s: le baseline compatte erano sotto-ottimizzate. Il miglior compatto ora è EEGNet originale + EA; il divario con CBraMod scende da 34 a 10 punti | ✅ su clip nuove 0.115 (al caso); finestre da 4 s 0.264 → il salto viene dalla finestra |
| Il primo blocco di valenza era solo su clip condivise | Valenza come task secondario su split FM | 🆕 Valenza anche su leave-clip-out e a parità di dati | ✅ |
| Perimetro | 5 dataset previsti in prospettiva | 🔀 Solo FACED e CBraMod; gli altri sono un'espansione futura | decisione |

---

## 6. Indici di scorciatoia: atteso contro ottenuto

### 6.1 Divario tra clip viste e non viste (leave-clip-out)

Atteso, da Xu & Plataniotis (encoder contrastivi): calo da 2.2 a 44.4 punti, corretto per permutazione.

| Modello | Viste | Non viste | Divario | Non viste contro il caso | Memorizzazione (permutate viste − caso) | Corretto (rispetto al caso) |
|---|---|---|---|---|---|---|
| Log-power | 0.213 | 0.131 | 8.2 | +2.0 (p 0.02) | 9.0 | −0.8 |
| DE | 0.196 | 0.120 | 7.6 | +0.9 (n.s.) | 7.6 | 0.0 |
| DE + z-score | 0.270 | 0.145 | 12.5 | +3.4 (p 0.006) | 13.0 | −0.5 |
| EEGNet + EA | 0.260 | 0.121 | 13.9 | +1.0 (n.s.) | — | — |
| 🆕 EEGNet originale (10 s, senza EA) | 0.242 | 0.118 | 12.4 | +0.7 (n.s.) | 8.9 | **+3.5 (p 3e-4)** |
| 🆕 EEGNet, framework CBraMod | 0.219 | 0.119 | 10.0 | +0.8 (n.s.) | — | — |
| 🆕 **EEGNet originale + EA** | 0.513 | **0.115** | 39.8 | +0.4 (p 0.32) | **38.0** | +1.8 (p 0.04) |
| 🆕 TSception originale + EA | 0.234 | **0.143** | 9.1 | **+3.2 (p 0.001)** | 12.0 | −2.9 |
| 🆕 EEGNet originale + EA, 4–47 Hz | 0.322 | **0.136** | 18.6 | **+2.5 (p 0.007)** | ⏳ | ⏳ |
| 🆕 CBraMod LoRA, 4–47 Hz | 0.186 | 0.119 | 6.7 | +0.8 (n.s.) | — | — |
| TSception + EA | 0.247 | 0.130 | 11.7 | +1.8 (p 0.01) | 14.2 | −2.5 |
| **CBraMod, ricetta ufficiale** | 0.588 | **0.111** | **47.8** | −0.05 (p 0.94) | — | — |
| **CBraMod LoRA** | 0.558 | **0.112** | **44.6** | +0.1 (p 0.94) | **44.5** | +0.1 |

**Esito:** il divario di CBraMod (45–48 punti) è al limite superiore o oltre il massimo riportato da Xu & Plataniotis. Su clip nuove CBraMod è al caso e 1.9 punti **sotto** TSception (p 0.035). Il vantaggio dei foundation model su FACED è memorizzazione dello stimolo.

### 6.2 Quantità di dati contro novità dello stimolo (🆕)

| Modello | Totale | Quantità di dati | Stimolo (Δstim^size) | Stimolo corretto | Indice appaiato corretto |
|---|---|---|---|---|---|
| TSception + EA | 8.4 | 1.6 (p 0.001) | 6.8 | −1.9 (p 0.016) | −2.5 |
| DE + z-score | 9.0 | 0.1 | 8.8 | −1.5 (n.s.) | −0.5 |
| **CBraMod LoRA** | **44.3** | 7.1 (p 4e-5) | 37.2 | **+2.2 (p 0.008)** | +0.1 |
| Valenza, TSception | 4.1 | −1.1 | 5.2 | +2.2 (n.s.) | +0.8 |
| Valenza, DE | 3.4 | 0.1 | 3.3 | −1.1 (n.s.) | −1.0 |

**Esito:**
- La quantità di dati spiega 0–20% del calo; il resto è novità dello stimolo.
- Il +2.2 di CBraMod esiste solo per i video visti in training: su video nuovi CBraMod è al caso.

### 6.3 Ablazione EMG

Atteso: dipendenza dall'EMG per i modelli deep moderni, non quantificata in letteratura.

| | TSception + EA | CBraMod LoRA |
|---|---|---|
| Banda piena | 0.212 | 0.555 |
| 4–30 Hz | 0.211 | **0.179 (−37.6)** |
| Senza bordo | 0.213 | 0.511 (−4.5, p 2e-5) |
| Senza 8 canali casuali | 0.220 | 0.552 (−0.3) |
| 4–30 Hz senza bordo | 0.210 | 0.156 |
| 4–30 Hz senza 8 casuali | 0.214 | 0.179 |
| 🆕 Solo passa-alto 4 Hz (4–47) | 0.195 (−1.7, n.s.) | **0.177 (−38.0)** |
| 🆕 Solo passa-basso 30 Hz (0–30) | 0.227 (+1.4, n.s.) | 0.567 (+1.0, n.s.) |
| 🆕 Solo 0–4 Hz | — | **0.576 (+1.9, p 0.05)** |
| 🆕 Passa-alto 1 Hz / 2 Hz | — | 0.259 (−29.7) / 0.201 (−35.6) |
| 🆕 Senza bordo frontale (Fp1, Fp2, F7, F8) | — | 0.547 (−1.0, n.s.) |
| 🆕 Senza bordo temporale (FC5, FC6, T7, T8) | — | 0.542 (−1.5, p 0.005) |
| 🆕 Senza 4 canali casuali | — | 0.556 (−0.1) |
| H4 (casuali − bordo, a 4–30 Hz) | +0.4 (n.s.) | **+2.0 (p 0.007)** |
| Recall per classe | amusement −5, disgust +8 | crollo uniforme su tutte le classi |

**Esito:**
- TSception non dipende né dalla banda né dal bordo.
- **CBraMod: tutta l'informazione sta sotto 4 Hz.** Togliere sopra 30 Hz (dove sta l'EMG) non costa nulla; togliere sotto 4 Hz costa 38 punti. Il crollo uniforme sulle classi è coerente.
- Il bordo pesa poco e in modo distribuito (frontale −1.0, temporale −1.5): nessuna traccia oculare specifica nei canali Fp1/Fp2.
- Con le clip nuove al caso, il contenuto sotto 4 Hz è specifico della clip (potenziali lenti agganciati al video, movimenti oculari, deriva). ⏳ Leave-clip-out a 4–47 Hz e bande 0–4 / 1–47 / 2–47.
- H4 resta confermata in senso statistico (+2.0), ma **l'etichetta "EMG" è smentita**: l'effetto del bordo a 4–30 Hz è piccolo e non muscolare in senso stretto.

### 6.4 Sonda dell'identità della clip

| Modello | Accuratezza sulla clip (caso 0.036) | BA dalla clip predetta | BA del modello |
|---|---|---|---|
| TSception | 0.105 | 0.209 | 0.217 |
| CBraMod LoRA | 0.084 | 0.156 | 0.555 |

Limite 🔀: per CBraMod la sonda sull'embedding medio (200 dimensioni) perde l'informazione che la testa da 64 000 dimensioni usa, quindi la sottostima.

---

## 7. Test-time adaptation (BN-α) e allineamento

| Voce | Atteso | Ottenuto |
|---|---|---|
| EA sui modelli compatti | aiuto | **+8.4 (TSception), +9.8 (EEGNet)**; senza EA i modelli sono quasi al caso |
| z-score delle feature per soggetto | aiuto | +4.7 (DE) |
| BN-α sulla sessione | aiuto modesto | TSception +2.2 (p 1e-5; +0.9 sulle clip non viste); EEGNet +1.1; n0 scelto quasi sempre 8 |
| BN-α online (prime n finestre / causale) | curva crescente con n | first32 ≈ metà del guadagno; streaming ≈ +1 |
| EA / z-score su CBraMod | aiuto? | **EA −16.0 (tutti i soggetti peggiorano); z-score −1.7 (p 0.048)**; clip non viste con EA 0.123 (n.s.). L'allineamento aiuta i modelli compatti ma non il foundation model |

---

## 8. Ipotesi primarie (Holm su 4)

H3 e H4 sono valutate sul modello migliore, CBraMod LoRA, come previsto dal piano.

| Ipotesi | Attesa | Ottenuto | p Holm | Esito |
|---|---|---|---|---|
| H1: il pretraining aiuta | +2.8 (paper) | +0.8 (ricette scelte in validation) | 0.21 | ❌ non confermata |
| H2: l'allineamento aiuta | sì | +8.4 (TSception) | 2e-6 | ✅ confermata per TSception; **non si estende a CBraMod** (EA −16.0, z-score −1.7; esplorativo) |
| H3: stimolo visto, corretto > 0 | > 0 | +2.2 (CBraMod); −1.9 (TSception) | 0.022 | ✅ in senso statistico, ma **non si trasferisce a stimoli nuovi** |
| H4: bordo/EMG oltre il controllo | > 0 | +2.0 (CBraMod); n.s. (TSception) | 0.022 | ✅ in senso statistico; **ma CBraMod non usa l'EMG** (passa-basso 30 Hz +1.0): usa il contenuto sotto 4 Hz |

---

## 9. Task secondario: valenza binaria

| | Atteso | Clip condivise | Clip non viste |
|---|---|---|---|
| TSception + EA | FACED paper 69.3% (protocollo diverso) | 0.556 (p 2e-4) | **0.515 (n.s.)** |
| DE | — | 0.543 (p 0.007) | **0.509 (n.s.)** |

Anche per la valenza il vantaggio sopra il caso dipende dalle clip condivise.

---

## 10. Run da eseguire

| # | Dove | Cosa | Cosa verificare | Ipotesi o risultato in esame |
|---|---|---|---|---|
| 1 | ✅ Fatto (account 2, notebook C, 15 job) | CBraMod LoRA con EA e con z-score (scala robusta) | Impara; EA −16.0, z-score −1.7 | H2 non si estende al foundation model |
| 2 | ✅ Fatto (account 2, notebook B) | Solo passa-alto 4 Hz e solo passa-basso 30 Hz | CBraMod −38.0 e +1.0 | L'informazione di CBraMod è sotto 4 Hz, non EMG |
| 3 | ✅ Fatto (account 2, notebook B) | Bordo frontale / temporale / 4 casuali | −1.0 / −1.5 / −0.1 | Nessun contributo oculare specifico |
| 8 | ✅ Fatto (account 4, notebook B, 18 job): solo 0–4 Hz 0.576; passa-alto 1 Hz 0.259; clip nuove a 4–47 Hz 0.119 (n.s.) | CBraMod LoRA: leave-clip-out a 4–47 Hz (3 seed); solo 0–4 Hz, passa-alto 1 Hz e 2 Hz (3 seed) | Clip non viste senza contenuto lento; BA con solo 0–4 Hz; dove cade la soglia | Tolta l'impronta lenta della clip, emerge una generalizzazione? Quanto della BA sta in 0–4 Hz? |
| 4 | Account 1, notebook D (330 job, circa 2 sessioni) | TSception, DE, DE+z-score su `lnso10`, `lnso10_lco`, `lnso10_sizematched` (+ permutati); EMG facoltativo | Effetto minimo rilevabile su 123 soggetti; clip non viste contro il caso; scomposizione | Gate di potenza; generalizzazione reale a stimoli nuovi (+1.8 su split FM, al limite) |
| 6 | ✅ Fatto (account 3, notebook E, 33 job) | EEGNet originale, framework CBraMod, originale + EA | 0.257 / 0.222 / **0.455**; clip non viste al caso senza EA | Il divario con lo 0.409 viene da finestra e allineamento, non dalla ricetta CBraMod |
| 9 | ✅ Fatto (notebook E, 15 job): solo 0–4 Hz 0.450 (= piena), 4–47 Hz 0.273; clip nuove a 4–47 Hz 0.136 (+2.5) | EEGNet originale + EA: passa-alto 4 Hz e solo 0–4 Hz (3 seed); leave-clip-out a 4–47 Hz (3 seed) | Il +19 delle finestre da 10 s viene dal contenuto lento? | Anche il miglior compatto usa la stessa impronta lenta di CBraMod? |
| 7 | ✅ Fatto (account 3, notebook E, 57 job) | EEGNet originale + EA: leave-clip-out + 2 permutati, a parità di dati + 2 permutati, finestre da 4 s, 400 epoche. TSception originale (Ding 2022, 10 s) con e senza EA su split FM; con EA anche su leave-clip-out + 2 permutati | Clip non viste contro il caso; scomposizione; effetto della finestra | **Il miglior modello compatto (0.455) generalizza a stimoli nuovi?** H3 sul compatto |
| 5 | Da decidere dopo i punti 1–3 | CBraMod sui 10 fold | Se serve: su clip nuove è già al caso con margine netto | Robustezza del risultato centrale su 123 soggetti |

**Facoltativi (💤 finché non servono):**
- clipping robusto degli artefatti;
- tuning annidato per fold: ottimismo della stima sui 10 fold, dovuto al tuning su sub080–099;
- arousal binario dai self-report: servono i raw.

**Archiviato:** emozione contro retrieval nello spazio latente → `IDEE_ARCHIVIATE.md`.

**Fuori perimetro:** altri dataset, REVE, Proposte 2–3, preprocessing dai raw.

---

## 10b. Incidenti operativi (2026-11-25)

- **Account 1, notebook D:** girava con un pacchetto di codice anteriore a `fe835b3`. Effetti:
  - vecchia coda D, con CBraMod sui 10 fold;
  - iperparametri TSception assenti → tutti i job TSception falliti subito.

  Nessun dato corrotto: i job falliti non scrivono righe, e i CBraMod sui 10 fold completati sono validi.
  Da riprendere con il codice aggiornato e l'output collegato.
- **Account 2:** lanciato il notebook C, già completato dall'account 3 → senza l'output dell'account 3 ripete tutti gli 88 job. Da sostituire con il notebook B (36 job nuovi).

## 11. Messaggio principale emerso finora

Su FACED la BA pubblicata dei foundation model (circa 0.55) e quella dei modelli compatti (circa 0.21) misurano soprattutto il **riconoscimento della clip**:
- su clip mai viste CBraMod è al caso e i modelli compatti superano il caso di 1–3 punti;
- la riduzione dei dati spiega al massimo il 20% del calo;
- il pretraining non dà un vantaggio significativo;
- CBraMod ed EEGNet+EA traggono tutta la loro accuratezza dal contenuto sotto 4 Hz (per CBraMod soprattutto sotto 1 Hz), non dall'EMG; le finestre lunghe servono proprio a vederlo;
- l'ordine delle clip è randomizzato tra i soggetti (paper FACED): il segnale lento è agganciato al contenuto della clip, non alla posizione nella sessione;
- tolto il contenuto sotto 4 Hz, EEGNet+EA generalizza a clip nuove un poco (+2.5), come i modelli compatti;
- **più un modello è accurato sulle clip condivise, meno generalizza a quelle nuove:** CBraMod 0.557 → 0.112, EEGNet originale + EA 0.455 → 0.115, TSception 0.214 → 0.143, DE + z-score 0.232 → 0.145;
- le baseline compatte erano in parte sotto-ottimizzate (EEGNet, per la finestra), ma il quadro su clip nuove non cambia.
