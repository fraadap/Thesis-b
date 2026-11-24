# Idee archiviate

Idee valutate ma rinviate a progetto concluso. Non fanno parte delle code né della checklist.

## 1. Emozione contro retrieval della clip nello spazio latente di CBraMod

Proposta il 2026-11-24. Archiviata il 2026-11-24. Si riprende solo a progetto finito, anche dopo l'espansione a nuovi dataset e modelli.

Proposta dell'utente: confrontare le feature/sezioni latenti di CBraMod rilevanti per l'emozione con quelle che permettono il retrieval della clip.

Valutazione critica:
- **Problema principale.** In FACED la label è una funzione deterministica della clip. Ogni direzione che separa le emozioni separa anche gruppi di clip, quindi una correlazione alta tra le due importanze è attesa per costruzione e non dimostra il retrieval.
- **Cosa misurare invece: l'informazione sulla clip *dentro* la stessa emozione**, cioè quella inutile per un riconoscimento genuino. Strumenti:
  1. **Decodifica della clip dentro la categoria:** sonda a 3 vie per emozione (caso 1/3).
  2. **RSA 28×28:** matrice di dissimilarità delle clip, confrontata con il modello "stessa emozione" e con quello "identità".
  3. **Ablazione causale:** si proietta via il sottospazio della clip ortogonale alle categorie. Se c'è memorizzazione, la BA sulle clip viste scende verso quella delle non viste, mentre le non viste non cambiano.
  4. **CKA tra il modello con label vere e quello con label permutate** (puro memorizzatore per costruzione), con due seed dello stesso modello come tetto e il modello leave-clip-out come contrasto.
  5. **Sonda leave-clip-out sugli embedding congelati**, per separare cosa è nel pretraining e cosa nel fine-tuning.
- **Mappe di attribuzione** (saliency, integrated gradients) tra testa emozione e testa clip: solo come supporto. Sono poco fedeli, e la correlazione riflette il backbone condiviso.
- **Prerequisiti:**
  - salvare gli embedding (o i pesi) dei modelli CBraMod: oggi si salvano solo le predizioni;
  - ripetere pochi job con un salvataggio compatto (embedding medi per canale o per clip);
  - verificare se l'ordine di presentazione delle clip in FACED è fisso o a blocchi (deriva temporale confusa con l'identità) [DA VERIFICARE].
- **Test più forte, ma diverso:** usare i self-report per soggetto (variazione dentro la stessa clip). Se il modello predice il rating del singolo oltre la media della clip, c'è affetto genuino. Servono i rating dai dati grezzi.
