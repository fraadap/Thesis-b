# Figure della tesi

Generate da `tools/make_figures.py` (testi delle figure in inglese). BA = balanced accuracy per finestra, media per soggetto; IC al 95% sui soggetti.

## F1 — Clip viste contro clip nuove per modello

File: `fig01_seen_vs_new_clips.pdf`, `fig01_seen_vs_new_clips.png`

Per ogni modello: BA sulle clip viste in training (pieno) e sulle clip nuove (vuoto) nel leave-clip-out; a sinistra split FM (23 soggetti di test), a destra 10 fold sui 123 soggetti. Mostra il rovesciamento della classifica: i modelli più accurati sulle clip viste sono i peggiori sulle nuove.

Numeri principali: `{"REVE (split_fm_lco)": {"seen": 0.455, "new": 0.132}, "CBraMod (split_fm_lco)": {"seen": 0.588, "new": 0.111}, "EEGNet + EA (split_fm_lco)": {"seen": 0.513, "new": 0.115}, "TSception + EA (split_fm_lco)": {"seen": 0.247, "new": 0.13}, "DE + z-score (split_fm_lco)": {"seen": 0.27, "new": 0.145}, "CBraMod (LoRA) (lnso10_lco)": {"seen": 0.696, "new": 0.124}, "TSception + EA (lnso10_lco)": {"seen": 0.281, "new": 0.145}, "DE + z-score (lnso10_lco)": {"seen": 0.323, "new": 0.166}}`

## F2 — Accuratezza con clip condivise contro accuratezza su clip nuove

File: `fig02_shared_vs_new_clip_accuracy.pdf`, `fig02_shared_vs_new_clip_accuracy.png`

Un punto per modello (23 soggetti di test): asse x BA su split FM (clip condivise), asse y BA sulle clip nuove del leave-clip-out; colori per famiglia di modelli. La correlazione tra le due misure riassume il rovesciamento.

Numeri principali: `{"r": -0.633, "DE": {"shared": 0.185, "new": 0.12}, "DE + z-score": {"shared": 0.232, "new": 0.145}, "TSception + EA": {"shared": 0.212, "new": 0.13}, "TSception + EA (10 s)": {"shared": 0.214, "new": 0.143}, "EEGNet": {"shared": 0.257, "new": 0.118}, "EEGNet + EA": {"shared": 0.455, "new": 0.115}, "CBraMod": {"shared": 0.558, "new": 0.111}, "CBraMod (LoRA)": {"shared": 0.557, "new": 0.112}, "REVE": {"shared": 0.423, "new": 0.132}}`

## F3 — Di cosa è fatta l'accuratezza sopra il caso

File: `fig03_accuracy_decomposition.pdf`, `fig03_accuracy_decomposition.png`

Per modello, in punti di BA sopra il caso: memorizzazione della clip (clip viste con label permutate a livello di clip), guadagno emotivo sulle clip viste (label vere meno permutate) e generalizzazione alle clip nuove. Leave-clip-out su split FM, modelli con run permutate.

Numeri principali: `{"DE + z-score": {"memorization": 13.0, "emotion_gain_seen": 2.9, "new_clips_above_chance": 3.4}, "TSception + EA": {"memorization": 14.2, "emotion_gain_seen": -0.6, "new_clips_above_chance": 1.8}, "TSception + EA (10 s)": {"memorization": 12.0, "emotion_gain_seen": 0.2, "new_clips_above_chance": 3.2}, "EEGNet + EA": {"memorization": 38.0, "emotion_gain_seen": 2.2, "new_clips_above_chance": 0.4}, "CBraMod (LoRA)": {"memorization": 44.5, "emotion_gain_seen": 0.3, "new_clips_above_chance": 0.1}, "REVE": {"memorization": 33.1, "emotion_gain_seen": 1.4, "new_clips_above_chance": 2.1}}`

## F4 — L'impronta lenta: curva del passa-alto e bande

File: `fig04_slow_activity_highpass.pdf`, `fig04_slow_activity_highpass.png`

A sinistra BA su split FM (clip condivise) al crescere della soglia del passa-alto; a destra banda piena, solo 0-4 Hz, solo 4-47 Hz e sotto 30 Hz. L'accuratezza dei modelli accurati sta sotto 4 Hz, in gran parte sotto 1 Hz; sopra 30 Hz (EMG) non serve.

Numeri principali: `{"CBraMod (LoRA)": {"highpass_0Hz": 0.557, "highpass_1Hz": 0.259, "highpass_2Hz": 0.201, "highpass_4Hz": 0.177, "base": 0.557, "bp0-4": 0.576, "bp4-47": 0.177, "bp0-30": 0.567}, "EEGNet + EA": {"highpass_0Hz": 0.455, "highpass_4Hz": 0.273, "base": 0.455, "bp0-4": 0.45, "bp4-47": 0.273}, "TSception + EA": {"highpass_0Hz": 0.212, "highpass_4Hz": 0.195, "base": 0.212, "bp4-47": 0.195, "bp0-30": 0.227}}`

## F5 — Clip nuove: distribuzione per soggetto

File: `fig05_new_clips_per_subject.pdf`, `fig05_new_clips_per_subject.png`

BA sulle clip nuove per ciascuno dei 123 soggetti (leave-clip-out con 10 fold di soggetti), con box plot e quota di soggetti sopra il caso.

Numeri principali: `{"DE + z-score": {"mean": 0.166, "frac_above_chance": 0.829, "n": 123}, "TSception + EA": {"mean": 0.145, "frac_above_chance": 0.854, "n": 123}, "CBraMod (LoRA)": {"mean": 0.124, "frac_above_chance": 0.602, "n": 123}}`

## F6 — Forest plot delle ipotesi (prima e seconda famiglia)

File: `fig06_hypotheses_forest_plot.pdf`, `fig06_hypotheses_forest_plot.png`

Effetto di ogni ipotesi pre-registrata in punti di BA con IC al 95%; pieno = significativo dopo Holm. Prima famiglia in blu (H1-H4), seconda in arancione (S1-S4).

Numeri principali: `{"S3_reve_nonviste_vs_caso": {"diff_points": 2.08, "ci95": [1.03, 3.21], "p_holm": 0.004692369811314357}, "S2_cbramod_dueFasi_nonviste_vs_caso": {"diff_points": 0.39, "ci95": [-0.47, 1.29], "p_holm": 0.18476745499062686}, "S1_pretraining_nonviste": {"diff_points": -1.2, "ci95": [-2.36, -0.01], "p_holm": 0.13007259368896484}, "H4_emg_vs_controllo": {"diff_points": 1.99, "ci95": [0.67, 3.23], "p_holm": 0.022356033325195312}, "H3_delta_stim_size_corr": {"diff_points": 2.23, "ci95": [0.81, 3.72], "p_holm": 0.022356033325195312}, "H2_allineamento": {"diff_points": 8.42, "ci95": [6.45, 10.46], "p_holm": 1.9073486328125e-06}, "H1_pretraining": {"diff_points": 0.81, "ci95": [-0.61, 2.21], "p_holm": 0.21228693977949908}}`

## F7 — Pretraining: interazione clip viste x clip nuove

File: `fig07_pretraining_seen_vs_new.pdf`, `fig07_pretraining_seen_vs_new.png`

CBraMod pre-addestrato (ricetta ufficiale) contro da zero (ricetta con warm-up, scelte in validation), leave-clip-out su split FM: differenza appaiata sulle clip viste e sulle nuove.

Numeri principali: `{"seen": 1.89, "new": -1.2, "p_seen": 0.02332472801208496, "p_new": 0.06503629684448242}`

## F8 — Effetto dell'allineamento per soggetto per famiglia di modelli

File: `fig08_alignment_effect.pdf`, `fig08_alignment_effect.png`

Differenza appaiata (allineato meno non allineato) su split FM, con IC al 95%: EA e z-score per soggetto su feature, modelli compatti e CBraMod.

Numeri principali: `{"CBraMod: EA": {"diff_points": -16.01, "p": 2.384185791015625e-07}, "CBraMod: z-score": {"diff_points": -1.72, "p": 0.052229881286621094}, "EEGNet: EA": {"diff_points": 19.8, "p": 2.384185791015625e-07}, "TSception: EA": {"diff_points": 8.42, "p": 4.76837158203125e-07}, "DE: z-score": {"diff_points": 4.68, "p": 0.004046094748723528}}`

## F11 — Heatmap clip x finestra, CBraMod (LoRA)

File: `fig11_heatmap_clip_window_cbramod_lora_r2.pdf`, `fig11_heatmap_clip_window_cbramod_lora_r2.png`

Percentuale di finestre classificate correttamente, media sui soggetti, per ognuna delle 28 clip (righe, per categoria) e delle 3 finestre da 10 s (colonne); a sinistra clip viste in training, a destra clip nuove (lnso10_lco, 30 job). Bianco = caso.

Numeri principali: `{"seen_mean_pct": 69.2, "new_mean_pct": 12.7, "by_window_pct": {"seen": [68.4, 70.8, 68.2], "new": [12.0, 13.0, 12.9]}, "best_new_clips": ["Neutral 3", "Neutral 2", "Neutral 4"]}`

## F11 — Heatmap clip x finestra, TSception + EA

File: `fig11_heatmap_clip_window_tsception_ea.pdf`, `fig11_heatmap_clip_window_tsception_ea.png`

Percentuale di finestre classificate correttamente, media sui soggetti, per ognuna delle 28 clip (righe, per categoria) e delle 5 finestre da 6 s (colonne); a sinistra clip viste in training, a destra clip nuove (lnso10_lco, 90 job). Bianco = caso.

Numeri principali: `{"seen_mean_pct": 28.2, "new_mean_pct": 14.7, "by_window_pct": {"seen": [26.6, 29.8, 30.2, 28.1, 26.4], "new": [14.8, 15.2, 15.4, 14.0, 14.4]}, "best_new_clips": ["Disgust 3", "Disgust 2", "Disgust 1"]}`

## F11 — Heatmap clip x finestra, CBraMod

File: `fig11_heatmap_clip_window_cbramod_ft_official.pdf`, `fig11_heatmap_clip_window_cbramod_ft_official.png`

Percentuale di finestre classificate correttamente, media sui soggetti, per ognuna delle 28 clip (righe, per categoria) e delle 3 finestre da 10 s (colonne); a sinistra clip viste in training, a destra clip nuove (split_fm_lco, 15 job). Bianco = caso.

Numeri principali: `{"seen_mean_pct": 58.8, "new_mean_pct": 11.5, "by_window_pct": {"seen": [57.8, 59.3, 59.2], "new": [10.8, 11.9, 11.7]}, "best_new_clips": ["Sadness 1", "Neutral 3", "Neutral 2"]}`

## F11 — Heatmap clip x finestra, EEGNet + EA

File: `fig11_heatmap_clip_window_eegnet_lawhern_ea.pdf`, `fig11_heatmap_clip_window_eegnet_lawhern_ea.png`

Percentuale di finestre classificate correttamente, media sui soggetti, per ognuna delle 28 clip (righe, per categoria) e delle 3 finestre da 10 s (colonne); a sinistra clip viste in training, a destra clip nuove (split_fm_lco, 9 job). Bianco = caso.

Numeri principali: `{"seen_mean_pct": 51.1, "new_mean_pct": 11.7, "by_window_pct": {"seen": [51.1, 52.8, 49.3], "new": [11.0, 12.2, 12.0]}, "best_new_clips": ["Sadness 1", "Disgust 1", "Neutral 2"]}`

## F11 — Heatmap clip x finestra, TSception + EA (10 s)

File: `fig11_heatmap_clip_window_tsception_ding_ea.pdf`, `fig11_heatmap_clip_window_tsception_ding_ea.png`

Percentuale di finestre classificate correttamente, media sui soggetti, per ognuna delle 28 clip (righe, per categoria) e delle 3 finestre da 10 s (colonne); a sinistra clip viste in training, a destra clip nuove (split_fm_lco, 9 job). Bianco = caso.

Numeri principali: `{"seen_mean_pct": 23.6, "new_mean_pct": 14.4, "by_window_pct": {"seen": [22.9, 24.7, 23.1], "new": [14.9, 15.2, 13.1]}, "best_new_clips": ["Disgust 3", "Neutral 2", "Tenderness 2"]}`

## F11 — Heatmap clip x finestra, REVE

File: `fig11_heatmap_clip_window_reve_base.pdf`, `fig11_heatmap_clip_window_reve_base.png`

Percentuale di finestre classificate correttamente, media sui soggetti, per ognuna delle 28 clip (righe, per categoria) e delle 3 finestre da 10 s (colonne); a sinistra clip viste in training, a destra clip nuove (split_fm_lco, 9 job). Bianco = caso.

Numeri principali: `{"seen_mean_pct": 45.2, "new_mean_pct": 13.2, "by_window_pct": {"seen": [45.9, 45.0, 44.6], "new": [11.6, 12.6, 15.3]}, "best_new_clips": ["Amusement 1", "Disgust 3", "Sadness 1"]}`

## CM — Matrici di confusione, DE + z-score

File: `fig_confusion_de_asym_lr_zsub.pdf`, `fig_confusion_de_asym_lr_zsub.png`

Matrici di confusione per finestra normalizzate per riga (% delle finestre della classe vera), clip viste a sinistra e clip nuove a destra (lnso10_lco, somma su fold e seed). La diagonale media è la recall media per classe.

Numeri principali: `{"seen": {"diag_mean_pct": 32.5, "per_class_recall_pct": {"Anger": 30.1, "Disgust": 49.8, "Fear": 31.7, "Sadness": 23.0, "Neutral": 43.8, "Amusement": 39.9, "Inspiration": 15.0, "Joy": 19.1, "Tenderness": 39.9}}, "unseen": {"diag_mean_pct": 16.3, "per_class_recall_pct": {"Anger": 7.1, "Disgust": 38.0, "Fear": 14.5, "Sadness": 8.6, "Neutral": 29.8, "Amusement": 19.5, "Inspiration": 5.9, "Joy": 6.1, "Tenderness": 17.1}}}`

## CM — Matrici di confusione, TSception + EA

File: `fig_confusion_tsception_ea.pdf`, `fig_confusion_tsception_ea.png`

Matrici di confusione per finestra normalizzate per riga (% delle finestre della classe vera), clip viste a sinistra e clip nuove a destra (lnso10_lco, somma su fold e seed). La diagonale media è la recall media per classe.

Numeri principali: `{"seen": {"diag_mean_pct": 28.1, "per_class_recall_pct": {"Anger": 25.0, "Disgust": 45.0, "Fear": 26.2, "Sadness": 18.5, "Neutral": 30.8, "Amusement": 38.0, "Inspiration": 13.2, "Joy": 18.5, "Tenderness": 38.2}}, "unseen": {"diag_mean_pct": 14.4, "per_class_recall_pct": {"Anger": 6.6, "Disgust": 32.5, "Fear": 10.1, "Sadness": 7.9, "Neutral": 22.7, "Amusement": 19.8, "Inspiration": 4.2, "Joy": 9.1, "Tenderness": 17.0}}}`

## CM — Matrici di confusione, CBraMod (LoRA)

File: `fig_confusion_cbramod_lora_r2.pdf`, `fig_confusion_cbramod_lora_r2.png`

Matrici di confusione per finestra normalizzate per riga (% delle finestre della classe vera), clip viste a sinistra e clip nuove a destra (lnso10_lco, somma su fold e seed). La diagonale media è la recall media per classe.

Numeri principali: `{"seen": {"diag_mean_pct": 69.7, "per_class_recall_pct": {"Anger": 71.2, "Disgust": 68.5, "Fear": 74.5, "Sadness": 51.5, "Neutral": 54.6, "Amusement": 78.8, "Inspiration": 72.2, "Joy": 78.0, "Tenderness": 78.0}}, "unseen": {"diag_mean_pct": 12.3, "per_class_recall_pct": {"Anger": 13.8, "Disgust": 14.9, "Fear": 9.6, "Sadness": 13.3, "Neutral": 21.5, "Amusement": 13.1, "Inspiration": 7.9, "Joy": 10.8, "Tenderness": 6.1}}}`

## CM — Matrici di confusione, EEGNet + EA

File: `fig_confusion_eegnet_lawhern_ea.pdf`, `fig_confusion_eegnet_lawhern_ea.png`

Matrici di confusione per finestra normalizzate per riga (% delle finestre della classe vera), clip viste a sinistra e clip nuove a destra (split_fm_lco, somma su fold e seed). La diagonale media è la recall media per classe.

Numeri principali: `{"seen": {"diag_mean_pct": 51.5, "per_class_recall_pct": {"Anger": 45.3, "Disgust": 47.4, "Fear": 59.5, "Sadness": 43.8, "Neutral": 41.1, "Amusement": 60.9, "Inspiration": 57.1, "Joy": 52.3, "Tenderness": 55.8}}, "unseen": {"diag_mean_pct": 11.4, "per_class_recall_pct": {"Anger": 7.1, "Disgust": 17.1, "Fear": 7.9, "Sadness": 15.0, "Neutral": 19.7, "Amusement": 13.8, "Inspiration": 7.6, "Joy": 8.1, "Tenderness": 6.6}}}`

## CM — Matrici di confusione, CBraMod

File: `fig_confusion_cbramod_ft_official.pdf`, `fig_confusion_cbramod_ft_official.png`

Matrici di confusione per finestra normalizzate per riga (% delle finestre della classe vera), clip viste a sinistra e clip nuove a destra (split_fm_lco, somma su fold e seed). La diagonale media è la recall media per classe.

Numeri principali: `{"seen": {"diag_mean_pct": 59.0, "per_class_recall_pct": {"Anger": 54.9, "Disgust": 58.6, "Fear": 72.6, "Sadness": 45.8, "Neutral": 53.2, "Amusement": 62.9, "Inspiration": 57.1, "Joy": 66.8, "Tenderness": 59.2}}, "unseen": {"diag_mean_pct": 11.0, "per_class_recall_pct": {"Anger": 8.9, "Disgust": 10.8, "Fear": 8.6, "Sadness": 18.5, "Neutral": 24.0, "Amusement": 10.0, "Inspiration": 4.5, "Joy": 7.5, "Tenderness": 6.2}}}`

## CM — Matrici di confusione, REVE

File: `fig_confusion_reve_base.pdf`, `fig_confusion_reve_base.png`

Matrici di confusione per finestra normalizzate per riga (% delle finestre della classe vera), clip viste a sinistra e clip nuove a destra (split_fm_lco, somma su fold e seed). La diagonale media è la recall media per classe.

Numeri principali: `{"seen": {"diag_mean_pct": 45.6, "per_class_recall_pct": {"Anger": 43.7, "Disgust": 50.7, "Fear": 50.8, "Sadness": 38.5, "Neutral": 33.2, "Amusement": 55.2, "Inspiration": 47.3, "Joy": 45.7, "Tenderness": 45.3}}, "unseen": {"diag_mean_pct": 13.1, "per_class_recall_pct": {"Anger": 9.2, "Disgust": 22.4, "Fear": 9.7, "Sadness": 15.3, "Neutral": 15.9, "Amusement": 21.7, "Inspiration": 7.2, "Joy": 8.2, "Tenderness": 8.1}}}`
