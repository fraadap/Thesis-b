"""Sessione 03: label da self-report (voti dati da ogni soggetto dopo ogni clip).

- `table`: passo 0, tabella dei voti (lettura di After_remarks, allineamento, controlli, colonne derivate);
- `targets`: label per (soggetto, clip) dei task `sr_*`, target entro clip per fold, permutazioni entro clip;
- `metrics`: metriche delle componenti tra clip ed entro clip, bootstrap incrociato, nulli, descrittive;
- `analysis`: unione dei fold, tabelle 2 x 2, rianalisi delle predizioni salvate, terza famiglia.
"""
