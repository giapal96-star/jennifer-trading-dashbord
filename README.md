# Jennifer Trading System

Sistema di screening e analisi strutturale asset — Layer 1 + Layer 2.

## Struttura

```
├── Sarada_Trading_System.py              # Layer 1 — Screening macro + tecnico
├── Sarada_trading_system_operatività.py  # Layer 2 — Setup operativi
├── Avvia_Sarada_Trading_System_v2.bat    # Avvio locale Windows
├── requirements.txt
├── docs/                                 # Dashboard HTML (GitHub Pages)
│   ├── index.html
│   ├── dashboard.html                    # Generato dal Layer 1
│   ├── jennifer_setups.html              # Generato dal Layer 2
│   └── manifest.json                     # PWA config
└── .github/workflows/jennifer.yml        # GitHub Actions
```

## Utilizzo locale (Windows)

1. Installa dipendenze: `pip install -r requirements.txt`
2. Doppio clic su `Avvia_Sarada_Trading_System_v2.bat`

## GitHub Actions (cloud automatico)

Il workflow `.github/workflows/jennifer.yml` esegue il sistema automaticamente
ogni 15 minuti durante gli orari di mercato (lun-ven, 9-22 UTC).

Per aggiungere la chiave FRED API (opzionale):
- Repository → Settings → Secrets → Actions → New secret
- Nome: `FRED_API_KEY`, Valore: la tua chiave

## Dashboard online

Attiva GitHub Pages dal repository:
- Settings → Pages → Source: Deploy from branch → Branch: main → /docs

La dashboard sarà disponibile su:
`https://giapal96-star.github.io/jennifer-trading/`

## Installazione come app mobile (PWA)

1. Apri il link sopra dal telefono
2. Browser menu → "Aggiungi a schermata Home"
3. L'app appare con icona propria, si apre a schermo intero

---
*Solo analisi statistica — non costituisce consulenza finanziaria.*
