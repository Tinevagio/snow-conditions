# ❄️ Névé — Conditions de neige en temps réel

Application web de visualisation des conditions nivologiques pour le ski de randonnée et l'alpinisme, combinant les données météo Open-Meteo et les Bulletins d'Estimation du Risque d'Avalanche (BERA) de Météo France.

**Frontend** : https://snow-conditions.netlify.app  
**API** : https://snow-conditions.onrender.com

---

## Architecture
```
snow-conditions/          ← ce repo (backend FastAPI)
├── api/
│   └── main.py           ← endpoints FastAPI
├── bera_corrector.py     ← fusion Open-Meteo + BERA
├── massif_locator.py     ← localisation GPS → massif
├── core/
│   ├── snow_model.py     ← modèle de qualité de neige
│   ├── terrain.py        ← grille terrain + exposition
│   └── solar_radiation.py
└── data/
    └── fetchers/
        └── openmeteo.py

Ski-touring-live/         ← repo séparé (data BERA)
├── scripts/
│   ├── fetch_bera.py     ← scrape Météo France quotidiennement
│   └── build_massif_polygons.py  ← one-shot, génère les polygones
└── data/
    ├── bera_enneigement.json     ← mis à jour chaque matin
    └── massif_polygons.json      ← polygones OSM des massifs
```

---

## Endpoints API

| Endpoint | Description |
|---|---|
| `GET /conditions?bbox=&date=&resolution_m=` | Grille de conditions nivologiques sur une zone |
| `GET /conditions/point?lat=&lon=&date=` | Conditions pour un point précis |
| `GET /best-window?bbox=&date=` | Meilleure fenêtre horaire de la journée |
| `GET /debug/bera?lat=&lon=` | Données BERA brutes pour un point |

### Exemple
```bash
curl "https://snow-conditions.onrender.com/debug/bera?lat=45.92&lon=6.87"
```

---

## Conditions de neige

| Code | Label | Description |
|---|---|---|
| `POWDER_COLD` | Poudreuse froide | Neige fraîche froide, conditions idéales |
| `SPRING_SNOW` | Neige de printemps | Regel matinal, transformation en journée |
| `OLD_PACKED` | Neige tassée | Vieille neige compacte |
| `WET_HEAVY` | Neige lourde | Neige humide et collante |
| `NO_SNOW` | Pas de neige | En dessous de la limite d'enneigement |

---

## Intégration BERA

Le `bera_corrector.py` recalibre les sorties Open-Meteo grâce aux données BERA :

- **Localisation** : `massif_locator.py` associe chaque point GPS à son massif via polygones OSM (36 massifs français)
- **Correction** : fusion pondérée `(1-α) × Open-Meteo + α × BERA` avec `α = 0.8`
- **Forçages** : mise à zéro sous la limite d'enneigement BERA, correction de la qualité selon l'exposition et l'altitude
- **Données** : `bera_enneigement.json` mis à jour chaque matin via GitHub Actions sur [Ski-touring-live](https://github.com/Tinevagio/Ski-touring-live)

---

## Stack technique

- **Backend** : FastAPI + Uvicorn, déployé sur [Render](https://render.com)
- **Frontend** : HTML/JS vanilla, déployé sur [Netlify](https://netlify.com)
- **Météo** : [Open-Meteo](https://open-meteo.com) (gratuit, sans clé API)
- **Altimétrie** : IGN Altimétrie REST
- **BERA** : [Météo France](https://donneespubliques.meteofrance.fr) — scraping XML quotidien
- **Polygones massifs** : OpenStreetMap via Overpass API + centroïdes IGN

---

## Développement local
```bash
git clone https://github.com/Tinevagio/snow-conditions
cd snow-conditions
pip install -r requirements.txt
uvicorn api.main:app --reload
```

L'API sera disponible sur `http://localhost:8000`.  
Les données BERA sont téléchargées automatiquement depuis GitHub au démarrage.