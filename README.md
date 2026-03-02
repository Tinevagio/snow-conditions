# Snow Conditions Model

Moteur de prédiction des conditions de neige en temps réel pour le ski de randonnée.

## Concept
Pour une zone géographique et une plage horaire données, le modèle prédit 
l'état de surface de la neige (poudre, transformée, croûte, etc.) 
versant par versant, heure par heure.

## Stack
- Python 3.11
- FastAPI
- Open-Meteo API (météo temps réel)
- MNT IGN (données terrain)
- pvlib (calculs solaires)

## Installation
```bash
git clone https://github.com/TON_USERNAME/snow-conditions.git
cd snow-conditions
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## Lancer l'API en local
```bash
uvicorn api.main:app --reload
```

## Structure
- `core/` — moteur physique pur, sans dépendances externes
- `data/fetchers/` — connecteurs APIs
- `api/` — exposition FastAPI
- `notebooks/` — exploration et validation du modèle
