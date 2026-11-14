# Feature catalog

All aggregated features are computed **exclusively from the training split**, and persisted as CSV for reuse by training or prediction clients. Raw item metadata is joined independently of future interaction labels.

| Entity | Feature | Type | Definition | Source |
|---|---|---|---|---|
| User | user_activity_count | integer | Count of historical training interactions | train.csv |
| User | user_average_rating | float | Mean rating over training interactions | train.csv |
| User | days_since_last_interaction | float | Days between maximum training timestamp and user's most recent training interaction | train.csv |
| User | preferred_genre | category | Most frequently interacted movie genre; alphabetical tie break | train.csv + items.csv |
| Item | item_popularity | integer | Count of training interactions for item | train.csv |
| Item | item_average_rating | float | Mean training rating; zero for unseen items | train.csv |
| Item | genre | category | Movie genre | items.csv |
| Item | release_year | integer | Movie release year | items.csv |

**Feature tables:** `data/features/user_features.csv`, `data/features/item_features.csv`.

**Model usage:** The trained baseline uses item count and mean rating; collaborative filtering uses the train-only interaction matrix. Other materialized features are available to downstream prediction or future models and are not falsely claimed as current model inputs.
