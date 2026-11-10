import pytest
np=pytest.importorskip('numpy')

from training.train_large_response_v10 import population_metrics, training_weights
from roguard.trajectory_model import RESPONSE_LABELS


def population():
    rows=[]
    for origin in ('historical','generated'):
        for language in ('ro','uk'):
            for gold in RESPONSE_LABELS:
                for i in range(1 if origin=='historical' else 7):
                    rows.append({'origin':origin,'language':language,'gold':gold,'draft':str((origin,language,gold,i))})
    return rows


def test_large_fit_weights_balance_origins_and_categories_despite_expansion():
    rows=population();weights=training_weights(rows,list(range(len(rows))))
    sums={}
    for row,weight in zip(rows,weights):
        key=(row['origin'],row['language'],row['gold']);sums[key]=sums.get(key,0.)+weight
    assert max(sums.values())==pytest.approx(min(sums.values()))
    assert sum(weights)==pytest.approx(len(rows))


def test_selection_retains_historical_errors_as_separate_population():
    rows=population();z=np.full((len(rows),5),-10.)
    for i,row in enumerate(rows):z[i,RESPONSE_LABELS.index(row['gold'])]=10.
    assert population_metrics(rows,z)['errors']==0
    z[0]=[20.,-10.,-10.,-10.,-10.]
    # Force the first row wrong regardless of label ordering.
    z[0,RESPONSE_LABELS.index(rows[0]['gold'])]=-20.
    metrics=population_metrics(rows,z)
    assert metrics['populations']['historical']['errors']==1
    assert metrics['populations']['generated']['errors']==0
    assert metrics['selection_rank'][0]==1.
