from training.interpreter_validation_v9 import summarize


def population():
    rows,predictions=[],[]
    for language in ('ro','uk'):
        for source in ('controlled_composition_v8','v7_training_frame_projection'):
            for gold in (0,1):
                for i in range(10):
                    identity=f'{language}-{source}-{gold}-{i}'
                    rows.append(dict(case_id=identity,language=language,source=source,attributes=dict(
                        subject='SELF',source='DIRECT',time='CURRENT',assertion='AFFIRMED',relation='NEW',concern=str(gold))))
                    predictions.append(dict(case_id=identity,language=language,gold=gold,predicted=gold))
    return rows,predictions


def test_binary_errors_cannot_win_only_through_lower_token_loss():
    rows,good=population();bad=[dict(v) for v in good];bad[0]['predicted']=1
    assert summarize(rows,good,.4)['rank'] < summarize(rows,bad,.1)['rank']


def test_worst_class_error_precedes_aggregate_tie():
    rows,a=population();b=[dict(v) for v in a]
    a[0]['predicted']=a[1]['predicted']=1
    b[0]['predicted']=b[20]['predicted']=1
    assert summarize(rows,b,.3)['rank'] < summarize(rows,a,.1)['rank']


def test_abstention_counts_as_error_and_extra_calibration_scores_are_unused():
    rows,predictions=population();predictions[0]['predicted']=None
    before=summarize(rows,predictions,.2)
    predictions.append(dict(case_id='unrelated',language='ro',gold=1,predicted=0))
    assert summarize(rows,predictions,.2)==before
    assert before['binary_errors']==1 and before['rank'][0]==1
