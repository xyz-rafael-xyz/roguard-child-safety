"""Separate real-support interpretation from revision/scope review obligations.

The Boolean OR preserves the existing evidence policy for every complete world.
These are two distinct language tasks, not a change to benchmark labels.
"""
HYPOTHESES = (
 'The focal message conveys a real person\'s emotional distress, safety problem or need for personal support, including uncertain, past or hypothetical concerns.',
 'The focal message withdraws an earlier concerning statement, or gives conflicting claims about whether that concerning statement is real or fictional.',
)


def targets(row):
    a=row['attributes']
    concern=a['concern']=='1'
    support=concern and a['subject']!='FICTIONAL' and a['source']!='FICTION' and a['assertion'] not in ('NEGATED','RETRACTED')
    revision=concern and (a['assertion']=='RETRACTED' or a['relation']=='RETRACTS' or ((a['subject']=='FICTIONAL') != (a['source']=='FICTION')))
    return int(support),int(revision)
