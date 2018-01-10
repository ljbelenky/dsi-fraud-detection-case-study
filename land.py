import json
import pandas as pd

df = pd.read_json('data/data.json')
df = df.iloc[:200,:]

df['target'] = df.acct_type.apply(lambda x:1*(x[:5]=='fraud'))
df['listed'] = df.listed.apply(lambda x:1*(x=='y'))

'''ONLY INTERESTED IN THE LENGTH'''
columns = ['previous_payouts', 'ticket_types']
for col in columns:
    df[col] = df[col].apply(lambda x:len(x))


''' DROP COLUMNS '''
drop_columns = ['acct_type', 'object_id']
df = df.drop(drop_columns, axis = 1)

''' CHANGE CATEGORICAL DATA TO STRING '''
categoricals = ['channels', 'org_facebook', 'org_twitter', 'fb_published', 'delivery_method', 'user_type']
for cat in categoricals:
    df[cat] = df[cat].apply(lambda x:str(x))


''' CALCULATE TIME INTERVALS'''
df['start_to_pay'] = df['approx_payout_date'] - df['event_start']
df['create_to_pay'] = df['approx_payout_date'] - df['event_created']
df['duration'] = df['event_end'] - df['event_start']
df['user_to_create_event'] = df['event_created']-df['user_created']
df['user_to_publish'] = df['event_published'] - df['user_created']
df['publish_to_start'] = df['event_start'] - df['event_published']

# df = df.drop(['approx_payout_date','event_start','approx_payout_date','event_created','event_end', 'event_published', 'user_created'], axis = 1)

'''TFIDF ON TEXT COLUMNS'''
tf_idf_columns = ['description']
df['description_prob'] = 0

dummy_columns = [col for col in df.columns if df[col].dtype=='O'].remove('description')


for col in dummy_columns:
    categories = df[col].nunique()
    # print(col, categories)
    if categories > 20:
        print('Dropping ', col)
    else:
        dummies = pd.get_dummies(df[col], prefix = col, dummy_na = True, drop_first = True)
        df = pd.concat([df, dummies], axis = 1)
    del df[col]
