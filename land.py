import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.stem.snowball import SnowballStemmer
from string import punctuation
from nltk import word_tokenize


class TextClassifier(object):
    """A text classifier model:
        - Vectorize the raw text into features.
        - Fit a naive bayes model to the resulting features.
    """

    def __init__(self):
        self._vectorizer = TfidfVectorizer(tokenizer = self.tokenizer, stop_words = 'english')
        self._classifier = MultinomialNB()

    def tokenizer(self,text):
        sbs = SnowballStemmer('english')
        punct = set(punctuation)
        return [sbs.stem(token) for token in word_tokenize(text) if token not in punct]

    def fit(self, X, y):
        """Fit a text classifier model.

        Parameters
        ----------
        X: A numpy array or list of text fragments, to be used as predictors.
        y: A numpy array or python list of labels, to be used as responses.

        Returns
        -------
        self: The fit model object.
        """
        # Code to fit the model.
        X = self._vectorizer.fit_transform(X)
        self.mnb = MultinomialNB()
        self.mnb.fit(X,y)
        return self.mnb

    def predict_proba(self, X):
        """Make probability predictions on new data."""
        X = self._vectorizer.transform(X)
        return self.mnb.predict_proba(X)

    def predict(self, X):
        """Make predictions on new data."""
        X = self._vectorizer.transform(X)
        return self.mnb.predict(X)

    def score(self, X, y):
        """Return a classification accuracy score on new data."""
        X = self._vectorizer.transform(X)
        return self.mnb.score(X,y)


def percent_capitals(text):
    text = [a for a in text if a in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"]

    original_length = len(text)
    if original_length == 0: return 0
    capitals = len([x for x in text if x!=x.lower()])
    return capitals/original_length


if __name__ == '__main__':


    df = pd.read_json('data/data.json')
    # df = df.iloc[:100,:]

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

    '''CALCULATE PERCENT CAPITALS'''
    df['description_capitals'] = df.description.apply(lambda x:percent_capitals(x))
    df['name_capitals'] = df.name.apply(lambda x:percent_capitals(x))


    ''' CALCULATE TIME INTERVALS'''
    s_d = 3600*24
    df['start_to_pay'] = (df['approx_payout_date'] - df['event_start'])/s_d
    df['create_to_pay'] = (df['approx_payout_date'] - df['event_created'])/s_d
    df['duration'] = (df['event_end'] - df['event_start'])/s_d
    df['user_to_create_event'] = (df['event_created']-df['user_created'])/s_d
    df['user_to_publish'] = (df['event_published'] - df['user_created'])/s_d
    df['publish_to_start'] = (df['event_start'] - df['event_published'])/s_d

    df = df.drop(['approx_payout_date','event_start','approx_payout_date','event_created','event_end', 'event_published', 'user_created'], axis = 1)

    '''TFIDF ON TEXT COLUMNS'''
    tf_idf_columns = ['description']
    df['description_prob'] = 0

    dummy_columns = [col for col in df.columns if df[col].dtype=='O']

    dummy_columns.remove('description')

    for col in dummy_columns:
        categories = df[col].nunique()
        # print(col, categories)
        if categories > 20:
            print('Dropping ', col)
        else:
            dummies = pd.get_dummies(df[col], prefix = col, dummy_na = (col!='delivery_method'), drop_first = True)
            df = pd.concat([df, dummies], axis = 1)
        del df[col]
    #
    # t_c = TextClassifier()
    # t_c.fit(df.description.values, df.target.values)
    # predictions = t_c.predict_proba(df.description.values)
