import pandas as pd
import pickle
from land import percent_capitals

s_per_d = 3600*24


def transform(X):
    Y = {
 'body_length': [X.body_length],
 'channels_10': [1*(X.channels==10)],
 'channels_11': [1*(X.channels==11)],
 'channels_12': [1*(X.channels==12)],
 'channels_13': [1*(X.channels==13)],
 'channels_4': [1*(X.channels==4)],
 'channels_5': [1*(X.channels==5)],
 'channels_6': [1*(X.channels==6)],
 'channels_7': [1*(X.channels==7)],
 'channels_8': [1*(X.channels==8)],
 'channels_9': [1*(X.channels==9)],
 'channels_nan': [1*(X.channels not in [0,10,11,12,13,4,5,6,7,8,9])],
 'create_to_pay': [(X.approx_payout_date-X.event_created)/s_per_d],
 'currency_CAD': [1*(X.currency=='CAD')],
 'currency_EUR': [1*(X.currency=='EUR')],
 'currency_GBP': [1*(X.currency=='GBP')],
 'currency_MXN': [1*(X.currency=='MXN')],
 'currency_NZD': [1*(X.currency=='NZD')],
 'currency_USD': [1*(X.currency=='USD')],
 'currency_nan': [1*(X.currency not in ['CAD','EUR','GBP','MXN','NZD','USD', 'AUD'])],
 'delivery_method_1.0': [1*(X.delivery_method==1)],
 'delivery_method_3.0': [1*(X.delivery_method==3)],
 'delivery_method_nan': [1*(X.delivery_method not in [1,3,0])],
 'description': [X.description],
 'description_capitals': [percent_capitals(X.description)],
 'description_prob': [0],
 'duration': [(X.event_end-X.event_start)/s_per_d],
 'fb_published_1': [1*(X.fb_published==1)],
 'fb_published_nan':[1*(X.fb_published not in [0,1])],
 'gts': [X.gts],
 'has_analytics': [X.has_analytics],
 'has_header': [X.has_header],
 'has_logo': [X.has_logo],
 'listed': [X.listed],
 'name_capitals': [percent_capitals(X['name'])],
 'name_length': [X.name_length],
 'num_order': [X.num_order],
 'num_payouts': [X.num_payouts],
 'payout_type_ACH': [1*(X.payout_type=='ACH')],
 'payout_type_CHECK': [1*(X.payout_type=='CHECK')],
 'payout_type_nan': [1*(X.payout_type not in ['ACH','CHECK', ''])],
 'previous_payouts': [len(X.previous_payouts)],
 'publish_to_start': [(X.event_start-X.event_published)/s_per_d],
 'sale_duration': [X.sale_duration],
 'sale_duration2': [X.sale_duration2],
 'show_map': [X.show_map],
 'start_to_pay': [(X.approx_payout_date-X.event_start)/s_per_d],
 'ticket_types': [len(X.ticket_types)],
 'user_age': [X.user_age],
 'user_to_create_event': [(X.event_created-X.user_created)/s_per_d],
 'user_to_publish': [(X.event_published-X.user_created)/s_per_d],
 'user_type_103': [1*(X.user_type==103)],
 'user_type_2': [1*(X.user_type == 2)],
 'user_type_3': [1*(X.user_type == 3)],
 'user_type_4': [1*(X.user_type == 4)],
 'user_type_5': [1*(X.user_type == 5)],
 'user_type_nan': [1*(X.user_type not in [1,2,3,4,5,103])],
 'venue_latitude': [X.venue_latitude],
 'venue_longitude':[X.venue_longitude]}

    return pd.DataFrame(data=Y)
