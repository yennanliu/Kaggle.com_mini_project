import pandas as pd, numpy as np
%matplotlib inline
%pylab inline
import seaborn 

'''
FEATURE:
1. total  # of bids that bidder have made 
2. average # of bids that bidder have made per auction
3. average gap of time between 2 series bids 
4. How many IP, phone that every bidder use totally

'''

df_bids = pd.read_csv('~/Desktop/Facebook _4_robot/bids.csv')
df_train = pd.read_csv('~/Desktop/Facebook _4_robot/train.csv')
df_test = pd.read_csv('~/Desktop/Facebook _4_robot/test.csv')
df_submission = pd.read_csv('~/Desktop/Facebook _4_robot/sampleSubmission.csv')

#train_merge = df_train.merge(df_bids, how = 'left', left_on = ['bidder_id'], right_on = ['bidder_id'])

train_merge = pd.merge(df_train, df_bids , how = 'left', on =['bidder_id'])

df_submission = pd.read_csv('~/Desktop/Facebook _4_robot/sampleSubmission.csv')
test_merge = pd.merge(df_test, df_bids , how = 'left', on =['bidder_id'])
test_merge = pd.merge(test_merge, df_submission , how = 'left', on =['bidder_id'])

#table = pd.pivot_table(train_merge, index = ['bidder_id', 'time'], aggfunc = np.size)
(test_merge.groupby('time').count())['bid_id'].plot()
(train_merge.groupby('time').count())['bid_id'].plot()

data1 = train_merge[train_merge['time'] < 966*(10**13)]
data2 = train_merge[966*(10**13) <  train_merge[train_merge['time'] < 972*(10**13)]]
data3 = train_merge[974*(10**13) <train_merge['time'] ]

(data3.groupby('time').count())['bid_id'].plot()
(data3.groupby('bidder_id').count())['bid_id'].plot()
data3[data3['outcome'] == 1].groupby(['outcome', 'bidder_id']).count()['bid_id']
data3[data3['outcome'] == 0].groupby(['outcome', 'bidder_id']).count()['bid_id']
data1.groupby(['auction', 'bid_id', 'bidder_id']).count()
###  
'''
 set up a function that can calculate the "time interval between two successive bid 
 for same bidder 
'''
###

def feature_engineering(data):
	df_feature = data[['bidder_id','outcome']]
	df_feature = df_feature.drop_duplicates()
	df_feature = pd.merge(df_feature , total_number_bid_bidder_made(data), how = 'left', on=['bidder_id'])
	df_feature = pd.merge(df_feature , ave_gap_of_time_in_2_bids(data), how = 'left', on=['bidder_id'])
	return df_feature

def total_number_bid_bidder_made(data):
	total_number_bid_bidder = data.groupby('bidder_id').count()['payment_account']
	total_number_bid_bidder = pd.DataFrame(total_number_bid_bidder).reset_index()
	total_number_bid_bidder.columns =['bidder_id', 'total_number_bid_bidder_made']
	return total_number_bid_bidder
	#return pd.merge(data, total_number_bid_bidder_made(data), how = 'inner', on= ['bidder_id'])
	#return data.groupby('bidder_id').count()['payment_account']

def avg_number_bid_bidder_made(data):
	#((data.groupby(['bidder_id', 'auction']).count()['outcome'].reset_index()).groupby('bidder_id').count())\
	avg_number_bid_bidder = (data.groupby(['bidder_id']).count()['outcome'])/(((data.groupby(['bidder_id', 'auction']).count()['outcome'].reset_index()).groupby('bidder_id').count()['outcome']))
	avg_number_bid_bidder = pd.DataFrame(avg_number_bid_bidder).reset_index()
	avg_number_bid_bidder.columns =['bidder_id', 'avg_number_bid_bidder_made']
	return avg_number_bid_bidder

def ave_gap_of_time_in_2_bids(data):
	ave_gap_of_time = [[] for i in range(2)]
	for k in list(set(data.bidder_id)):
		data_k = data[data['bidder_id'] == k]
		if max(data_k['time']) and min(data_k['time']) : 
			ave_gap_of_time[0].append(k)
			ave_gap_of_time[1].append(float(max(data_k['time'])) - float(min(data_k['time'])))
			df = pd.DataFrame(ave_gap_of_time).transpose()
			df.columns = ['bidder_id','gap_time_max_min']
	return df 

def time_range_all(datak):

	time_range=[[] for i in range(2)]
	for k in list(set(datak.auction)):
		data = datak[datak['auction'] == k]
		time_range[0].append(k)
		time_range[1].append(float(max(data.time)) - float(min(data.time)))
		df = pd.DataFrame(time_range).transpose()
		#data.time_range_all = time_range
	return df

def avg_number_bid_per_auction(data):
	data0 = (data[data['outcome'] == 0])
	(data0.groupby(['auction']).count()['outcome'])
	((data0.groupby(['auction','bidder_id']).count()['outcome'].reset_index()).groupby('auction').count())['bidder_id']
	(data0.groupby(['auction']).count()['outcome'])/(((data0.groupby(['auction','bidder_id']).count()['outcome'].reset_index()).groupby('auction').count())['bidder_id'])
	xx = (data0.groupby(['auction']).count()['outcome'])/(((data0.groupby(['auction','bidder_id']).count()['outcome'].reset_index()).groupby('auction').count())['bidder_id'])
	xxx = pd.DataFrame(xx) 
	data2 = (data[data['outcome'] == 1])
	yy = (data2.groupby(['auction']).count()['outcome'])/(((data2.groupby(['auction','bidder_id']).count()['outcome'].reset_index()).groupby('auction').count())['bidder_id'])
	yyy = pd.DataFrame(yy)
	xxx= xxx[xxx <100]
	#yyy= yyy[yyy <100]
	print (xxx)
	print ("****")
	print (yyy)
	pyplot.hist(xxx.dropna(), alpha=0.3, label = "0" , color= 'k', bins=40, range=[0, 100])
	pyplot.hist(yyy.dropna(),  alpha=0.3,label = "1", bins=40, range=[0, 100])
	pyplot.title('avg_number_bid_auction')
	pyplot.xlabel('bids')
	pyplot.ylabel('accumulation')
	pyplot.legend( ('0 (human)', '1 (bot)') )
	pyplot.show()
	#xxx[xxx <100].hist(bins = 30 , normed=True)
	#yyy[yyy <100].hist(bins = 30 , normed=True)

def time_interval_auction(auction,bidder_id):
	#list(set(data1.bidder_id))
 	sample = (train_merge[train_merge['bidder_id'] == bidder_id ])
 	sample = (sample[sample['auction'] == auction ])['time']
 	
 	gap = sample - sample.shift(periods=1, freq=None, axis=0)
 	gap = gap.replace(NaN,0) 
 	gap = gap[gap <= gap.quantile(.9)]
 	print (sum(gap)/len(gap))
 	gap.plot(kind = 'kde')
 	plt.show()
 	gap.plot()
 	plt.show()
 	return gap 

def time_interval(bidder_id):
	#list(set(data1.bidder_id))
 	sample = (train_merge[train_merge['bidder_id'] == bidder_id ])['time']
 	gap = sample - sample.shift(periods=1, freq=None, axis=0)
 	gap = gap.replace(NaN,0) 
 	gap = gap[gap <= gap.quantile(.9)]
 	print (sum(gap)/len(gap))
 	gap.plot(kind = 'kde')
 	plt.show()
 	gap.plot()
 	plt.show()
 	return gap 

def avg_time_interval(datak,outcome):
	avg_value = []
	datak = datak[datak['outcome'] == outcome]
	for j in list(set(datak.bidder_id)):
		sample = (datak[datak['bidder_id'] == j])['time']
		gap = sample - sample.shift(periods=1, freq=None, axis=0)
		gap = gap.replace(NaN,0) 
		avg_value.append(sum(gap)/len(gap))
		#print (average)
		#print ("#######")

	pd.DataFrame(avg_value).plot(kind='kde')
	return pd.DataFrame(avg_value)
	#return gap

###Feature Engineering###
def time_range_all(datak):
	time_range=[]
	for k in list(set(datak.auction)):
		data = datak[datak['auction'] == k]
		time_range.append(float(max(data.time)) - float(min(data.time)))

		#data.time_range_all = time_range
	return (k, time_range)

def time_gap_final_bid_end_auction(datak,bid_id):
	time_gap = []
	for k in list(set(datak.auction)):
		data_a = datak[datak['auction'] == k]
		data_b = datak[datak['bid_id'] == bid_id]
		time_gap.append(float(max(data_a.time)) - float(max(data_b.time)))
		return k, time_gap
