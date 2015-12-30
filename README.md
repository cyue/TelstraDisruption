# TelstraDisruption
https://www.kaggle.com/c/telstra-recruiting-network

# Input information
1. train.csv				id,location,fault_severity	# without replacement
2. event_type.csv			id,event_type			# with replacement
3. resource_type.csv			id,resource_type		# with replacement
4. severity_type.csv			id,severity_type		# without replacement
5. log_feature.csv			id,log_feature,volume		# with replacement


# explanation
fault_severity - 
	0: no fault
	1: a few fault
	2: many fault
	

# Feature extracted
id	location	event_type	resource_type	severity_type	log_feature	log_volumn	fault
61838	929	49	10	5	331	254	3

# fault distribution
0:59%	1:26%	2:14%
