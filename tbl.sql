USE test;
DROP TABLE IF EXISTS tbl;
CREATE TABLE tbl(
	id	int not null,
	location	int not null,
	event_type	int not null,
	resource_type	int not null,
	severity_type	int not null,
	feature	int not null,
	vol	int not null,
	fault	int not null
)ENGINE=MYISAM CHARSET=UTF8;

load data local infile './train.csv' into table tbl fields terminated by ',' ignore 1 lines;
