import sys

def extract_train(csv_train):
    with open(csv_train) as f:
        f.readline()
        for line in f:
            row = line.strip().split(',')
            id = row[0]
            location = row[1].split(' ')[1]
            fault = row[2]
            yield id, location, fault


def extract_event(csv_event_type):
    with open(csv_event_type) as f:
        f.readline()
        for line in f:
            row = line.strip().split(',')
            id = row[0]
            event_type = row[1].split(' ')[1]
            yield id, event_type

def extract_res(csv_res_type):
    with open(csv_res_type) as f:
        f.readline()
        for line in f:
            row = line.strip().split(',')
            id = row[0]
            res_type = row[1].split(' ')[1]
            yield id, res_type

def extract_sevr(csv_sevr_type):
    with open(csv_sevr_type) as f:
        f.readline()
        for line in f:
            row = line.strip().split(',')
            id = row[0]
            sevr_type = row[1].split(' ')[1]
            yield id, sevr_type

def extract_log(csv_log):
    with open(csv_log) as f:
        f.readline()
        for line in f:
            row = line.strip().split(',')
            id = row[0]
            log_code = row[1].split(' ')[1]
            vol = row[2]
            yield id, log_code, vol

def make_table(csv_train, csv_event_type, csv_res_type, csv_sevr_type, csv_log):
    tbl = {}

    for record in extract_train(csv_train):
        tbl.setdefault(record[0], [record[1],record[2]])

    for record in extract_event(csv_event_type):
        if record[0] not in tbl:
            # positive sample recorded
            tbl.setdefault(record[0], ['0', '0', record[1]])
        else:
            # negative sample 
            tbl[record[0]].append(record[1])

    for record in extract_res(csv_res_type):
        if record[0] not in tbl:
            # positive sample recorded
            tbl.setdefault(record[0], ['0', '0', '0', record[1]])
        else:
            # negative sample 
            tbl[record[0]].append(record[1])

    for record in extract_sevr(csv_sevr_type):
        if record[0] not in tbl:
            # positive sample recorded
            tbl.setdefault(record[0], ['0', '0', '0', '0', record[1]])
        else:
            # negative sample 
            tbl[record[0]].append(record[1])

    for record in extract_log(csv_log):
        if record[0] not in tbl:
            # positive sample recorded
            tbl.setdefault(record[0], ['0', '0', '0', '0', '0', record[1], record[2]])
        else:
            # negative sample 
            tbl[record[0]].append(record[1])
            tbl[record[0]].append(record[2])
        print tbl

    return tbl


def test():
    csv_train = '../data/train.csv'
    csv_event_type = '../data/event_type.csv'
    csv_sevr_type= '../data/severity_type.csv'
    csv_res_type = '../data/resource_type.csv'
    csv_log = '../data/log_feature.csv'

    for rec in extract_log(csv_log):
        print rec
        sys.exit()

def main():
    csv_train = '../data/train.csv'
    csv_event_type = '../data/event_type.csv'
    csv_sevr_type= '../data/severity_type.csv'
    csv_res_type = '../data/resource_type.csv'
    csv_log = '../data/log_feature.csv'

    

    tbl = make_table(csv_train, csv_event_type, csv_res_type, csv_sevr_type, csv_log)
    #for key in tbl:
    #    print key, ' '.join(tbl[key])


if __name__ == '__main__':
    #main()
    test()

    
    
            
