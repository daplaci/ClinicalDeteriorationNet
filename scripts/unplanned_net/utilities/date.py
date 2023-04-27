import datetime
from dateutil import tz

END_OF_DATA = '2018-04-10'

def get_date_adm(line_dict):
    if len(line_dict["v_indtime"]) == 1:
        h = '0' + line_dict["v_indtime"]
    if len(line_dict["v_indminut"]) == 1:
        m = '0' + line_dict["v_indminut"]
    
    if line_dict["v_indtime"] =='NULL':
        h = "00"
    else:
        h = line_dict["v_indtime"]
    if line_dict["v_indminut"] == "NULL":
        m = "00"
    else:
        m = line_dict["v_indminut"]
    dt = "{} {}:{}".format(line_dict["d_inddto"],h,m)
    return dt

def get_date_disch(line_dict):
    if len(line_dict["v_udtime"]) == 1:
        h = '0' + line_dict["v_udtime"]    
    if line_dict["v_udtime"] =='NULL':
        h = "00"
    else:
        h = line_dict["v_udtime"]

    m = "00"
    dt = "{} {}:{}".format(line_dict["d_uddto"],h,m)
    return dt

def get_bth_date_adm (line_dict):
    if not line_dict["ADM_YMD"]:
        return None
    
    if line_dict["ADM_HM"] == "NULL" or not line_dict["ADM_HM"]:
        hm = "0000"
    else:
        hm = line_dict["ADM_HM"]
    
    dt = "{}{}".format(line_dict["ADM_YMD"],hm)
    return dt

def get_bth_date_disch (line_dict):
    if not line_dict["DIS_YMD"]:
        return None
    
    if line_dict["DIS_HM"] == "NULL" or not line_dict["DIS_HM"]:
        hm = "0000"
    else:
        hm = line_dict["DIS_HM"]
    
    dt = "{}{}".format(line_dict["DIS_YMD"],hm)
    return dt 

def parse_date(date_str, is_utc=False):
    if date_str == 'nan' or date_str == 'NA':
        date_str = END_OF_DATA
    if len(date_str) == 10:
        format_str = '%Y-%m-%d'
    elif len(date_str) == 4:
        format_str = '%Y'
    elif len(date_str) == 12:
        format_str = '%Y%m%d%H%M'
    elif len(date_str) == 20:
        format_str = "%Y-%m-%dT%H:%M:%SZ"
    else:
        format_str = '%Y-%m-%d %H:%M'
        
    dt = datetime.datetime.strptime(date_str, format_str)
    
    if is_utc:
        timestamp = dt.replace(tzinfo=datetime.timezone.utc).timestamp()
        return timestamp
    
    return dt.timestamp()


def timestamp_to_datetime(timestamp):
    return datetime.datetime.fromtimestamp(timestamp).isoformat()

START_BTH = parse_date('2011-10-28')
END_BTH = parse_date('2016-06-30')