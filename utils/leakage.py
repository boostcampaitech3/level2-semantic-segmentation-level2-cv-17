import json

def leakage_list():

    with open('/opt/ml/input/data/test_before.json','r') as f:
        js_be = json.load(f)

    with open('/opt/ml/input/data/test.json','r') as f:
        js_af = json.load(f)

    be_files = set()
    af_files = set()

    for be in js_be['images']:
        be_files.add(be['file_name'])
    for af in js_af['images']:
        af_files.add(af['file_name'])

    leak = be_files - af_files
    return leak

def filter_json(leak):
    """filter leak file information from data.json

    Args:
        leak (list): 195 leakge file names list
    """

    with open('/opt/ml/input/data/data.json','r') as f:
        js = json.load(f)

    leak_files = []
    leak_files_id = []
    for data in js['images']:
        if data['file_name'] in leak:
            leak_files.append(data)
            leak_files_id.append(data['id'])

    leak_annos = []
    for data in js['annotations']:
        if data['image_id'] in leak_files_id and data['category_id']!=0:
            leak_annos.append(data)

    js['images'] = leak_files
    js['annotations'] = leak_annos
   
    with open('/opt/ml/input/data/leak.json','w') as w:
        json.dump(js,w,indent=4)

if __name__ =='__main__':
    leak = leakage_list()
    filter_json(leak)
