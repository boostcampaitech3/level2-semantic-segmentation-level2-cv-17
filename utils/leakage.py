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
    leak.remove('batch_03/0259.jpg')
    return leak

def filter_json(leak):
    """filter leak file information from data.json

    Args:
        leak (list): 195 leakge file names list

    - initalize ['images']['id'] and ['annotations']['image_id'],['annotations']['id']
    - remove 'UNKNOWN' category which category index == 0
    """

    with open('/opt/ml/input/data/data.json','r') as f:
        js = json.load(f)

    leak_files = []
    leak_files_id = []
    leak_files_dict = dict()
    a = 0
    for data in js['images']:
        if data['file_name'] in leak:
            leak_files_id.append(data['id'])
            leak_files_dict[str(data['id'])] = a
            data['id'] = a
            a += 1
            leak_files.append(data)
            
    print(leak_files_dict)

    leak_annos = []
    b = 0
    for data in js['annotations']:
        if data['image_id'] in leak_files_id and data['category_id']!=0:
            data['image_id'] = leak_files_dict[str(data['image_id'])]
            data['id'] = b
            b += 1
            leak_annos.append(data)

    js['images'] = leak_files
    js['annotations'] = leak_annos
   
    with open('/opt/ml/input/data/leak.json','w') as w:
        json.dump(js,w,indent=4)

if __name__ =='__main__':
    leak = leakage_list()
    filter_json(leak)
