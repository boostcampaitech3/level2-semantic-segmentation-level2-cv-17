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

def merge_json(leak,json_path):
    """merge leak file information to train or train_all.json

    Args:
        leak (list): 195 leakge file names list
        json_path (str): json path which you want to merged from leakage -> ex) /opt/ml/input/data/train.json
    """

    with open('/opt/ml/input/data/data.json','r') as f:
        js = json.load(f)
    with open(json_path,'r') as f:
        merge_js = json.load(f)

    leak_files = []
    leak_files_id = []
    for data in js['images']:
        if data['file_name'] in leak:
            leak_files_id.append(data['id'])

            data['id'] = data['id']+merge_js['images'][-1]['id']+1
            leak_files.append(data)
            

    leak_annos = []
    for data in js['annotations']:
        if data['image_id'] in leak_files_id:
            data['id'] = data['id']+merge_js['annotations'][-1]['id']+1
            data['image_id'] = data['image_id']+merge_js['images'][-1]['id']+1
            leak_annos.append(data)

    leak_files.sort(key = lambda x : x['id'])
    leak_annos.sort(key = lambda x : x['image_id'])


    merge_js['images'].extend(leak_files)
    merge_js['annotations'].extend(leak_annos)
    # check len(merge_js['images']) == 195+ len(origin_images)
    print(len(merge_js['images']))
    print(merge_js['images'][-1])
    print(len(merge_js['annotations']))
    print(merge_js['annotations'][-1])

    tmp = json_path.split('.')
    tmp[0] += '+leakage'
    out_json_path = '.'.join(tmp)
    with open(out_json_path,'w') as w:
        json.dump(merge_js,w,indent=4)

if __name__ =='__main__':
    leak = leakage_list()
    merge_json(leak,'/opt/ml/input/data/train.json')
    merge_json(leak,'/opt/ml/input/data/train_all.json')
