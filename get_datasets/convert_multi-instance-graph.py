import os



for dataset in ['products', 'arxiv', 'reddit', 'reddit-body', 'AS']:
    for instance in [2, 3, 4]:
        cmd = f'python convert_graph_sample.py --dataset {dataset} --num-instance {instance}'
        #print(cmd)
        os.system(cmd)