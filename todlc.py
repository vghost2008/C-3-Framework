import os
import sys
print(sys.path)
onnx_path = '/home/stone/ai/mldata/crowd_density_counting/pb/cdc.onnx'
save_path = '/home/stone/ai/mldata/crowd_density_counting/pb/cdc.dlc'
path = ""
for x in sys.path[:-1]:
    path+=x+":"
path+=sys.path[-1]
cmd = "/root/snpe-1.45.2.2427/bin/x86_64-linux-clang/snpe-onnx-to-dlc --input_network {} --output_path {}".format(path,onnx_path,save_path)
print(cmd)
os.system(cmd)
cmd = "/root/snpe-1.45.2.2427/bin/x86_64-linux-clang/snpe-dlc-info -i {}".format(save_path)
print(cmd)
os.system(cmd)

