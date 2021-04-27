import os
onnx_path = '/home/stone/ai/mldata/crowd_density_counting/pb/cdc.onnx'
save_path = '/home/stone/ai/mldata/crowd_density_counting/pb/cdc.dlc'
cmd = "/root/snpe-1.45.2.2427/bin/x86_64-linux-clang/snpe-onnx-to-dlc --input_network {} --output_path {}".format(onnx_path,save_path)
print(cmd)
os.system(cmd)
