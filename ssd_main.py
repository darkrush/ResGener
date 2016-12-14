import sys
from layers import *

def main(argv):
    file = open(argv[1],'w')
    s='name: "ResNet_on_ilsvrc"'
    s=s+gen_img_data('ilsvrc',['data','label'],'TRAIN','true',256,'data/ilsvrc12/imagenet_mean.binaryproto','data/ilsvrc12/train_shuff.txt',32,280)
    s=s+gen_img_data('ilsvrc',['data','label'],'TEST','false',256,'data/ilsvrc12/imagenet_mean.binaryproto','data/ilsvrc12/val_full.txt',50,280)

    s=s+gen_conv('conv1','data','conv1',16,3,0,1,'true','xavier')
    s=s+gen_conv('conv2','conv1','conv2',32,3,0,1,'true','xavier')
    s=s+gen_pool('pool0','conv2','pool0','MAX',2,2)
    128

    s=s+res_unit('u1_1','pool0','u1_1',32,[32,64])
    s=s+res_unit('u1_2','u1_1','u1_2',64,[32,64])
    s=s+gen_pool('pool1','u1_2','pool1','MAX',2,2)
    64

    s=s+res_unit('u2_1','pool1','u2_1',64,[64,128])
    s=s+res_unit('u2_2','u2_1','u2_2',128,[64,128])
    s=s+res_unit('u2_3','u2_2','u2_3',128,[64,128])
    s=s+gen_pool('pool2','u2_3','pool2','MAX',2,2)
    32

    s=s+res_unit('u3_1','pool2','u3_1',128,[128,256])
    s=s+res_unit('u3_2','u3_1','u3_2',256,[128,256])
    s=s+res_unit('u3_3','u3_2','u3_3',256,[128,256])
    s=s+gen_pool('pool3','u3_3','pool3','MAX',2,2)
    16

    s=s+res_unit('u4_1','pool3','u4_1',256,[256,512])
    s=s+res_unit('u4_2','u4_1','u4_2',512,[256,512])
    s=s+res_unit('u4_3','u4_2','u4_3',512,[256,512])
    s=s+gen_pool('pool4','u4_3','pool4','MAX',2,2)
    8

    s=s+res_unit('u5_1','pool4','u5_1',512,[512,1024])
    s=s+res_unit('u5_2','u5_1','u5_2',1024,[512,1024])
    s=s+res_unit('u5_3','u5_2','u5_3',1024,[512,1024])
    s=s+gen_pool('pool5','u5_3','pool5','AVE',-1,2)
    4

    s=s+gen_conv('conv1k','pool5','conv1k',1000,3,1,1,'true','xavier')
    s=s+gen_pool('pool_ave','conv1k','pool_ave','AVE',-1,2)
    s=s+gen_acc('acc',['pool_ave','label'],'acc','TEST')
    s=s+gen_softmaxloss('loss',['pool_ave','label'],'loss')
    file.write(s)
    file.close()
    return

if __name__=='__main__':
    main(sys.argv)
