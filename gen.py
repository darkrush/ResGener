import sys

def gen_relu(name,in_blob,out_blob):
    s='layer {\n'
    s=s+'name: "'+name+'"\n'
    s=s+'type: "ReLU"\n'
    s=s+'bottom: "'+in_blob+'"\n'
    s=s+'top: "'+out_blob+'"\n'
    s=s+'}\n'
    return s

def gen_scale(name,in_blob,out_blob,bias_term):
    s='layer {\n'
    s=s+'name: "'+name+'"\n'
    s=s+'type: "Scale"\n'
    s=s+'bottom: "'+in_blob+'"\n'
    s=s+'top: "'+out_blob+'"\n'
    s=s+'scale_param {\n'
    s=s+'bias_term: '+bias_term+'\n'
    s=s+'}\n'
    s=s+'}\n'
    return s

def gen_BN(name,in_blob,out_blob,phase):
    s='layer {\n'
    s=s+'name: "'+name+'"\n'
    s=s+'type: "BatchNorm"\n'
    s=s+'bottom: "'+in_blob+'"\n'
    s=s+'top: "'+out_blob+'"\n'
    s=s+'include {\n'
    s=s+'phase: '+phase+'\n'
    s=s+'}\n'
    s=s+'batch_norm_param {\n'
    if(phase=='TEST'):
        s=s+'use_global_stats: true\n'
    else:
        s=s+'use_global_stats: false\n'
    s=s+'}\n'
    s=s+'}\n'
    return s

def gen_pool(name,in_blob,out_blob,pool_type,kernal,stride):
    s='layer {\n'
    s=s+'name: "'+name+'"\n'
    s=s+'type: "Pooling"\n'
    s=s+'bottom: "'+in_blob+'"\n'
    s=s+'top: "'+out_blob+'"\n'
    s=s+'pooling_param {\n'
    s=s+'pool: '+pool_type+'\n'
    if(kernal==-1):
        s=s+'global_pooling: true\n'
    else:
        s=s+'kernel_size: '+'%d'%kernal +'\n'
        s=s+'stride: '+'%d'%stride +'\n'

    s=s+'}\n'
    s=s+'}\n'
    return s

def gen_eltwise(name,in_blob,out_blob):
    s='layer {\n'
    s=s+'name: "'+name+'"\n'
    s=s+'type: "Eltwise"\n'
    s=s+'bottom: "'+in_blob[0]+'"\n'
    s=s+'bottom: "'+in_blob[1]+'"\n'
    s=s+'top: "'+out_blob+'"\n'
    s=s+'}\n'
    return s

def gen_tile(name,in_blob,out_blob,axis,tiles):
    s='layer {\n'
    s=s+'name: "'+name+'"\n'
    s=s+'type: "Tile"\n'
    s=s+'bottom: "'+in_blob+'"\n'
    s=s+'top: "'+out_blob+'"\n'
    s=s+'tile_param {\n'
    s=s+'axis: '+'%d'%axis+'\n'
    s=s+'tiles: '+'%d'%tiles+'\n'
    s=s+'}\n'
    s=s+'}\n'
    return s


def gen_conv(name,in_blob,out_blob,out_channel,kernel_size,pad,stride,bias_term,weight_filler_type):
    s='layer {\n'
    s=s+'name: "'+name+'"\n'
    s=s+'type: "Convolution"\n'
    s=s+'bottom: "'+in_blob+'"\n'
    s=s+'top: "'+out_blob+'"\n'
    s=s+'convolution_param {\n'
    s=s+'num_output: '+'%d'%out_channel+'\n'
    s=s+'kernel_size: '+'%d'%kernel_size+'\n'
    s=s+'pad: '+'%d'%pad+'\n'
    s=s+'stride: '+'%d'%stride+'\n'
    s=s+'bias_term: '+bias_term+'\n'
    s=s+'weight_filler {\n'
    s=s+'type: "'+weight_filler_type+'"\n'
    s=s+'}\n'
    s=s+'}\n'
    s=s+'}\n'
    return s

def res_unit(unit_name,in_blob,out_blob,in_channel,channels):
    s=''
    s=s+gen_BN(unit_name+'b1',in_blob,unit_name+'b1','TRAIN')
    s=s+gen_BN(unit_name+'b1',in_blob,unit_name+'b1','TEST')
    s=s+gen_scale(unit_name+'s1',unit_name+'b1',unit_name+'s1','true')
    s=s+gen_relu(unit_name+'r1',unit_name+'s1',unit_name+'r1')
    s=s+gen_conv(unit_name+'c1',unit_name+'r1',unit_name+'c1',channels[0],3,1,1,'false','xavier')
    s=s+gen_BN(unit_name+'b2',unit_name+'c1',unit_name+'b2','TRAIN')
    s=s+gen_BN(unit_name+'b2',unit_name+'c1',unit_name+'b2','TEST')
    s=s+gen_scale(unit_name+'s2',unit_name+'b2',unit_name+'s2','true')
    s=s+gen_relu(unit_name+'r2',unit_name+'s2',unit_name+'r2')
    s=s+gen_conv(unit_name+'c2',unit_name+'r2',unit_name+'c2',channels[1],3,1,1,'false','xavier')
    s=s+gen_tile(unit_name+'t1',unit_name+'c2',unit_name+'t1',1,channels[1]/in_channel)
    print channels[1]/in_channel
    s=s+gen_eltwise(unit_name+'e1',[unit_name+'t1',in_blob],out_blob)


    return s


def main(argv):
    file = open(argv[1],'w')
    file.write(res_unit('u1','in_blob','out_blob',16,[16,32]))
    file.close()
    return

if __name__=='__main__':
    main(sys.argv)
