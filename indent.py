import sys

def main(argv):
  in_file = open(argv[1],'r')
  count=0
  s=''
  for line in in_file:
    count=count-line.count('}')
    s=s+count*'  '+line.strip()+'\n'
    count=count+line.count('{')
  in_file.close()
  out_file = open(argv[2],'w')
  out_file.write(s)
  out_file.close()


if __name__=='__main__':
  main(sys.argv)
