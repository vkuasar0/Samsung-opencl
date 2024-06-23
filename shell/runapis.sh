#!/bin/bash
export LD_LIBRARY_PATH=/data/local/tmp
chmod -R 777 myModule
#echo -e "0 - add\n1 - cvtColor\n2 - crop\n3 - boxblur\n4 - gaussianblur\n5 - reshape\n6 - pyrup\n7 - pyrdown"
#for i in {1..$2}
for i in $(seq 1 $2)
do
 ./myModule $1
done
exit