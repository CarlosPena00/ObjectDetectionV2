for name in $(cat NameList.txt);do ls $name | cat -n | while read n f; do mv "$name$f" "/home/kaka/Desktop/ObjectDetection/Data/$name$n.JPEG"; done;done;
