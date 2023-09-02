#/bin/bash

OUT="output"
FAMILY="gemm_blis_family/${OUT}"

for cnn in "resnet50" "vgg16" "square";
do
	echo "#Layer;BLIS;FAMILY+BLIS;FAMILY;FAMILY+EXO" > ${FAMILY}/${cnn}.dat
	paste -d ";" ${FAMILY}/${cnn}_BLIS.dat ${FAMILY}/${cnn}_FAMILY_BLIS.dat ${FAMILY}/${cnn}_FAMILY.dat ${FAMILY}/${cnn}_FAMILY_EXO.dat | cut -d";" -f1,5,10,15,20 >> ${FAMILY}/${cnn}.dat

done

mkdir -p plots

gnuplot plotting.p 

