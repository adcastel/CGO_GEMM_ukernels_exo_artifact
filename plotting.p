
set terminal postscript color font "Helvetica, 16" enhanced 
#set key autotitle columnhead
set style data histograms
set style histogram cluster gap 3
set style fill solid
set datafile separator ";"
set boxwidth 1
set grid ytics

set key top left box Left reverse width 2
#set yrange [0:35]

set title "Gemm algorithm - Single core" font "Helvetica-Bold,22"
set output "plots/figure14.eps"
set ylabel font ",22" "GFLOPS" 
set xlabel font ",22" "Matrix dimensions (M=N=K)"
plot "./gemm_blis_family/output/square.dat" using 2:xtic(1) t "ALG+NEON" lc 6, \
      '' u 3 t "ALG+BLIS" lc 2, \
      '' u 5 t "BLIS" lc 4, \
      '' u 4 t "ALG+EXO" lc 1, \

set title "ResNet50v1.5 - Single core" font "Helvetica-Bold,22"
set output "plots/figure15.eps"
set ylabel font ",22" "GFLOPS" 
set xlabel font ",22" "# of layer"
plot "./gemm_blis_family/output/resnet50.dat" using 2:xtic(1) t "ALG+NEON" lc 6, \
      '' u 3 t "ALG+BLIS" lc 2, \
      '' u 5 t "BLIS" lc 4, \
      '' u 4 t "ALG+EXO" lc 1, \

set title "VGG16 - Single core" font "Helvetica-Bold,22"
set output "plots/figure17.eps"
set ylabel font ",22" "GFLOPS" 
set xlabel font ",22" "# of layer"
plot "./gemm_blis_family/output/vgg16.dat" using 2:xtic(1) t "ALG+NEON" lc 6, \
      '' u 3 t "ALG+BLIS" lc 2, \
      '' u 5 t "BLIS" lc 4, \
      '' u 4 t "ALG+EXO" lc 1, \

unset datafile separator
set title "Microkernel performance - Single core" font "Helvetica-Bold,22"
set output "plots/figure13.eps"
set ylabel font ",22" "GFLOPS" 
set xlabel font ",22" "Microkernel dimensions (mr x nr)"
plot "./driver_microkernel/output/ukernel_results.dat" using 2:xtic(1) t "NEON" lc 6, \
      '' u 3 t "BLIS" lc 4, \
      '' u 4 t "EXO" lc 1

