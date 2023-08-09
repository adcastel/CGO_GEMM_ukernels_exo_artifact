from base_ukr import generate_optimized_ukr, generate_original_ukr


MR = 8
NR = 12
KC = 512
LANE = 4
alpha1=True
beta1=True

# Version of the paper CG0 2024
x = generate_optimized_ukr(MR, NR, KC, alpha1, beta1, LANE,windowing=1) #windowing version
print("VFINAL\n",x)
"""
NR = 8
## MR = 8 NR = 8
y = generate_optimized_ukr(MR, NR, KC, alpha1, beta1, LANE,windowing=1) #windowing version
print("VFINAL\n",y)
NR = 4
## MR = 8 NR = 4
z = generate_optimized_ukr(MR, NR, KC, alpha1, beta1, LANE,windowing=1) #windowing version
print("VFINAL\n",z)

MR = 4
NR = 12
z412 = generate_optimized_ukr(MR, NR, KC, alpha1, beta1, LANE,windowing=1) #windowing version
print("VFINAL\n",z412)


NR = 8
z48 = generate_optimized_ukr(MR, NR, KC, alpha1, beta1, LANE,windowing=1) #windowing version
print("VFINAL\n",z48)

NR = 4
z44 = generate_optimized_ukr(MR, NR, KC, alpha1, beta1, LANE,windowing=1) #windowing version
print("VFINAL\n",z44)
#Just for check
# if we want use fp16 
#w = generate_optimized_ukr(MR, NR, KC, alpha1, beta1, LANE,windowing=1, data="f16") #windowing version
#print(w)
"""
