import numpy as np
from PIL import Image 
import sys

A = np.array(Image.open(sys.argv[1]))
B = np.array(Image.open(sys.argv[2]))
for row_A, row_B in zip(A,B):
    for line1, line2 in zip(row_A, row_B):
        if np.array_equal(line1, line2):
            line2.fill(0)

result = Image.fromarray(B)

result.save('./ans_two.png')