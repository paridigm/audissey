# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 23:34:19 2016

@author: Tom
"""

"""
x is the index
y is the value
"""
for x, y in enumerate( reversed(list(range(10))) ):
    print(x, ",", y)
    x = x + 1

print()
print("NEXT")

"""
enumerate backwards
"""
for x, y in reversed( list(enumerate( range(10) )) ):
    print(x, ",", y)


for x in range(9, -1, -1):
    print(x)

for x in range(10):
    print(x)
    