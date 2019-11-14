#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert temperature from F to C or C to F. Input arguments are F/C and temperature to convert.
"""

import sys



if sys.argv[1]=="F":
    temp = (int(sys.argv[2])-32)*5/9
    print(f"{temp:.2f} C")

if sys.argv[1]=="C":
    temp = int(sys.argv[2]) *9/5 + 32
    print(round(temp,2),"F")
