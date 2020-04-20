from enum import Enum
from random import uniform
import numpy as np
import math as mth
import cv2 as cv


def scale_point(number, minimal, maximal, new_minimal, new_maximal):
    return ((number - minimal) / (maximal - minimal)) * (new_maximal - new_minimal) + new_minimal


class Colors(Enum):
    blue = 1
    red = 4
    green = 2
    navy_blue = 9
    yellow = 14


def ppm_writer(f):
    f.write("P3 \n")
    f.write("500 500 \n")
    f.write("255 \n")


def f_p(s):
    result = 0

    if s <= 0.0:
        result = 0
    elif s > 0.0:
        result = 1

    if result == 0:
        return Colors.blue
    elif result == 1:
        return Colors.red


def f_linear(s):
    result = s
    if result < -2.0:
        return Colors.navy_blue
    elif -2.0 <= result < 0.0:
        return Colors.blue
    elif 0.0 <= result < 2.0:
        return Colors.green
    elif result >= 2.0:
        return Colors.red


def f_sig(s):
    result = 1.0 / (1.0 + mth.exp(-s))

    if 0 <= result <= 0.25:
        return Colors.yellow
    elif 0.25 < result <= 0.5:
        return Colors.blue
    elif 0.5 < result <= 0.75:
        return Colors.green
    elif 0.75 < result <= 1:
        return Colors.red


def f_p2(s):
    if s <= 0.0:
        return 0
    elif s > 0.0:
        return 1


def f_linear2(s):
    return s


def f_sig2(s):
    return 1.0 / (1.0 + mth.exp(-s))


def print_result(y, f):
    if y == Colors.red:
        f.write("255 0 0 ")
    elif y == Colors.blue:
        f.write("0 0 255 ")
    elif y == Colors.green:
        f.write("0 255 0 ")
    elif y == Colors.yellow:
        f.write("250 210 1 ")
    elif y == Colors.navy_blue:
        f.write("2 178 161 ")


def temp_result(i, j, bias, weight, x1):
    result = 0.0

    result += scale_point(i, 0, 500, x1[0], x1[1]) * weight[0]
    result += scale_point(j, 0, 500, x1[0], x1[1]) * weight[1]
    result += bias * weight[2]
    return result


def temp_result2(i, j, bias, weight, x1):
    result1 = 0.0
    result2 = 0.0

    result1 += scale_point(i, 0, 500, x1[0], x1[1]) * weight[0]
    result1 += scale_point(j, 0, 500, x1[0], x1[1]) * weight[1]
    result1 += bias * weight[4]
    result2 += scale_point(i, 0, 500, x1[0], x1[1]) * weight[2]
    result2 += scale_point(j, 0, 500, x1[0], x1[1]) * weight[3]
    result2 += bias * weight[5]

    return result1, result2


def first(bias, weight, x1, f):
    for i in range(0, 500):
        for j in range(0, 500):
            result = temp_result(i, j, bias, weight, x1)
            y = f_p(result)

            print_result(y, f)


def second(bias, weight, x1, f):
    for i in range(0, 500):
        for j in range(0, 500):
            result = temp_result(i, j, bias, weight, x1)
            y = f_linear(result)

            print_result(y, f)


def third(bias, weight, x1, f):
    for i in range(0, 500):
        for j in range(0, 500):
            result = temp_result(i, j, bias, weight, x1)
            y = f_sig(result)

            print_result(y, f)


def fourth(bias, weight, x1, f):
    for i in range(0, 500):
        for j in range(0, 500):
            result1, result2 = temp_result2(i, j, bias, weight, x1)

            result1 = f_p2(result1)
            result2 = f_p2(result2)

            result3 = f_p(result1 * weight[6] + result2 * weight[7] + bias * weight[8])

            print_result(result3, f)


def fifth(bias, weight, x1, f):
    for i in range(0, 500):
        for j in range(0, 500):
            result1, result2 = temp_result2(i, j, bias, weight, x1)

            result1 = f_linear2(result1)
            result2 = f_linear2(result2)

            result3 = f_linear(result1 * weight[6] + result2 * weight[7] + bias * weight[8])

            print_result(result3, f)


def sixth(bias, weight, x1, f):
    for i in range(0, 500):
        for j in range(0, 500):
            result1, result2 = temp_result2(i, j, bias, weight, x1)

            result1 = f_sig2(result1)
            result2 = f_sig2(result2)

            result3 = f_sig(result1 * weight[6] + result2 * weight[7] + bias * weight[8])

            print_result(result3, f)


def main():
    x1 = np.linspace(-5.0, 5.0, 2)
    weight = np.zeros(10, dtype=float)

    for q in range(weight.size):
        weight[q] = uniform(-5.0, 5.0)

    print("1 - neurone 2 inputs, treshold f()")
    print("2 - neurone 2 inputs, linear f()")
    print("3 - neurone 2 inputs, sigmoidal f()")
    print("4 - two-layer 2 neurone inputs and 1 neurone output, treshold f()")
    print("5 - two-layer 2 neurone inputs and 1 neurone output, linear f()")
    print("6 - two-layer 2 neurone inputs and 1 neurone output, sigmoidal f()")
    print("7 - All cases")
    choose = int(input("Case selection (1-7) "))

    bias = float(input("Bias value "))

    if choose != 7:
        f = open("out.ppm", "w+")
        f.write("P3 \n")
        f.write("500 500 \n")
        f.write("255 \n")

        if choose == 1:
            first(bias, weight, x1, f)
        elif choose == 2:
            second(bias, weight, x1, f)
        elif choose == 3:
            third(bias, weight, x1, f)
        elif choose == 4:
            fourth(bias, weight, x1, f)
        elif choose == 5:
            fifth(bias, weight, x1, f)
        elif choose == 6:
            sixth(bias, weight, x1, f)
    elif choose == 7:
        f1 = open("out1.ppm", "w+")
        ppm_writer(f1)
        first(bias, weight, x1, f1)
        f1.close()
        f2 = open("out2.ppm", "w+")
        ppm_writer(f2)
        second(bias, weight, x1, f2)
        f2.close()
        f3 = open("out3.ppm", "w+")
        ppm_writer(f3)
        third(bias, weight, x1, f3)
        f3.close()
        f4 = open("out4.ppm", "w+")
        ppm_writer(f4)
        fourth(bias, weight, x1, f4)
        f4.close()
        f5 = open("out5.ppm", "w+")
        ppm_writer(f5)
        fifth(bias, weight, x1, f5)
        f5.close()
        f6 = open("out6.ppm", "w+")
        ppm_writer(f6)
        sixth(bias, weight, x1, f6)
        f6.close()

    if choose != 7:
        f.close()
        i = cv.imread('out.ppm')
        cv.imwrite('out.png', i)


if __name__ == "__main__":
    main()
