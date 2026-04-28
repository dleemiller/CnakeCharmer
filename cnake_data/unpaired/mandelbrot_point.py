def veriPoint(re, im, power=17, thresh=100):
    x = complex(0, 0)
    c = complex(re, im)
    for _ in range(thresh):
        x = x**power + c
        if abs(x.real) > 2.0 or abs(x.imag) > 2.0:
            return False
    return True


def veriPoint_c(re, im):
    thresh = 30
    old_re = 0.0
    old_im = 0.0
    for _ in range(thresh):
        new_re = old_re * old_re - old_im * old_im + im
        new_im = 2 * old_re * old_im + re
        old_re = new_re
        old_im = new_im
        if new_im**2.0 + new_re**2.0 > 4.0:
            return False
    return True
