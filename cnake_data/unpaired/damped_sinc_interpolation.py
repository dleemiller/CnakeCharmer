import math


def get_pixel_shifts(sci_wl, sky_wl):
    """Compute integer pixel anchor and fractional shift for each sky wavelength."""
    nsci = len(sci_wl)
    nsky = len(sky_wl)
    mpix = [0] * nsky
    dpix = [0.0] * nsky

    j = 1
    for ii in range(nsky):
        while (j < nsci - 1) and (sci_wl[j] < sky_wl[ii]):
            j += 1
        dlam = sci_wl[j] - sci_wl[j - 1]
        shift = (sky_wl[ii] - sci_wl[j]) / dlam
        mpix[ii] = int(math.floor(shift + 0.5)) + j
        dpix[ii] = shift - float(mpix[ii]) + j
    return mpix, dpix


def calc_sinc(nsinc, sincrad, sincdamp):
    nsinch = int(nsinc / 2)
    dx = float((2.0 * sincrad) / (nsinc - 1.0))
    out = [0.0] * nsinc

    for kk in range(nsinch):
        x = (kk - nsinch) * dx
        if math.ceil(x) == x:
            out[kk] = 0.0
        else:
            out[kk] = math.exp(-1.0 * ((x / sincdamp) ** 2)) * math.sin(math.pi * x) / (math.pi * x)
    out[nsinch] = 1.0
    for kk in range(nsinch + 1, nsinc):
        out[kk] = out[nsinc - kk - 1]
    return out


def _interpolate_core(src_wl, dst_wl, src_flux, sincrad, sincbin, sincdamp, radius):
    nsinc = int((2 * sincrad) * sincbin + 1)
    psinc = calc_sinc(nsinc, sincrad, sincdamp)

    mpix, dpix = get_pixel_shifts(src_wl, dst_wl)
    nsrc = len(src_wl)
    ndst = len(dst_wl)
    out = [0.0] * ndst

    nkpix = int(2 * radius + 1)
    k_offset = float(sincrad) - radius
    pmin = mpix[0]
    pmax = mpix[ndst - 1]

    for ii in range(ndst):
        shift = dpix[ii] - k_offset
        kernel = [0.0] * nkpix
        tsum = 0.0

        for kk in range(nkpix):
            x = float(kk - shift) * sincbin
            low = psinc[int(x)]
            high = psinc[int(x + 1)]
            kernel[kk] = low + (high - low) * (x - math.floor(x))
            tsum += kernel[kk]

        rsum = 1.0 / tsum
        for kk in range(nkpix):
            kernel[kk] *= rsum

        if ii == 0 and pmin > -1.0 * radius:
            npix = int(pmin + radius + 1)
            sign = -1
        elif ii == ndst - 1 and pmax < nsrc - 1 + radius:
            npix = int(nsrc - pmax + radius)
            sign = 1
        else:
            npix = 1
            sign = 1

        for hh in range(npix):
            j = mpix[ii] - int(radius) + (sign * hh)
            for kk in range(nkpix):
                jj = j + kk
                if 0 <= jj < nsrc:
                    out[ii] += src_flux[jj] * kernel[kk]

    return out


def interpolate(sci_wl, sky_wl, sky_flux):
    """Interpolate sky_flux(sky_wl) onto sci_wl sampling."""
    return _interpolate_core(
        sky_wl, sci_wl, sky_flux, sincrad=6.0, sincbin=10000, sincdamp=3.25, radius=5.0
    )


def interpolate_os(sky_wl, sci_wl, sci_flux):
    """Oversampled interpolation variant."""
    return _interpolate_core(
        sci_wl, sky_wl, sci_flux, sincrad=4.0, sincbin=10000, sincdamp=1.15, radius=2.0
    )
