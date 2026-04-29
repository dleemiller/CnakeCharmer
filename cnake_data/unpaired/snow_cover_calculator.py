"""Daily snow-cover and soil-surface temperature update."""

from __future__ import annotations

import math


def model_snow_cover(
    iCropResidues,
    iSoilTempArray,
    iTempMax,
    iTempMin,
    iRadiation,
    Albedo,
    iRAIN,
    SnowWaterContent,
    SoilSurfaceTemperature,
    iPotentialSoilEvaporation,
    iLeafAreaIndex,
    AgeOfSnow,
):
    tiCropResidues = iCropResidues * 10.0
    tiSoilTemp0 = iSoilTempArray[0]
    TMEAN = 0.5 * (iTempMax + iTempMin)
    TAMPL = 0.5 * (iTempMax - iTempMin)
    DST = TMEAN + (TAMPL * (iRadiation * (1 - Albedo) - 14) / 20)

    if iRAIN > 0 and (tiSoilTemp0 < 1 or SnowWaterContent > 3 or SoilSurfaceTemperature < 0):
        SnowWaterContent += iRAIN

    snow_iso = 1.0
    if tiCropResidues < 10:
        snow_iso = tiCropResidues / (tiCropResidues + math.exp(5.34 - (2.4 * tiCropResidues)))

    if SnowWaterContent < 1e-10:
        snow_iso *= 0.85
        soil_surface = 0.5 * (DST + ((1 - snow_iso) * DST) + (snow_iso * tiSoilTemp0))
    else:
        snow_iso = max(
            SnowWaterContent / (SnowWaterContent + math.exp(0.47 - (0.62 * SnowWaterContent))),
            snow_iso,
        )
        soil_surface = (1 - snow_iso) * DST + (snow_iso * tiSoilTemp0)

    if SnowWaterContent == 0 and not (iRAIN > 0 and tiSoilTemp0 < 1):
        SnowWaterContent = 0.0
    else:
        EAJ = 0.5
        if SnowWaterContent < 5:
            EAJ = math.exp(-max((0.4 * iLeafAreaIndex), (0.1 * (tiCropResidues + 0.1))))
        snow_evap = iPotentialSoilEvaporation * EAJ
        age_factor = AgeOfSnow / (AgeOfSnow + math.exp(5.34 - (2.395 * AgeOfSnow)))
        SNPKT = 0.3333 * (2 * min(soil_surface, tiSoilTemp0) + iTempMax)
        if TMEAN > 0:
            snow_melt = max(
                0.0, math.sqrt(iTempMax * iRadiation) * (1.52 + (0.54 * age_factor * SNPKT))
            )
        else:
            snow_melt = 0.0

        total_loss = snow_melt + snow_evap
        if total_loss > SnowWaterContent and total_loss > 0:
            snow_melt = snow_melt / total_loss * SnowWaterContent
            snow_evap = snow_evap / total_loss * SnowWaterContent

        SnowWaterContent -= snow_melt + snow_evap
        if SnowWaterContent < 0:
            SnowWaterContent = 0.0

        if SnowWaterContent < 5:
            AgeOfSnow = 0
        else:
            AgeOfSnow += 1

    return {
        "SnowWaterContent": SnowWaterContent,
        "SnowIsolationIndex": snow_iso,
        "SoilSurfaceTemperature": soil_surface,
        "AgeOfSnow": AgeOfSnow,
    }
