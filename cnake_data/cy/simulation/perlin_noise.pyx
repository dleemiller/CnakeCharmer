# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""Perlin noise with fractal Brownian motion.

Keywords: perlin noise, fbm, procedural generation, noise, simulation, cython
"""

from libc.math cimport floor

from cnake_data.benchmarks import cython_benchmark

cdef int _p[512]

cdef void _init_perm() nogil:
    cdef int base[256]
    base[0]=151; base[1]=160; base[2]=137; base[3]=91; base[4]=90
    base[5]=15; base[6]=131; base[7]=13; base[8]=201; base[9]=95
    base[10]=96; base[11]=53; base[12]=194; base[13]=233; base[14]=7
    base[15]=225; base[16]=140; base[17]=36; base[18]=103; base[19]=30
    base[20]=69; base[21]=142; base[22]=8; base[23]=99; base[24]=37
    base[25]=240; base[26]=21; base[27]=10; base[28]=23; base[29]=190
    base[30]=6; base[31]=148; base[32]=247; base[33]=120; base[34]=234
    base[35]=75; base[36]=0; base[37]=26; base[38]=197; base[39]=62
    base[40]=94; base[41]=252; base[42]=219; base[43]=203; base[44]=117
    base[45]=35; base[46]=11; base[47]=32; base[48]=57; base[49]=177
    base[50]=33; base[51]=88; base[52]=237; base[53]=149; base[54]=56
    base[55]=87; base[56]=174; base[57]=20; base[58]=125; base[59]=136
    base[60]=171; base[61]=168; base[62]=68; base[63]=175; base[64]=74
    base[65]=165; base[66]=71; base[67]=134; base[68]=139; base[69]=48
    base[70]=27; base[71]=166; base[72]=77; base[73]=146; base[74]=158
    base[75]=231; base[76]=83; base[77]=111; base[78]=229; base[79]=122
    base[80]=60; base[81]=211; base[82]=133; base[83]=230; base[84]=220
    base[85]=105; base[86]=92; base[87]=41; base[88]=55; base[89]=46
    base[90]=245; base[91]=40; base[92]=244; base[93]=102; base[94]=143
    base[95]=54; base[96]=65; base[97]=25; base[98]=63; base[99]=161
    base[100]=1; base[101]=216; base[102]=80; base[103]=73; base[104]=209
    base[105]=76; base[106]=132; base[107]=187; base[108]=208; base[109]=89
    base[110]=18; base[111]=169; base[112]=200; base[113]=196; base[114]=135
    base[115]=130; base[116]=116; base[117]=188; base[118]=159; base[119]=86
    base[120]=164; base[121]=100; base[122]=109; base[123]=198; base[124]=173
    base[125]=186; base[126]=3; base[127]=64; base[128]=52; base[129]=217
    base[130]=226; base[131]=250; base[132]=124; base[133]=123; base[134]=5
    base[135]=202; base[136]=38; base[137]=147; base[138]=118; base[139]=126
    base[140]=255; base[141]=82; base[142]=85; base[143]=212; base[144]=207
    base[145]=206; base[146]=59; base[147]=227; base[148]=47; base[149]=16
    base[150]=58; base[151]=17; base[152]=182; base[153]=189; base[154]=28
    base[155]=42; base[156]=223; base[157]=183; base[158]=170; base[159]=213
    base[160]=119; base[161]=248; base[162]=152; base[163]=2; base[164]=44
    base[165]=154; base[166]=163; base[167]=70; base[168]=221; base[169]=153
    base[170]=101; base[171]=155; base[172]=167; base[173]=43; base[174]=172
    base[175]=9; base[176]=129; base[177]=22; base[178]=39; base[179]=253
    base[180]=19; base[181]=98; base[182]=108; base[183]=110; base[184]=79
    base[185]=113; base[186]=224; base[187]=232; base[188]=178; base[189]=185
    base[190]=112; base[191]=104; base[192]=218; base[193]=246; base[194]=97
    base[195]=228; base[196]=251; base[197]=34; base[198]=242; base[199]=193
    base[200]=238; base[201]=210; base[202]=144; base[203]=12; base[204]=191
    base[205]=179; base[206]=162; base[207]=241; base[208]=81; base[209]=51
    base[210]=145; base[211]=235; base[212]=249; base[213]=14; base[214]=239
    base[215]=107; base[216]=49; base[217]=192; base[218]=214; base[219]=31
    base[220]=181; base[221]=199; base[222]=106; base[223]=157; base[224]=184
    base[225]=84; base[226]=204; base[227]=176; base[228]=115; base[229]=121
    base[230]=50; base[231]=45; base[232]=127; base[233]=4; base[234]=150
    base[235]=254; base[236]=138; base[237]=236; base[238]=205; base[239]=93
    base[240]=222; base[241]=114; base[242]=67; base[243]=29; base[244]=24
    base[245]=72; base[246]=243; base[247]=141; base[248]=128; base[249]=195
    base[250]=78; base[251]=66; base[252]=215; base[253]=61; base[254]=156
    base[255]=180
    cdef int i
    for i in range(256):
        _p[i] = base[i]
        _p[i + 256] = base[i]

_init_perm()


cdef inline double fade(double t) nogil:
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


cdef inline double lerp(double t, double a, double b) nogil:
    return a + t * (b - a)


cdef inline double grad(int h, double x, double y, double z) nogil:
    cdef double u, v, r
    h = h & 15
    u = x if h < 8 else y
    v = y if h < 4 else (x if h == 12 or h == 14 else z)
    r = u if (h & 1) == 0 else -u
    r += v if (h & 2) == 0 else -v
    return r


cdef double noise(double x, double y, double z) nogil:
    cdef int xi, yi, zi
    cdef double u, v, w
    cdef int a, aa, ab, b, ba, bb

    xi = <int>floor(x) & 255
    yi = <int>floor(y) & 255
    zi = <int>floor(z) & 255
    x -= floor(x)
    y -= floor(y)
    z -= floor(z)
    u = fade(x)
    v = fade(y)
    w = fade(z)

    a = _p[xi] + yi
    aa = _p[a] + zi
    ab = _p[a + 1] + zi
    b = _p[xi + 1] + yi
    ba = _p[b] + zi
    bb = _p[b + 1] + zi

    return lerp(w,
        lerp(v,
            lerp(u, grad(_p[aa], x, y, z),
                     grad(_p[ba], x - 1, y, z)),
            lerp(u, grad(_p[ab], x, y - 1, z),
                     grad(_p[bb], x - 1, y - 1, z))),
        lerp(v,
            lerp(u, grad(_p[aa + 1], x, y, z - 1),
                     grad(_p[ba + 1], x - 1, y, z - 1)),
            lerp(u, grad(_p[ab + 1], x, y - 1, z - 1),
                     grad(_p[bb + 1], x - 1, y - 1, z - 1))))


cdef double fbm(double x, double y, double z, int octaves) nogil:
    cdef double amplitude = 1.0
    cdef double frequency = 1.0
    cdef double accum = 0.0
    cdef int i
    for i in range(octaves):
        accum += amplitude * noise(x * frequency, y * frequency, z * frequency)
        amplitude *= 0.5
        frequency *= 2.0
    return accum


@cython_benchmark(syntax="cy", args=(80,))
def perlin_noise(int n):
    """Evaluate FBM Perlin noise on an n*n grid.

    Args:
        n: Grid dimension.

    Returns:
        Tuple of (total_sum, min_val, max_val, sample_at_half).
    """
    cdef double total = 0.0
    cdef double min_val = 1e300
    cdef double max_val = -1e300
    cdef double sample_val = 0.0
    cdef int half = n // 2
    cdef int i, j
    cdef double val

    with nogil:
        for i in range(n):
            for j in range(n):
                val = fbm(i * 0.1, j * 0.1, 0.5, 6)
                total += val
                if val < min_val:
                    min_val = val
                if val > max_val:
                    max_val = val
                if i == half and j == half:
                    sample_val = val

    return (total, min_val, max_val, sample_val)
