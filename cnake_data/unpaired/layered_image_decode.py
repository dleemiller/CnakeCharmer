def decode_layered_image(digits, width=25, height=6):
    """Decode layered image stream using 0/1/2 transparency composition."""
    layer_size = width * height
    picture = [0 for _ in range(layer_size)]

    for i, ch in enumerate(digits):
        cur = i % layer_size
        if i < layer_size or picture[cur] == 2:
            picture[cur] = int(ch)

    return "".join(str(x) for x in picture)
