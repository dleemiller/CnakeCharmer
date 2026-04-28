import ipaddress


def find_family(addr):
    """Return 4 for IPv4 and 6 for IPv6."""
    return 6 if ":" in addr else 4


def inet_fix(addr, masklen):
    """Normalize an IP string to its network address at masklen."""
    net = ipaddress.ip_network(f"{addr}/{masklen}", strict=False)
    return str(net.network_address)


def prefix_addrs(addr, size):
    """Enumerate host offsets within a small prefix-size delta.

    Behavior follows original trace intent: IPv4 uses 2**size addresses,
    IPv6 uses size**2 addresses, varying only the last byte.
    """
    ip = ipaddress.ip_address(addr)
    total = (2**size) if ip.version == 4 else (size**2)
    packed = bytearray(ip.packed)
    base = packed[-1]

    out = []
    for i in range(total):
        packed[-1] = (base + i) & 0xFF
        out.append(str(ipaddress.ip_address(bytes(packed))))
    return out


def otherside(addr, num):
    """Return opposite endpoint in /31-/30 (IPv4) or /127-/126 (IPv6)-style pairs."""
    if num not in (2, 4):
        raise ValueError(f"Invalid number of addresses in prefix {num}")

    ip = ipaddress.ip_address(addr)
    packed = bytearray(ip.packed)
    last = packed[-1]

    if num == 2:
        packed[-1] = (last + 1) & 0xFF if (last % 2 == 0) else (last - 1) & 0xFF
    elif last % 4 == 1:
        packed[-1] = (last + 1) & 0xFF
    elif last % 4 == 2:
        packed[-1] = (last - 1) & 0xFF
    else:
        suffix = "/30" if ip.version == 4 else "/126"
        raise ValueError(f"Invalid host address {addr} for {suffix} prefix")

    return str(ipaddress.ip_address(bytes(packed)))


def valid_asn(asn):
    """Validate ASN per reserved/legacy exclusions from original trace."""
    return asn != 23456 and (0 < asn < 64496 or 131071 < asn < 400000)
