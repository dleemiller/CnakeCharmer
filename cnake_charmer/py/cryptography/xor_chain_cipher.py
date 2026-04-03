"""
XOR chain cipher with CBC-like chaining within 8-byte blocks.

Keywords: cryptography, xor, chain, cipher, cbc, chaining, benchmark
"""

from cnake_charmer.benchmarks import python_benchmark

N = 10000


@python_benchmark(args=(N,))
def xor_chain_cipher(n: int) -> tuple:
    """Apply XOR chain cipher on n blocks of 8 bytes with 16-round key schedule.

    Each block of 8 bytes is processed through 16 rounds. In each round,
    byte 0 is taken from data at a key-indexed offset, and subsequent bytes
    are XORed with the previous cipher byte (CBC-like chaining).

    Args:
        n: Number of 8-byte blocks to process.

    Returns:
        Tuple of (total_checksum, first_block_byte0, last_block_byte7).
    """
    # Key schedule: 16 rounds of key offsets
    keys = [0] * 16
    for i in range(16):
        keys[i] = (i * 37 + 11) % 8

    total_checksum = 0
    first_block_byte0 = 0
    last_block_byte7 = 0

    for i in range(n):
        # Generate block deterministically
        data = [0] * 8
        for j in range(8):
            data[j] = (i * 7 + j * 13 + 42) % 256

        # Apply 16 rounds of chained XOR
        cipher = [0] * 8
        for k in range(16):
            key = keys[k]
            cipher[0] = data[key % 8]
            for j in range(1, 8):
                cipher[j] = data[(key + j) % 8] ^ cipher[j - 1]
            # Copy cipher back to data
            for j in range(8):
                data[j] = cipher[j]

        # Accumulate checksum
        for j in range(8):
            total_checksum += data[j]

        if i == 0:
            first_block_byte0 = data[0]
        if i == n - 1:
            last_block_byte7 = data[7]

    return (total_checksum, first_block_byte0, last_block_byte7)
