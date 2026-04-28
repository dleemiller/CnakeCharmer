import multiprocessing
import random
import statistics

MINRELAYTXFEE = 1.0


def calc_stranding_single(txs):
    """Compute stranding feerate for one reverse-sorted sample."""
    k = 0
    maxk = 0
    maxidx = len(txs) - 1
    sfr = float("inf")

    for idx in range(maxidx + 1):
        tx = txs[idx]
        if tx[1] is True:
            k += 1
        else:
            k -= 1

        if idx < maxidx and txs[idx + 1][0] == tx[0]:
            continue

        if k > maxk:
            maxk = k
            sfr = tx[0]

    return sfr


def bootstrap_sample(txs):
    n = len(txs)
    sample = [txs[int(random.random() * n)] for _ in range(n)]
    sample.sort(reverse=True)
    return sample


def _processwork(args):
    txs, n = args
    return [calc_stranding_single(bootstrap_sample(txs)) for _ in range(n)]


def get_bs_estimates(txs, n_bootstrap, numprocesses):
    if numprocesses == 1:
        return [calc_stranding_single(bootstrap_sample(txs)) for _ in range(n_bootstrap)]

    workers = multiprocessing.Pool(processes=numprocesses)
    n_chunk = n_bootstrap // numprocesses
    result = workers.map_async(_processwork, [(txs, n_chunk)] * numprocesses)
    workers.close()
    bs_estimates = sum(result.get(), [])
    workers.join()
    return bs_estimates


def calc_stranding_feerate(txs, bootstrap=True, numprocesses=None, minrelaytxfee=MINRELAYTXFEE):
    if numprocesses is None:
        numprocesses = multiprocessing.cpu_count()

    if not txs:
        raise ValueError("Empty txs list")

    txs = list(txs)
    txs.sort(key=lambda x: x[0], reverse=True)
    sfr = calc_stranding_single(txs)

    abovek = aboven = belowk = belown = 0
    alt_bias_ref = None
    for tx in txs:
        if tx[0] >= sfr:
            abovek += tx[1]
            aboven += 1
        else:
            if alt_bias_ref is None:
                alt_bias_ref = tx[0]
            belowk += int(not tx[1])
            belown += 1

    if alt_bias_ref is None:
        alt_bias_ref = minrelaytxfee

    if bootstrap and sfr != float("inf"):
        n_bootstrap = 1000
        bs_estimates = get_bs_estimates(txs, n_bootstrap, numprocesses)

        if not any(b == float("inf") for b in bs_estimates):
            mean = statistics.fmean(bs_estimates)
            std = statistics.pstdev(bs_estimates)
            bias = mean - alt_bias_ref
            alt_bias = mean - sfr
            if abs(alt_bias) > abs(bias):
                bias = alt_bias
        else:
            bias = std = mean = float("inf")
    else:
        bias = std = mean = float("inf")

    return {
        "sfr": sfr,
        "bias": bias,
        "mean": mean,
        "std": std,
        "abovekn": (abovek, aboven),
        "belowkn": (belowk, belown),
        "altbiasref": alt_bias_ref,
    }
