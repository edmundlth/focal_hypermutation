
import os
import gzip
import warnings
from typing import TextIO, Optional

import numpy as np
import pandas as pd
from Bio.bgzf import BgzfWriter

#########
# Compressed file I/O handle
#########
def is_gzip(filepath: str) -> bool:
    """
    Check if a the file specified by filepath
    is a gzipped file. We do this by checking
    whether the first two bytes is the magic
    numbers included in the file header.
    We check them bytewise to avoid issue with
    endianness.

    Note that this wouldn't detect if the gzip
    file is a concatenation of multiple "members".
    See gzip specification at:
     https://www.ietf.org/rfc/rfc1952.txt
    """
    if not os.path.isfile(filepath):
        warnings.warn("The file %s does not exist" % filepath)
        return False

    with open(filepath, 'rb') as filehandle:
        byte1 = filehandle.read(1)
        byte2 = filehandle.read(1)
    if byte1 == b'\x1f' and byte2 == b'\x8b':
        return True
    else:
        return False


def infile_handler(filepath: str) -> TextIO:
    """
    Detect if the file specified by `filepath` is gzip-compressed
    and open the the file in read mode using appriate open handler.
    """
    if is_gzip(filepath):
        return gzip.open(filepath, "rt")
    else:
        return open(filepath, "rt")


def outfile_handler(filepath: str,
                    compression: Optional[str] = None) -> TextIO:
    """
    Return a file handle in write mode using the appropriate
    handle depending on the compression mode.
    Valid compression mode:
        compress = None | "None" | "gzip" | "gz" | "bgzip" | "bgz"
    If compress = None or other input, open the file normally.
    """
    if os.path.isfile(filepath):
        warnings.warn("Overwriting the existing file: %s" % filepath)

    if compression is None:
        return open(filepath, mode="wt")
    elif type(compression) == str:
        if compression.lower() in ["gzip", "gz"]:
            return gzip.open(filepath, mode="wt")
        elif compression.lower() in ["bgzip", "bgz"]:
            return BgzfWriter(filepath)
        elif compression.lower() == "none":
            return open(filepath, mode="wt")
    else:
        raise Exception("`compression = %s` invalid." % str(compression))


############
# Other
############


def variant_tally(variant_record_list, columns=None):
    result = {}
    num_record = len(variant_record_list)
    for i, record in enumerate(variant_record_list):
        for variant in record:
            if variant not in result:
                result[variant] = np.zeros(num_record)
            result[variant][i] += 1
    return pd.DataFrame.from_dict(data=result,
                                  orient="index",
                                  columns=columns,
                                  dtype=np.int8)


def binomial_KL_divergence(n, p1, p2):
    """
    Compute the KL-divergence of two binomial distribution
    with the sample number of trials, n but with
    different success probability p1 and p2, i.e.
        KL-D(Binom(n, p1) || Binom(n, p2))
    """
    return n * (p1 * np.log(p1/p2) +
                (1- p1) * np.log((1 - p1) / (1-p2)))


