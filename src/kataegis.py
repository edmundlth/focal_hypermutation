import matplotlib.pyplot as plt
from bokeh.plotting import figure
from bokeh.transform import factor_cmap
from bokeh.palettes import Paired12
from bokeh.models import BoxAnnotation, Range1d

import numpy as np
import intervaltree
import scipy.cluster.hierarchy as shc
from scipy.stats import binom_test

CHROMS = [f"{i}" for i in range(1, 23)] + ["X", "Y"]
# HG19
chrom_lengths = {'1': 249250621, '2': 243199373, '3': 198022430, '4': 191154276,
                 '5': 180915260, '6': 171115067, '7': 159138663, 'X': 155270560,
                 '8': 146364022, '9': 141213431, '10': 135534747, '11': 135006516,
                 '12': 133851895, '13': 115169878, '14': 107349540, '15': 102531392,
                 '16': 90354753, '17': 81195210, '18': 78077248, '20': 63025520,
                 'Y': 59373566, '19': 59128983, '22': 51304566, '21': 48129895}
############################################################################################################
# Rainfall plots
############################################################################################################


def rainfall_plot(dataset, sample_ids=None, chroms=None):
    if sample_ids is None:
        sample_ids = dataset.sample_ids
    if chroms is None:
        chroms = CHROMS
    nrows = len(sample_ids)
    ncols = 24
    fig, ax = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3), sharex='col', sharey='row')
    ax = np.ravel(ax)

    c = 0
    cmap = {('A', 'T'): 'k', ('A', 'G'): 'g', ('A', 'C'): 'b', ('T', 'G'): 'y', ('T', 'C'): 'r', ('G', 'C'): 'c',
            ('T', 'A'): 'k', ('G', 'A'): 'g', ('C', 'A'): 'b', ('G', 'T'): 'y', ('C', 'T'): 'r', ('C', 'G'): 'c'}

    for sample in sorted(sample_ids):
        donor_id = dataset[sample].metadata["Subject/Donor ID"]
        for chrom in chroms:
            df = dataset[sample].df_consensus.groupby("in_dbSNP").get_group(False)
            color = [cmap[(a, b)] for x, a, b in df.reset_index().loc[:, ["REF", "ALT_DKFZ"]].itertuples()]
            if chrom in df.index:
                positions = df.loc[chrom].reset_index().loc[:, "POS"]
                x = sorted(positions)
                y = [x[i] - x[i - 1] for i in range(1, len(x))]
                ax[c].scatter(x[1:], y, s=3, color=color)
                ax[c].set_title(f"{sample} ({donor_id}) chr{chrom}, n={len(x)}", fontsize=8)
                ax[c].set_yscale('log')
            c += 1
    return fig, ax


def bokeh_rainfall_plot(dataset,
                        sample_ids=None,
                        sample_alias=None,
                        chroms=None,
                        df_centromere=None,
                        kataegis_func=None,
                        gene_interval_tree_by_chrom=None,
                        point_size=3,
                        title_fontsize=10,
                        plot_width=250,
                        plot_height=250):
    if sample_ids is None:
        sample_ids = dataset.sample_ids
    if sample_alias is None:
        sample_alias = sample_ids
    assert len(sample_alias) == len(sample_ids)
    if chroms is None:
        chroms = CHROMS
    factors = ['AG', 'TC', 'AC', 'TG', 'AT', 'TA', 'CA', 'GT', 'CT', 'GA', 'CG', 'GC']
    tooltips = [
        ("x", "$x"),
        ("y", "$y"),
        ("pos", "@POS"),
        ("dist", "@dist"),
        ("sub", "@sub")
    ]
    # tools = [HoverTool(), PanTool(), ResetTool(), WheelZoomTool(), BoxZoomTool(), UndoTool() SaveTool()]
    grid = []
    x_ref_range = {chrom: None for chrom in chroms}
    for sample, alias in zip(sample_ids, sample_alias):
        fig_chrom = []
        y_ref_range = None
        for chrom in chroms:
            df = dataset[sample].df_consensus.groupby("in_dbSNP").get_group(False).drop("in_dbSNP", axis=1)
            if chrom in df.index:
                df_data = df.loc[chrom].reset_index().loc[:, ["POS", "REF", "ALT"]].sort_values(by="POS")
                df_data["sub"] = df_data["REF"] + df_data["ALT"]
                df_data["dist"] = np.diff(df_data["POS"], prepend=0)
                title = f"{alias} chr{chrom} n={df_data.shape[0]}"
                if y_ref_range is None:
                    y_ref_range = Range1d(1, np.max(df_data["dist"]))
                if x_ref_range[chrom] is None:
                    x_ref_range[chrom] = Range1d(0, chrom_lengths[chrom])

                fig = figure(plot_width=plot_width,
                             plot_height=plot_height,
                             title=title,
                             tooltips=tooltips,
                             y_axis_type="log",
                             y_range=y_ref_range,
                             x_range=x_ref_range[chrom])
                fig.scatter("POS", "dist",
                            source=df_data,
                            size=point_size,
                            color=factor_cmap('sub', Paired12, factors))
                fig.title.text_font_size = f"{title_fontsize}pt"
                kataegis_windows = kataegis_func(df_data["POS"])
                if kataegis_windows:
                    fig.title.text_color = "red"
                    fig.title.text += f" k={len(kataegis_windows)}"
                for start, end in kataegis_windows:
                    fig.add_layout(BoxAnnotation(left=start, right=end, fill_color="red", fill_alpha=0.2))
                    if gene_interval_tree_by_chrom is not None:
                        for gene_start, gene_end, gene_name in gene_interval_tree_by_chrom[chrom].overlap(start, end):
                            distances = df_data["dist"]
                            width = gene_end - gene_start
                            height = np.max(distances)
                            x = gene_start + width / 2
                            y = height / 2
                            fig.rect(x=x, y=y, height=height, width=width,
                                     name=gene_name, fill_alpha=0.2, fill_color='blue')
            else:
                fig = figure(plot_width=plot_width,
                             plot_height=plot_height,
                             title=f"{sample} chr{chrom} n=0",
                             y_axis_type="log")
                fig.circle([0], [0])
            if df_centromere is not None:
                if "chr" not in chrom:
                    chrom = f"chr{chrom}"
                start, end = df_centromere.loc[chrom, ["chromStart", "chromEnd"]]
                box = BoxAnnotation(left=start, right=end, fill_color='grey', fill_alpha=0.2)
                fig.add_layout(box)
            fig_chrom.append(fig)
        grid.append(fig_chrom)
    return grid

############################################################################################################
# Kataegis detection
############################################################################################################


def kataegis_detection_by_num_var_in_window(positions,
                                            min_num_var=6,
                                            window_size=1000,
                                            is_sorted=False):
    n = len(positions)
    if n < 2:
        return []
    x = np.array(positions)
    if not is_sorted:
        x.sort()

    result = []
    i = 0
    while i < n - 1:
        current_pos = x[i]
        num = 1
        dist = x[i + num] - current_pos
        while dist < window_size and i + num < n:
            num += 1
            dist = x[i + num] - current_pos
        if num > min_num_var:
            result.append(x[i:i + num])
        i += num
    return result


def kataegis_detection_by_average_variant_distance2(positions,
                                                    min_num_var=6,
                                                    max_avg_dist=1000,
                                                    is_sorted=False):
    total_num_var = len(positions)
    if total_num_var < 2:
        return []
    x = np.array(positions)
    y = np.diff(x)  # len(y) == len(x) - 1
    if not is_sorted:
        x.sort()
    i = 0
    result = []
    while i < total_num_var - min_num_var:
        num_var = min_num_var
        start = x[i]
        while np.mean(y[i: i + num_var - 1]) <= max_avg_dist and i + num_var < total_num_var:
            num_var += 1
        num_var -= 1
        if np.mean(y[i: i + num_var - 1]) <= max_avg_dist and num_var >= min_num_var:
            end = x[i + num_var - 1]
            result.append((start, end))
            i += num_var
        else:
            i += 1
    return result


def kataegis_detection_by_average_variant_distance(positions,
                                                   min_num_var=6,
                                                   max_avg_dist=1000,
                                                   is_sorted=False):
    total_num_var = len(positions)
    if total_num_var < 2:
        return []
    x = np.array(positions)
    y = np.diff(x)  # len(y) == len(x) - 1
    if not is_sorted:
        x.sort()
    result = []
    largest_end = 0
    for i in range(total_num_var - min_num_var):
        start = x[i]
        end = None
        for j in range(i + min_num_var - 1, total_num_var):
            if x[j] > largest_end and np.mean(y[i: j]) <= max_avg_dist:
                end = x[j]
                largest_end = end
        if end is not None:
            result.append((start, end))
    return result


def kataegis_detection_by_hierarchical_clustering(positions,
                                                  min_num_var=6,
                                                  max_avg_dist=1000,
                                                  clustering_method='centroid',
                                                  is_sorted=False):
    x = np.array(positions)
    if len(x) < 2:
        return []
    if not is_sorted:
        x.sort()
    result = []
    links = shc.linkage(x.reshape(-1, 1), method=clustering_method)
    clusters = get_clusters_from_linkage_matrix(links, x)
    windows = []
    for i, points in clusters.items():
        if len(points) > min_num_var:
            points = np.array(points)
            points.sort()
            if np.mean(np.diff(points)) < max_avg_dist:
                windows.append((points[0], points[-1]))
    windows = sorted(windows)
    if windows:
        start, end = windows[0]
        for current_start, current_end in windows[1:]:
            if current_start < start:
                start = current_start
            elif current_start < end:
                end = current_end
            else:
                result.append((start, end))
                start = current_start
                end = current_end
    return result


def kataegis_detection_by_binomial_p(positions, chrom_length, p=10e-10, is_sorted=False):
    x = np.array(positions)
    n = len(x)
    if n < 2:
        return []
    if not is_sorted:
        x.sort()
    windows = []
    # bonferroni correction
    num_test = (n - 1) * n // 2
    p = p / num_test
    p_hypothesis = n / chrom_length
    i = 0
    while i < n - 1:
        start = x[i]
        end = None
        num_var = 2
        while i + num_var - 1 < n:
            current_end = x[i + num_var - 1]
            length = int(current_end - start + 1)
            p_test = binom_test(x=num_var, n=length, p=p_hypothesis, alternative="greater")
            if p_test < p:
                end = current_end
                print((length, num_var, length / num_var, p_test))
            num_var += 1
        if end is not None:
            windows.append((start, end))
        i += 1
    return windows


def get_clusters_from_linkage_matrix(links, observations):
    n = len(observations)
    cluster_rec = {i: [observations[i]] for i in range(n)}

    def recur(linkage_matrix, iteration_num):
        i, j = linkage_matrix[iteration_num, :2]
        if i in cluster_rec:
            left_cluster = cluster_rec[i]
        else:
            left_cluster = recur(linkage_matrix, i - n)

        if j in cluster_rec:
            right_cluster = cluster_rec[j]
        else:
            right_cluster = recur(linkage_matrix, j - n)
        cluster_rec[iteration_num + n] = left_cluster + right_cluster
        return cluster_rec

    for k in range(links.shape[0]):
        recur(links, k)
    return cluster_rec


#####################################


def gaussian_smoothing():
    pass


def variant_interval_tree(positions):
    x = np.array(positions)
    return intervaltree.IntervalTree.from_tuples(zip(x, x + 1))
