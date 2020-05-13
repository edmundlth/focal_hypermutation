
import pandas as pd
import numpy as np
from .utility import infile_handler


#################################
# Class defintion : Dataset
#################################
class Dataset(object):
    def __init__(self, sample_ids):
        self.sample_ids = list(sample_ids)
        self.data = {sample_id: Sample(sample_id) for sample_id in sample_ids}
        self.dbsnp_vcf = None

    def add_vcf(self, sample_id, vcfname, vcfpath):
        self.data[sample_id].add_vcf(vcfname, vcfpath)
        return

    def get_pass_variants(self):
        for sample_id in self.sample_ids:
            self.data[sample_id].get_pass_variants()
        return

    def __str__(self):
        string = "Sample IDs : %s " % ', '.join(self.sample_ids)
        return string

    def __setitem__(self, sample_id, data):
        self.data[sample_id] = data
        return

    def __getitem__(self, sample_id):
        return self.data[sample_id]


#################################
# Class defintion : Sample
#################################
class Sample(object):
    def __init__(self, sample_id, metadata=None):
        self.sample_id = sample_id
        self.metadata = metadata
        self.vcfs = {}
        self.pass_vcfs = {}
        self.df_consensus = None
        self.df_sanger_VAF = None
        self.df_consensus_nondbSNP = None
        self.df_consensus_nondbSNP_VAF = None

    def process_sample(self):
        self.get_pass_variants()
        self.compute_consensus_variants()
        self.get_sanger_VAF()
        return

    def add_vcf(self, vcfname, vcfpath):
        self.vcfs[vcfname] = VCF(vcfpath, pandas_engine='c')
        return self.vcfs[vcfname]

    def get_pass_variants(self):
        def filter_has_pass(df):
            select = ["PASS" in filter_tag for filter_tag in df["FILTER"]]
            return df.loc[select]
        for vcfname in self.vcfs:
            self.pass_vcfs[vcfname] = filter_has_pass(self.vcfs[vcfname].df)
        return self.pass_vcfs

    def get_vcf(self, vcfname):
        return self.vcfs[vcfname]

    def get_sample_type(self):
        return self.metadata["Sample Type"]

    def compute_consensus_variants(self):
        pass_vcfnames = sorted(self.pass_vcfs.keys())
        fst_name = pass_vcfnames[0]
        df_result = self.pass_vcfs[fst_name].loc[:, ["ALT"]]
        for name in pass_vcfnames[1:]:
            df2 = self.pass_vcfs[name].loc[:, ["ALT"]]
            df_result = pd.merge(df_result, df2,
                                 how="inner",
                                 on=["CHROM", "POS", "REF", "ALT"])
        self.df_consensus = df_result
        return df_result

    def get_sanger_VAF(self, df_sanger=None, sanger_samplename="TUMOUR"):
        if not df_sanger:
            df_sanger = self.pass_vcfs["SANGER"]
        result = []
        for row in df_sanger.itertuples():
            # index = row[0]
            alt = row[2]
            if len(alt) != 1:
                print(row)
            info_dict = {}
            for rec in row[5].split(';'):
                if '=' in rec:
                    key, val = rec.split('=')
                    if str.isnumeric(val):
                        val = float(val)
                else:
                    key = rec
                    val = True
                info_dict[key] = val
            # row = df_sanger.loc[index, :]
            dp = info_dict["DP"]
            fmt = dict(
                # TODO: The TUMOUR sample column is column 8.. Need to abstract this.
                zip(row[6].split(':'), row[8].split(':'))
            )
            # fmt = VCF.get_row_sample_genotype(row, sanger_samplename)
            fad = int(fmt['F' + alt + 'Z'])
            rad = int(fmt['R' + alt + 'Z'])
            ad = fad + rad
            result.append([dp, ad / dp, fad / dp, rad / dp])
        df_result = pd.DataFrame(result, index=df_sanger.index,
                                 columns=["DP", "VAF", "FVAF", "RVAF"])
        self.df_sanger_VAF = df_result
        return df_result

    def __str__(self):
        string = "Sample ID : %s \n" % self.sample_id
        string += "VCFs     : %s \n" % str(self.vcfs.keys())
        string += "Metadata : \n%s \n" % str(self.metadata)
        return string

    def dereference_original_vcfs(self):
        self.vcfs = {}
        return

    def dereference_pass_vcfs(self):
        self.pass_vcfs = {}
        return

    def consensus_nondbSNP(self, dbsnp_snv_variant_dict):
        def in_dbSNP(row):
            return f"{row.name[0]}_{row.name[1]}_{row.name[2]}>{row['ALT']}" in dbsnp_snv_variant_dict
        self.df_consensus["in_dbSNP"] = self.df_consensus.apply(in_dbSNP, axis=1)
        self.consensus_nondbSNP = self.df_consensus.groupby("in_dbSNP").get_group(False).drop("in_dbSNP", axis=1)
        return self.consensus_nondbSNP

    def consensus_nondbSNP_VAF(self, dbsnp_snv_variant_dict):
        df = self.consensus_nondbSNP(dbsnp_snv_variant_dict)
        self.df_consensus_nondbSNP_VAF = (
            pd.merge(df,
                     self.df_sanger_VAF,
                     how="inner",
                     on=["CHROM", "POS", "REF"]))
        return self.df_consensus_nondbSNP_VAF


#################################
# Class defintion : VCF
#################################
class VCF(object):
    """
    This looks a lot like reimplementing PyVCF.Reader
    just with Pandas.DataFrame object to hold the main body
    of the vcf file.
    """
    def __init__(self, vcf_filepath,
                 pandas_engine="c"):
        self.vcf_filepath = vcf_filepath
        self.pandas_engine = pandas_engine
        # parse the body of vcf into a Pandas.DataFrame object
        self.df, self.header, self.SAMPLES = VCF.parse_vcf(vcf_filepath, pandas_engine=pandas_engine)

    @staticmethod
    def parse_vcf(vcf_filepath, pandas_engine="python"):
        with infile_handler(vcf_filepath) as vcf_file:
            header = []
            for line in vcf_file:
                line = line.rstrip()
                if line.startswith("##"):
                    # if it is a header line
                    header.append(line)
                elif line.startswith("#CHROM"):
                    # if it is the column names line
                    columns = line[1:].split('\t')
                    # immediately break the loop
                    # so that the filehandle starts with the main body
                    # of the vcf file
                    break
            # which we then read into a Pandas.DataFrame object
            df = pd.read_csv(vcf_file, sep='\t',
                             header=None,
                             engine=pandas_engine,
                             dtype={0: 'category',
                                    1: np.uint32})
        # Give correct header
        df.columns = columns
        # slice out samples
        samples = columns[9:]
        # index dataframe using CHROM and POS (i.e. genomic loci)
        df.set_index(["CHROM", "POS", "REF"], inplace=True)
        # sort dataframe using genomic loci.
        df.sort_index(inplace=True)
        return df, header, samples

    def get_format_tags_intersection(self):
        common = set(self.df["FORMAT"][0].split(':'))
        for tags in self.df["FORMAT"]:
            current = set(tags.split(':'))
            common = common.intersection(current)
        return common

    def get_info_tags_intersection(self):
        def get_tags(info):
            return set(field.split('=', maxsplit=1)[0] for field in info.split(';'))
        common = set(get_tags(self.df["INFO"][0]))
        for info in self.df["INFO"]:
            current = get_tags(info)
            common = common.intersection(current)
        return common

    def melt_genotype_column(self, samples=None):
        if samples is None:
            samples = self.SAMPLES
        elif type(samples) == int:
            samples = [self.SAMPLES[samples]]

        common_format_tags = self.get_format_tags_intersection()
        for sample in samples:
            for tag in common_format_tags:
                data = []
                col_name = sample + '_' + "FORMAT" + '_' + tag
                self.df[col_name] = np.zeros(self.df.shape[0])
                for i, row in self.df.iterrows():
                    row_sample_genotype = VCF.get_row_sample_genotype(row,
                                                                      sample)
                    data.append(row_sample_genotype[tag])
                self.df[col_name] = data
        return self.df

    def melt_info_column(self):
        common_info_tags = self.get_info_tags_intersection()
        for tag in common_info_tags:
            data = []
            for i, row in self.df.iterrows():
                data.append(VCF.get_row_info(row)[tag])
            self.df["INFO" + '_' + tag] = data
        return self.df

    @staticmethod
    def get_all_allelic_change(df):
        data = []
        for i, row in df.iterrows():
            for alt in row["ALT"].split(','):
                change = "{ref}>{alt}".format(ref=row["REF"], alt=alt)
                record = [row["CHROM"], row["POS"], change]
                data.append(record)
        return pd.DataFrame(data, columns=["CHROM", "POS", "CHANGE"])

    ########
    # VCF dataframe utilities
    ########
    @staticmethod
    def get_row_sample_genotype(row, sample_name):
        tags = row["FORMAT"].split(':')
        tag_values = row[sample_name].split(':')
        result = dict(zip(tags, tag_values))
        for key, val in result.items():
            try:
                result[key] = np.float(val)
            except ValueError:
                pass
        return result

    @staticmethod
    def get_row_info(row):
        result = {}
        for field in row["INFO"].split(';'):
            if '=' in field:
                key, val = field.split('=')
                result[key] = val
            else:
                result[field] = True
        for key, val in result.items():
            if type(val) is bool:
                continue
            try:
                result[key] = np.float(val)
            except ValueError:
                pass
        return result