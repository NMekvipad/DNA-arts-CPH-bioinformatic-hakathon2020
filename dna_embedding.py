import os
import numpy as np
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("snp_file_dir", help="raw_text.txt SNPs file with the information of SNPs on all chromosome", default=None)
parser.add_argument("--out_file", help="raw_text.txt SNPs file with the information of SNPs on all chromosome", default="snp_embedding")

snp_list=["AA","AT","AC","AG","TA","TT","TC","TG","GA","GT","GC","GG", "CA", "CT", "CG", "CC"]

def create_embedding(snp_dir):

    chr_data_list = list()
    for file in os.listdir(snp_dir):
        df = pd.read_csv(os.path.join(snp_dir, file), sep="\t", header=None, names=["id", "chr", "position", "SNPs"], encoding="ISO-8859-1")
        df["chr"] = df["chr"].astype('str')
        data = df.groupby(by=['chr', 'SNPs']).count().reset_index()
        data = data.iloc[:, :3]
        data.columns = ["chr", "SNPs", "count"]
        dummy_df = pd.DataFrame({"chr": ["dummy"] * 16, "SNPs": snp_list, "count": [9999] * 16})
        data = pd.concat([data, dummy_df])
        data = data.pivot("chr", "SNPs", "count").reset_index()
        data = data[~(data["chr"] == "dummy")]
        data = data.loc[data["chr"].isin([str(j) for j in range(1, 23)]), :]
        data = data.fillna(0).astype('int32')
        data = data.sort_values('chr')
        data = data[snp_list]
        data = np.array(data)
        chr_data_list.append(data)

    freq = np.array(chr_data_list)
    return freq

if __name__ == '__main__':
    args = parser.parse_args()
    snp_dir = args.snp_file_dir
    out_filename = args.out_file
    freq = create_embedding(snp_dir)
    np.save(out_filename, freq)