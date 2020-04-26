import os
import numpy as np
import zipfile as zf
import re
import pandas as pd


def unzip_file(file):
	"""unzip data of type input_data.zip"""
	e_f=re.match(r'((?<=\.\w{3} )|^)[\w ]+\.input_data.zip',file)
	if e_f:
		with zf.ZipFile(file, 'r') as zipobj:
			zipobj.extractall("/home/ubuntu/data/pres")
			#print("The %s is extracted" %e_f)


def file_to_snp_freq(file):
	snp_list=["AA","AT","AC","AG","TA","TT","TC","TG","GA","GT","GC","GG", "CA", "CT", "CG", "CC"]
	df=pd.read_csv(file, sep="\t",header=None,names=["id","chr","position","SNPs"],encoding="ISO-8859-1")
	df["chr"]=df["chr"].astype('str')
	data=df.groupby(by=['chr', 'SNPs']).count().reset_index()
	#data=data.drop("id","position")
	#dummy_df = pd.DataFrame({"chr":np.tile(list(range(1,23)), "SNPs": np.tile["AA","AT","AC","AG","TA","TT","TC","TG","GA","GT","GC","GG"],"count":[9999]*22})
	#df1=data.set_index(['chr','SNPs']).count(level="SNPs")
	data=data.iloc[:,:3]
	data.columns=["chr","SNPs","count"]
	dummy_df=pd.DataFrame({"chr":["dummy"]*16, "SNPs": snp_list,"count":[9999]*16})
	data=pd.concat([data,dummy_df])
	data=data.pivot("chr","SNPs","count").reset_index()
	data=data[~(data["chr"] == "dummy")]
	data=data.loc[data["chr"].isin([str(j) for j in range (1,23)]),:]
#	data = data[snp_list]
	data.fillna(0).astype("int32")
	data = data[snp_list]
	data=np.array(data)
	return(data)
	#np.savetxt("%s_frq.txt"%file,data,delimiter="\t")


entries=os.listdir("/home/ubuntu/data")
os.chdir("/home/ubuntu/data")
#print("Initial path is",os.getcwd())
for folder in entries:
#	print("Folder is", folder)
	path1=os.path.join("/home/ubuntu/data",folder)
	os.chdir("/home/ubuntu/data")
#		print("path to folders is", path1)
	if os.path.isdir(path1):
		#print("The path for the directory in use is", path1)
		subfol_files = os.listdir(folder)
		os.chdir(path1)
		for file in subfol_files:
			unzip_file(file)
print("Files have been unziped")
def get_snp_freq(path):
	
	os.chdir(path)
	chr_data_list = list()
	for file in os.listdir(path):
		data=file_to_snp_freq(file)
		chr_data_list.append(data)
	final_array=np.array(chr_data_list)
	np.save("dna_encoding",final_array)
	print("Document with all the dna data encoded has been saved")
get_snp_freq("/home/ubuntu/data/pres")
