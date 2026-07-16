import numpy as np
import torch   
from pathlib import Path
#from exp_utils import get_files
DIR = "DESBO_run"
indir = Path(DIR + "/Input")
outdir = indir
outdir.mkdir(exist_ok=True)
TKWARGS = {
    "dtype": torch.double,# Datatype used by tensors
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"), # Declares the 'device' location where the Tenosrs will be stored
}
def get_files(indir,file_names,suffix=None):
    """
    Extracts files from specified directory
    """
    data = {}
	
    for file_name in file_names:
        if suffix is not None:
            get_name = file_name + suffix
        else:
            get_name = file_name
        load_in = torch.load(indir /  f"{get_name}.pt").to(**TKWARGS)
        data[file_name] = load_in

    return data
# Save test data first 
test_data = get_files(indir=indir,file_names=['test_x','test_y','test_sigma2'])
test_x = test_data['test_x'].numpy()
test_y = test_data['test_y'].numpy()
test_sigma2 = test_data['test_sigma2'].numpy()
np.save(outdir / "test_x.npy", test_x)
np.save(outdir / "test_y.npy", test_y)
np.save(outdir / "test_sigma2.npy", test_sigma2)

#Convert all torch data to numpy and save as .npy files 
for m in range(0,50):

    data = get_files(indir=indir,file_names=['train_x','train_y','train_n'],suffix=f"_m{m}")

    train_x = data['train_x'].numpy()
    train_y = data['train_y'].numpy()
    train_n = data['train_n'].numpy()

# outdir=Path("/home/newtonh3/hetGPy/bo_data") 
    np.save(outdir / f"train_x_m{m}.npy", train_x)
    np.save(outdir / f"train_y_m{m}.npy", train_y)
    np.save(outdir / f"train_n_m{m}.npy", train_n)

