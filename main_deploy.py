import numpy as np
import torch
import torch.nn.functional as F

from model import Transfer_Cnn14


classes_num=294
model_path="best_models/best_model.pth"
sample_rate=32000
window_size=1024
hop_size=320
mel_bins=64
fmin=50
fmax=14000
batch_size=4

def pred():
    ## making dummy data
    dummy_data=[]
    for i in range(batch_size):
        x=np.linspace(0, 100*np.pi, 10000)
        dummy_data.append(np.sin(x+i/4*np.pi))
    dummy_input=torch.tensor(dummy_data,dtype=torch.float32)
    print("dummy input(batch_size x wave_length):",dummy_input.shape)
    
    ## loading model
    model = Transfer_Cnn14(sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num, freeze_base=False)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    with torch.no_grad():   
        pred_y=model(dummy_input)
        print("== output(log probability) ==")
        print(pred_y)
        print("== output shape (batch_size x #label) ==")
        print(pred_y.shape)


if __name__ == "__main__":
    pred()

