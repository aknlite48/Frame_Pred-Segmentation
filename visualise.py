from torchvision.utils import make_grid
import numpy as np
import matplotlib.pyplot as plt
import torch
from ipywidgets import interact
import imageio
import torchvision
from IPython.display import Image
from ipywidgets import interact
import imageio
import io

def visualize_frame_and_mask_with_values_from_dataset(frame,mask,str_label):
    frame_np = frame.permute(1, 2, 0).numpy()
    mask_np = mask.numpy()

    # Get unique mask values
    unique_values = np.unique(mask_np)

    # Prepare the subplot
    fig, axes = plt.subplots(1, len(unique_values) + 2, figsize=(15, 5))

    fig.suptitle(str_label)

    # Display the original frame
    axes[0].imshow(frame_np)
    axes[0].set_title('Frame')
    axes[0].axis('off')

    # Display the full mask
    axes[1].imshow(mask_np, cmap='gray')
    axes[1].set_title('Full Mask')
    axes[1].axis('off')

    # Display mask effects for each unique value
    for i, val in enumerate(unique_values):
        # Create a mask for the current unique value only
        single_value_mask = np.zeros_like(mask_np)
        single_value_mask[mask_np == val] = val

        # Overlay this mask on the frame
        overlay = frame_np.copy()
        overlay[single_value_mask != val] = 0  # Black out non-target areas

        axes[i + 2].imshow(overlay)
        axes[i + 2].set_title(f'Value {val}')
        axes[i + 2].axis('off')

    plt.tight_layout()
    plt.show()

def visualise_mask(i,data,model):

    f,m=data.get_image(i)


    visualize_frame_and_mask_with_values_from_dataset(f,m,'Original')

    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        res = model(data[i][0].to(device).unsqueeze(0))
    out = mask_output(res.squeeze())
    visualize_frame_and_mask_with_values_from_dataset(f,out,'Predicted')
    print('IOU : %f'%(iou(m.unsqueeze(0),out.unsqueeze(0),49)))

def mask_output(f_mask):
    return torch.argmax(f_mask,dim=0).detach().cpu()
    
def iou(pred, target, n_classes):
    # pred and target shapes are (batch_size, 1, H, W)
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)

    for cls in range(n_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds[target_inds]).long().sum().item()
        union = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection
        if union == 0:
            ious.append(float('nan'))  
        else:
            ious.append(float(intersection) / union)


    ious = [x for x in ious if not np.isnan(x)]
    if len(ious) == 0:
        return float('nan') 
    return np.mean(ious)

def video_gen(i,inp_data,fname,model,vp):
    def display_sequence(images):
        def _show(frame=(1, len(images))):
            return images[frame-1]
        return interact(_show)
    img_conv = torchvision.transforms.ToPILImage(mode=None)
    videos = []
    data,_ = inp_data[i]
    for i in range(data.shape[0]):
        videos.append(img_conv((data[i,:,:,:])))
    display_sequence(videos)
    videos = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vp.eval()
    with torch.no_grad():
        d = vp((data.to(device).unsqueeze(0))).squeeze(0)
    for i in range(d.shape[0]):
        videos.append(img_conv((d[i,:,:,:])))
    display_sequence(videos)

    videos = []
    model.eval()
    with torch.no_grad():
        d = model.pred_vid((data.to(device).unsqueeze(0))).squeeze(0)
    for i in range(d.shape[0]):
        videos.append(img_conv((d[i,:,:,:])))
    display_sequence(videos)

def gif_gen(i,inp_data,fname,model,vp):
    def display_sequence(videos,mod):        
        animated_gif = io.BytesIO()
        videos[0].save(fname+'_'+mod+'.gif',
                       format='GIF',
                       save_all=True,
                       append_images=videos[1:],      # Pillow >= 3.4.0
                       delay=0.1,
                       loop=0)
        #animated_gif.seek(0,2)
        #print ('GIF image size = ', animated_gif.tell())

        #animated_gif.seek(0)
        #ani = Image.open(animated_gif)
        #ani.show()

    img_conv = torchvision.transforms.ToPILImage(mode=None)
    videos = []
    data = inp_data.get_vid(i)
    for i in range(data.shape[0]):
        videos.append(img_conv((data[i,:,:,:])))
    display_sequence(videos,'orig')
    videos = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vp.eval()
    with torch.no_grad():
        d = vp((data.to(device).unsqueeze(0))).squeeze(0)
    for i in range(d.shape[0]):
        videos.append(img_conv((d[i,:,:,:])))
    display_sequence(videos,'simvp')

    videos = []
    model.eval()
    with torch.no_grad():
        d = model.pred_vid((data.to(device).unsqueeze(0))).squeeze(0)
    for i in range(d.shape[0]):
        videos.append(img_conv((d[i,:,:,:])))
    display_sequence(videos,'mod')