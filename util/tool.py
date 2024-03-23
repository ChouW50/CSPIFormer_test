import cv2
import numpy as np
import torch.nn as nn
import torch, os, time
import sklearn.metrics as skm
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from .dataset import ImageLbDataset, TxtLabelDataset, NoLabelDataset, ImageLabelDataset
from sklearn.metrics import confusion_matrix, classification_report
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from IPython.display import clear_output
from skimage.measure import block_reduce
from pytorch_grad_cam import GradCAM
from datetime import datetime
from PIL import Image, ImageDraw
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def CreateNF(path):
    if not os.path.isdir(path):
        os.mkdir(path)

def CreateNewFile(last_path):
    CreateNF(last_path)
    now = datetime.now()
    timestr = now.strftime("%Y_%m_%d_%H_%M_%S")
    # print('年_月_日_时_分_秒：', timestr)
    pth = last_path + timestr + '\\'
    CreateNF(pth)
    return pth

def check_dataset(path, size, split, transform, ImageLabel, **kwargs):
    if ImageLabel == 0:
        return TxtLabelDataset(path, size, split, transform,**kwargs)
    elif ImageLabel == 1:
        return ImageLbDataset(path, split, transform, size[0], size[1], **kwargs)
    elif ImageLabel == 3: 
        return ImageLabelDataset(path, split, transform, size[0], size[1], **kwargs)
    else:
        return NoLabelDataset(path, size, split, transform, **kwargs)
        

def check_checkpoint(model, optimizer, scheduler = None, weight_path = '', str_ = '', Checkpoint_path = '', data_name = ''):
    if os.path.exists(f"{Checkpoint_path}model_checkpoint_{data_name}{str_}.tar"):
        checkpoint = torch.load(f"{Checkpoint_path}model_checkpoint_{data_name}{str_}.tar")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint and scheduler != None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        epoch_continue = checkpoint["epoch"]
        best_F1 = checkpoint["best_F1score"]
        weight_path = checkpoint["weight_path"]
        if "train_time" in checkpoint:
            train_time = checkpoint["train_time"]
        else: train_time = [0]
        if "val_time" in checkpoint:
            val_time = checkpoint["val_time"]
        else: val_time = [0]
        print(f"take last checkpoint:")
        return model, optimizer, scheduler, epoch_continue, best_F1, weight_path, train_time, val_time
    else: return model, optimizer, scheduler, 0, 0, weight_path, [0], [0]

def cls_check_checkpoint(model, optimizer, scheduler = None, weight_path = '', str_ = '', Checkpoint_path = '', data_name = ''):
    if os.path.exists(f"{Checkpoint_path}model_checkpoint_{data_name}{str_}.tar"):
        checkpoint = torch.load(f"{Checkpoint_path}model_checkpoint_{data_name}{str_}.tar")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint and scheduler != None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        epoch_continue = checkpoint["epoch"]
        best_loss = checkpoint["best_loss_score"]
        best_acc = checkpoint["best_acc_score"]
        weight_path = checkpoint["weight_path"]
        loss_list = checkpoint["matrics_loss"]
        acc_list = checkpoint["matrics_acc"]
        if "train_time" in checkpoint:
            train_time = checkpoint["train_time"]
        else: train_time = [0]
        if "val_time" in checkpoint:
            val_time = checkpoint["val_time"]
        else: val_time = [0]
        print(f"take last checkpoint:")
        return model, optimizer, scheduler, epoch_continue, best_loss, best_acc, weight_path, loss_list, acc_list, train_time, val_time
    else: return model, optimizer, scheduler, 0, 100, 0, weight_path, [], [], [0], [0]

def color_set(img_list, num_class):
    if num_class == 4:
        for i in range(len(img_list)):
            for j in range(len(img_list[0])):
                if img_list[i][j] == 0:
                    # background
                    img_list[i][j] = 0
                elif img_list[i][j] == 1:
                    # barretts
                    img_list[i][j] = 127
                elif img_list[i][j] == 2:
                    # normal
                    img_list[i][j] = 255
                elif img_list[i][j] == 3:
                    # Gastric tissue
                    img_list[i][j] = 60
        return img_list
    else: return img_list

def load_image(img_path, img_num, img_size):
    dir = os.path.dirname(img_path)
    img_name = os.listdir(dir)
    return Image.open(f"{img_path}{img_name[img_num]}").convert('RGB').resize((img_size[0], img_size[1])), img_name[img_num]

def show_img(epoch, img, label, pred, type1, num_class, path):
    # 轉換 PyTorch 張量為 NumPy 數組
    input_img = img.cpu().numpy().transpose((1, 2, 0))
    target_img = color_set(label.cpu().numpy(), num_class)
    pred_img = color_set(pred.cpu().numpy(), num_class)

    # 顯示圖像和分割圖
    fig, ax = plt.subplots(1, 3, figsize = (10, 4))
    fig.suptitle(f'Epoch {epoch} {type1}')
    fig.tight_layout(h_pad = 2)
    ax[0].set_title('input image')
    ax[0].axis('off')
    ax[0].xaxis.set_visible(False)
    ax[0].yaxis.set_visible(False)
    ax[0].imshow(input_img)
    ax[1].set_title('label image')
    ax[1].xaxis.set_visible(False)
    ax[1].yaxis.set_visible(False)
    ax[1].imshow(target_img)
    ax[2].set_title('prediction image')
    ax[2].xaxis.set_visible(False)
    ax[2].yaxis.set_visible(False)
    ax[2].imshow(pred_img)

    plt.show()
    if epoch % 10 == 0:
        fig.savefig(f'{path}{type1}_Epochs_{epoch}.png', bbox_inches = 'tight')
    pass

def test_show_img(image_num, img, label, pred, path, num_class, input_path, img_size):
    # 轉換 PyTorch 張量為 NumPy 數組
    input_img = img.cpu().numpy().transpose((1, 2, 0))
    target_img = color_set(label.cpu().numpy(), num_class)
    pred_img = color_set(pred.cpu().numpy(), num_class)
    or_img, idex = load_image(f"{input_path}JPEGImages\\test\\", image_num, img_size)
    # 顯示圖像和分割圖
    fig, ax = plt.subplots(1, 3, figsize = (10, 4))
    fig.suptitle(f'image {image_num + 1} # {idex}')
    fig.tight_layout(h_pad = 2)
    ax[0].set_title('input image')
    ax[0].axis('off')
    ax[0].imshow(or_img)
    ax[1].set_title('label image')
    ax[1].axis('off')
    ax[1].imshow(target_img)
    ax[2].set_title('prediction image')
    ax[2].axis('off')
    ax[2].imshow(pred_img)
    plt.show()
    # save test images
    fig.savefig(f'{path}num_{image_num + 1}.png', bbox_inches = 'tight',)

def loss_function(input, target, loss_name = 'cross_entropy', weight = None, size_avg = True, gamma = 2):
    n, c, h, w = input.size()
    nt, ht, wt = target.size()
    # Handle inconsistent size between input and target
    if h != ht and w != wt:  # upsample labels
        input = F.interpolate(input, size = (
            ht, wt), mode = "bilinear", align_corners = True)
    input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(-1)
    if loss_name == 'cross_entropy': 
        loss = F.cross_entropy(
        input, target, weight = weight, size_average = size_avg, ignore_index = 250
        )
    elif loss_name == 'focal_loss':
        cro_loss = nn.CrossEntropyLoss(weight = weight)(input, target)
        pt = torch.exp(-cro_loss)
        loss = (1 - pt) ** gamma * cro_loss
    return loss
def cross_entropy2d(input, target, weight = None, size_average = True):
    n, c, h, w = input.size()
    nt, ht, wt = target.size()
    # print(f"input.shape: {input.shape}, target.shape: {target.shape}")

    # Handle inconsistent size between input and target
    if h != ht and w != wt:  # upsample labels
        input = F.interpolate(input, size = (
            ht, wt), mode = "bilinear", align_corners = True)

    input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(-1)
    loss = F.cross_entropy(
        input, target, weight = weight, size_average = size_average, ignore_index = 250
    )
    return loss

'''We have used skelarn libraries to calculate Accuracy and Jaccard Score'''

def get_metrics(gt_label, pred_label):
    # Accuracy Score
    acc = skm.accuracy_score(gt_label, pred_label, normalize = True)

    # Jaccard Score/IoU
    js = skm.jaccard_score(gt_label, pred_label, average = 'micro')

    result_gm_sh = [acc, js]
    return (result_gm_sh)


'''
Calculation of confusion matrix from :
https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/metrics.py

Added modifications to calculate 3 evaluation metrics - 
Specificity, Senstivity, F1 Score
'''


class runningScore(object):
    def __init__(self, n_classes, per):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))
        self.per = per

    def _fast_hist(self, label_true, label_pred):
        mask = (label_true >= 0) & (label_true < self.n_classes)
        hist = np.bincount(
            self.n_classes * label_true[mask].astype(int) + 
            label_pred[mask], minlength = self.n_classes ** 2).reshape(self.n_classes, self.n_classes)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            # print(f"lt.shape: {lt.shape}, lp.shape: {lp.shape}")
            self.confusion_matrix += self._fast_hist(
                lt.flatten(), lp.flatten())

    def get_scores(self):
        # confusion matrix
        hist = self.confusion_matrix
        
        #              T
        #         0    1    2
        #    0   TP   FP   FP
        #  P 1   FN   TN   TN       This is wrt to class 0
        #    2   FN   TN   TN

        #         0    1    2
        #    0   TP   FP   FP
        #  P 1   FP   TP   FP       This is wrt prediction classes; AXIS = 1
        #    2   FP   FP   TP

        #         0    1    2
        #    0   TP   FN   FN
        #  P 1   FN   TP   FN       This is wrt true classes; AXIS = 0
        #    2   FN   FN   TP

        TP = np.diag(hist)
        TN = hist.sum() - hist.sum(axis = 1) - hist.sum(axis = 0) + np.diag(hist)
        FP = hist.sum(axis = 1) - TP
        FN = hist.sum(axis = 0) - TP
        # print(hist.sum(axis = 1))
        # 1e-6 was added to prevent corner cases where denominator = 0

        # Specificity: TN / (TN + FP)
        specif_ = (TN) / (TN + FP + 1e-6)
        specif = np.nanmean(specif_)

        # Senstivity/Recall: TP / (TP + FN)
        sensti_ = (TP) / (TP + FN + 1e-6)
        sensti = np.nanmean(sensti_)
        
        # Precision: TP / (TP + FP)
        prec_ = (TP) / (TP + FP + 1e-6)
        prec = np.nanmean(prec_)
        
        # Pixel Accuracy: (TP + TN) / (TP + TN + FP + FN)
        pixel_acc_ = (TP + TN) / (TP + TN + FP + FN + 1e-6)
        pixel_acc = np.nanmean(pixel_acc_)

        # IoU: 
        iou = np.diag(hist) / (hist.sum(axis = 1) + hist.sum(axis = 0) - np.diag(hist))
        # MIoU:
        Miou = np.nanmean(iou)
        # FWIoU:
        freq = hist.sum(axis = 1) / hist.sum()
        FWiou = (freq[freq > 0] * iou[freq > 0]).sum()
        # F1 = 2 * Precision * Recall / Precision + Recall
        f1 = (2 * prec * sensti) / (prec + sensti + 1e-6)
        f1_ = (2 * prec_ * sensti_) / (prec_ + sensti_ + 1e-6)
        if self.per:
            # return ({
            #         "Specificity": specif,
            #         "Senstivity": sensti,
            #         "F1": f1_*100,
            #         "Precision": prec,
            #         "Pixel_Accuracy": pixel_acc_*100,
            #         "IoU": iou*100,
            #         "MIoU": Miou,
            #         "FWIoU": FWiou,
            #         })
            return ({
                    "F1": f1_*100,
                    "Pixel_Accuracy": pixel_acc_*100,
                    "IoU": iou*100,
                    "MIoU": Miou,
                    "Precision": prec_*100,
                    })
        else:
            return ({
                    "Specificity": specif,
                    "Senstivity": sensti,
                    "F1": f1,
                    "Precision": prec,
                    "Pixel_Accuracy": pixel_acc,
                    "IoU": iou,
                    "MIoU": Miou,
                    "FWIoU": FWiou,
                    })

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))
def train(train_loader, model, optimizer, epoch_i, epoch_total, scheduler, ImageLabel):
    train_loop = tqdm(train_loader)
    # List to cumulate loss during iterations
    loss_list, train_total, train_correct, train_loss = [], 0, 0, 0
    model.train()
    for index, (images, labels) in enumerate(train_loop, start = 1):

        # we used model.eval() below. This is to bring model back to training mood.

        images = images.to(device, non_blocking = True)
        labels = labels.to(device, non_blocking = True)
        # print(images.shape)
        # print(labels.shape)
        # Model Prediction
        # print(images.shape)
        pred = model(images)
        # Loss Calculation
        # loss = cross_entropy2d(pred, labels)
        loss = loss_function(pred, labels, loss_name = 'cross_entropy')
        loss_list.append(loss)
        if ImageLabel:
            _, preds = torch.max(pred, dim = 1)
        else:
            _, preds = torch.max(pred.data, dim = 1)
            train_total += labels.size(0)
            train_correct += (preds == labels).sum()
            train_loss += loss.item()

        # optimiser
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        train_loop.set_description(f'Epoch [{epoch_i:0>4d}/{epoch_total:0>4d}]')
        train_loop.set_postfix(crossentropy_Loss = loss.item())

    return images[0], labels[0], preds[0], (loss_list)

def validate(val_loader, model, epoch_i, epoch_total, num_class, per:bool = False):
    '''
        per: bool, if True, print the result of score in separate classes, else print the average score of all classes
    '''
    val_loop = tqdm(enumerate(val_loader), total = len(val_loader))

    # tldr: to make layers behave differently during inference (vs training)
    model.eval()

    # enable calculation of confusion matrix for n_classes = 19
    running_metrics_val = runningScore(n_classes = num_class, per = per)
    # empty list to add Accuracy and Jaccard Score Calculations
    acc_sh = []
    js_sh = []

    with torch.no_grad():
        for image_num, (val_images, val_labels) in val_loop:
            # print(f"label: {val_labels.shape}")
            val_images = val_images.to(device, non_blocking = True)
            val_labels = val_labels.to(device, non_blocking = True)

            # Model prediction
            val_pred = model(val_images)
            _, preds = torch.max(val_pred, dim = 1)

            # Coverting val_pred from (1, 19, 512, 1024) to (1, 512, 1024)
            # considering predictions with highest scores for each pixel among 19 classes
            pred = val_pred.data.max(1)[1].cpu().numpy()
            gt = val_labels.data.cpu().numpy()
            # print(pred.shape, gt.shape)
            # Updating Mertics
            running_metrics_val.update(gt, pred)
            sh_metrics = get_metrics(gt.flatten(), pred.flatten())
            acc_sh.append(sh_metrics[0])
            js_sh.append(sh_metrics[1])
            val_loop.set_description(
                f'Epoch [{epoch_i:0>4d}/{epoch_total:0>4d}]')
            val_loop.set_postfix(
                Accuracy = sh_metrics[0], Jaccard_Score = sh_metrics[1])

    score = running_metrics_val.get_scores()
    running_metrics_val.reset()

    acc_s = sum(acc_sh)/len(acc_sh)
    js_s = sum(js_sh)/len(js_sh)
    score["acc"] = acc_s
    score["js"] = js_s
    # print(f"score: {score}")

    return val_images[0], val_labels[0], preds[0], (score)

def test(test_loader, model, path, num_class, input_path, img_size, per:bool = True):
    '''
        per: bool, if True, print the result of score in separate classes, else print the average score of all classes
    '''
    F1_total, Pixel_Accuracy_total, MIoU_total = 0, 0, 0

    test_loop = tqdm(enumerate(test_loader), total = len(test_loader))

    # tldr: to make layers behave differently during inference (vs training)
    model.eval()

    # enable calculation of confusion matrix for n_classes = 19
    running_metrics_val = runningScore(n_classes = num_class, per = per)
    
    # empty list to add Accuracy and Jaccard Score Calculations
    acc_sh = []
    js_sh = []

    with torch.no_grad():
        for image_num, (test_images, test_labels) in test_loop:
            time_start = time.time()
            test_images = test_images.to(device)
            test_labels = test_labels.to(device)

            # Model prediction
            test_pred = model(test_images)
            _, preds = torch.max(test_pred, dim = 1)

            # Coverting val_pred from (1, 19, 512, 1024) to (1, 512, 1024)
            # considering predictions with highest scores for each pixel among 19 classes
            pred = test_pred.data.max(1)[1].cpu().numpy()
            gt = test_labels.data.cpu().numpy()

            # calculation MIoU
            # Updating Mertics
            running_metrics_val.update(gt, pred)
            sh_metrics = get_metrics(gt.flatten(), pred.flatten())
            acc_sh.append(sh_metrics[0])
            js_sh.append(sh_metrics[1])
            test_loop.set_postfix(
                Accuracy=sh_metrics[0], Jaccard_Score = sh_metrics[1])
            # 將 batch 中的第一個樣本取出
            input_img = test_images[0]
            target_img = test_labels[0]
            pred_img = preds[0]
            time_end = time.time()
            test_show_img(image_num, input_img, target_img, pred_img, path, num_class, input_path, img_size)
            print(f"Totally cost: {time_end - time_start}")
            score = running_metrics_val.get_scores()
            F1_total += score["F1"]
            Pixel_Accuracy_total += score["Pixel_Accuracy"]
            MIoU_total += score["MIoU"]
            print(score)
            
            running_metrics_val.reset()

    # score = running_metrics_val.get_scores()
    # print(score)
    # running_metrics_val.reset()

    acc_s = sum(acc_sh)/len(acc_sh)
    js_s = sum(js_sh)/len(js_sh)
    score["acc"] = acc_s
    score["js"] = js_s
    print(f"F1_total: {F1_total/len(test_loader)}, mPA: {Pixel_Accuracy_total/len(test_loader)}, total MIoU: {MIoU_total/len(test_loader)}")

    return print(f"fin: {score}")

def train_cls (train_loader, model, optimizer, loss, epoch_i, epoch_total, scheduler):
    train_loop = tqdm(train_loader)
    train_correct, train_total, lo = 0, 0, 0.0
    model.train()
    for index, (images, labels) in enumerate(train_loop):
        images = images.to(device, non_blocking = True)
        labels = torch.squeeze(labels)
        labels = labels.to(device, non_blocking = True)
        # Model Prediction
        pred = model(images.float())
        train_loss = loss(pred, labels)
        scheduler.step()
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        
        # calculate the training dataset accuracy
        _, preds = torch.max(pred.data, dim = 1)
        train_total += labels.size(0)
        train_correct += (preds == labels).sum().item()
        train_loop.set_description(f'Epoch [{epoch_i:0>4d}/{epoch_total:0>4d}]')
        train_loop.set_postfix(crossentropy_Loss = train_loss.item(), accuracy = 100 * train_correct / train_total)
    lo += train_loss.item()
    acc = 100 * train_correct / train_total
    return images[0], labels, pred, lo, acc

def val_cls(val_loader, model, loss, epoch_i, epoch_total, optimizer):
    val_loop = tqdm(enumerate(val_loader), total = len(val_loader))
    val_correct, val_total, lo = 0, 0, 0
    model.eval()
    # running_metrics_val = runningScore(n_classes = num_class)
    # acc_sh, js_sh = [], []
    with torch.no_grad():
        for index, (images, labels) in val_loop:
            images = images.to(device, non_blocking = True)
            labels = torch.squeeze(labels)
            labels = labels.to(device, non_blocking = True)
            optimizer.zero_grad()
            # Model Prediction
            pred = model(images.float())
            val_loss = loss(pred, labels)
            # val_CN += val_loss.item()
            _, preds = torch.max(pred.data, dim = 1)
            val_total += labels.size(0)
            val_correct += (preds == labels).sum().item()
            val_loop.set_description(f'Epoch [{epoch_i:0>4d}/{epoch_total:0>4d}]')
            val_loop.set_postfix(val_crossentropy_Loss = val_loss.item(), accuracy = 100 * val_correct / val_total)
    lo += val_loss.item()
    acc = 100 * val_correct / val_total
    return images[0], labels, pred, lo, acc

def test_cls(model, test_loader, loss):
    y_pred, y_true, lo = [], [], 0
    model.eval()

    with torch.no_grad():
        total_time = []
        for i, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            # labels = torch.squeeze(labels)
            labels = labels.to(device)
            time_start = time.time()
            pred = model(images)
            # print(f"pred: {pred.shape}, {pred}, labels: {labels.shape}, {labels}")
            _, preds = torch.max(pred, 1)  # preds是預測結果
            test_loss = loss(pred, labels)
            y_pred.extend(preds.view(-1).detach().cpu().numpy())
            y_true.extend(labels.view(-1).detach().cpu().numpy())
            time_end = time.time()
            total_time.append(time_end - time_start)
    lo += test_loss.item()
    # print(f"loss: {lo}")
    cf_matrix = confusion_matrix(y_true, y_pred)
    # print(cf_matrix)
    classification_reports = classification_report(y_true = y_true,
                                                    y_pred = y_pred,
                                                    output_dict = True)
    mean_time = sum(total_time) / len(total_time)
    mean_fps = sum([n * 0.5 for n in total_time]) / len(total_time)
    print(f"mean used time: {mean_time}s")
    print(f"mean fps: {mean_fps * 1000}")
    return classification_reports
# return results of Test prediction
def test_(model, dataloader):
    y_pred = []
    model.eval()
    with torch.no_grad():
        total_time = []
        for images in dataloader:
            images = images.to(device)
            time_start = time.time()
            pred = model(images)
            _, preds = torch.max(pred, 1)
            y_pred.extend(preds.view(-1).detach().cpu().numpy())
            time_end = time.time()
            total_time.append(time_end - time_start)
    mean_time = sum(total_time) / len(total_time)
    mean_fps = sum([n * 0.5 for n in total_time]) / len(total_time)
    print(f"mean used time: {mean_time}s")
    print(f"mean fps: {mean_fps * 1000}")
    return y_pred
# Grad CAM
def img_set(path, size):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size[0], size[1]))
    img_ = np.float32(img) / 255.0
    img_tens = transforms.ToTensor()(img)
    img_tens = img_tens.unsqueeze(0)
    return img_tens, img_

def reshape_transform(tensor, H = 8, W = 8):
    # print(tensor.shape)
    _, H, W, _ = tensor.shape
    # result = tensor.reshape(tensor.size(0), H, W, tensor.size(1))
    result = tensor.reshape(tensor.size(0), H, W, tensor.size(3))
    # print(result.shape)

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def att_map(model, size, class_num, att_stage = [4], in_path = '', out_path = '', img_name = [], dataset_name = ''):
    input_tensor, img = img_set(in_path, size)
    input_tensor = input_tensor.to(device)
    layer_ = []
    # print(model.encoder.blocks4.abc)
    # layer_.append(model.encoder.blocks4.)
    # print(len(layer_))
    if 1 in att_stage:
        layer_.append(model.encoder.blocks1.dense[-1])
    elif 2 in att_stage:
        layer_.append(model.encoder.blocks2.dense[-1])
    elif 3 in att_stage:
        layer_.append(model.encoder.blocks3.dense[-1])
    elif 4 in att_stage:
        layer_.append(model.encoder.blocks4.dense[-1])
    cam = GradCAM(model = model, target_layers = layer_, use_cuda = True, reshape_transform = reshape_transform)
    targets = [ClassifierOutputTarget(class_num - 1)]
    grayscale_cam = cam(input_tensor = input_tensor, targets = targets)
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(img = img, mask = grayscale_cam)
    CreateNF(f"{out_path}GradCAM_stage{att_stage}_{dataset_name}\\")
    CreateNF(f"{out_path}GradCAM_stage{att_stage}_{dataset_name}\\{img_name[0]}\\")
    cv2.imwrite(f"{out_path}GradCAM_stage{att_stage}_{dataset_name}\\{img_name[0]}\\{img_name[1]}", visualization)

# Self-attention map
def save_fig_(attn, path, name_path, size):
    # fig_s = 8
    # dpi = int(size[0] / fig_s)
    # fig = plt.figure(figsize = (fig_s, fig_s), dpi = dpi)
    # plt.imshow(attn)
    # plt.axis('off')
    # plt.savefig(path, pad_inches=0, dpi = dpi)
    # plt.show()
    dpi = 100
    figsize = (size[0] / dpi, size[1] / dpi)
    tem_path = f"{path}_tem.jpg"
    fig = plt.figure(figsize = figsize, dpi = dpi)
    plt.imshow(attn)
    plt.axis('off')
    # Temporarily save to a buffer
    plt.savefig(tem_path, bbox_inches='tight', pad_inches = 0, dpi = dpi)
    # plt.savefig(tem_path, dpi = dpi)
    plt.close(fig)  # Close the figure to free memory

    # Open the image and remove white edges
    img = Image.open(tem_path)
    # img = img.crop(img.getbbox())  # This removes the border using PIL
    
    # Resize back to desired output size if necessary
    img = img.resize(size, Image.ANTIALIAS)
    save_path = f"{path}{name_path}"
    img.save(save_path)
def grid_show(to_shows, cols: int, img_name: list, layer, path: str, stage_nm: str = '', dataset_name: str = '', size = (64, 64)):
    '''
        to_shows: list, the list of images to show
        cols: int, the number of columns
        img_name: list, the name of the image
        path: str, the path to save the image
    '''
    rows = (len(to_shows)) // cols
    # print(f"rows: {rows}, cols: {cols}")
    it = iter(to_shows)
    fig, axs = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
    fig.suptitle(f'=== image #{img_name[:]} ··· layer #{layer} ===')
    if rows == 1 and cols == 1:
        axs = [axs]
    elif rows == 1 or cols == 1:
        axs = axs.flatten()  # Make axs a flat list for single row or single column
    else:
        axs = axs.reshape(-1)  # Make axs a flat list for multiple rows and columns

    for ax, to_show in zip(axs, to_shows):
        image, title = to_show
        im = ax.imshow(image)
        ax.set_title(title)
        ax.set_yticks([])
        ax.set_xticks([])
        # Create colorbar for the current axis
        fig.colorbar(im, ax=ax)
    final_image = image
    # Adjust layout
    plt.tight_layout()
    plt.show()
    # fig.savefig(f'{path}{img_name[:-4]}_layer{layer}_heads.png', bbox_inches = 'tight',)
    
    CreateNF(f"{path}SelfAttn_stage{stage_nm}_{dataset_name}\\")
    CreateNF(f"{path}SelfAttn_stage{stage_nm}_{dataset_name}\\{img_name[0]}\\")
    CreateNF(f"{path}SelfAttn_stage{stage_nm}_{dataset_name}\\{img_name[0]}\\{img_name[1][:-4]}\\")
    save_fig_(final_image, f"{path}SelfAttn_stage{stage_nm}_{dataset_name}\\", f"{img_name[0]}\\{img_name[1][:-4]}\\{layer}-{img_name[1]}", size)
    # cv2.imwrite(f"{path}SelfAttn_stage{stage_nm}_{dataset_name}\\{img_name[0]}\\{layer}-{img_name[1]}", final_image)
def visualize_head(att_map):
    ax = plt.gca()
    # Plot the heatmap
    im = ax.imshow(att_map)
    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    plt.show()
def visualize_heads(att_map, img_name, layer, path, stage_nm = '', dataset_name = ''):
    to_shows, mean = [], True
    # print(att_map.shape)
    size = (att_map.shape[2], att_map.shape[3])
    if att_map.shape[1] == 1:
        mean = False
        att_map = att_map.reshape(1, att_map.shape[2], att_map.shape[3])
        cols = 1
    else: att_map = att_map.squeeze()
    for i in range(att_map.shape[0]):
        to_shows.append((att_map[i], f'Head {i+1}'))
    if mean:
        average_att_map = att_map.mean(axis=0)
        to_shows.append((average_att_map, 'Head Average'))
        cols = int(len(to_shows) / 2)
    
    grid_show(to_shows, cols=cols, img_name = img_name, layer = layer, path = path, stage_nm = stage_nm, dataset_name = dataset_name, size = size)
    
def cls_padding(image, mask, cls_weight, grid_size):
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)
        
    image = np.array(image)

    H, W = image.shape[:2]
    delta_H = int(H/grid_size[0])
    delta_W = int(W/grid_size[1])
    padding_w = delta_W
    padding_h = H
    padding = np.ones_like(image) * 255
    padding = padding[:padding_h, :padding_w]
    
    padded_image = np.hstack((padding,image))
    padded_image = Image.fromarray(padded_image)
    # draw = ImageDraw.Draw(padded_image)
    # draw.text((int(delta_W/4),int(delta_H/4)),'CLS', fill=(0,0,0)) # PIL.Image.size = (W,H) not (H,W)

    mask = mask / max(np.max(mask),cls_weight)
    cls_weight = cls_weight / max(np.max(mask),cls_weight)
    
    if len(padding.shape) == 3:
        padding = padding[:,:,0]
        padding[:,:] = np.min(mask)
    mask_to_pad = np.ones((1,1)) * cls_weight
    mask_to_pad = Image.fromarray(mask_to_pad)
    mask_to_pad = mask_to_pad.resize((delta_W, delta_H))
    mask_to_pad = np.array(mask_to_pad)

    padding[:delta_H,  :delta_W] = mask_to_pad
    padded_mask = np.hstack((padding, mask))
    padded_mask = padded_mask
    
    meta_mask = np.zeros((padded_mask.shape[0], padded_mask.shape[1],4))
    meta_mask[delta_H:,0: delta_W, :] = 1 
    
    return padded_image, padded_mask, meta_mask

def visualize_grid_to_grid(att_map, grid_index, head_num, image, img_name, path, ly, grid_size=32, alpha=0.6):
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)
    
    H,W = att_map.shape
    # print(f"att_map: {att_map.shape}")
    with_cls_token = False
      
    grid_image = highlight_grid(image, [grid_index], grid_size)
    print(f"att_map: {att_map[grid_index].shape}")
    mask = att_map[grid_index].reshape(grid_size[0], grid_size[1])
    mask = Image.fromarray(mask).resize((image.size))
    
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    fig.suptitle(f'image #{img_name} head #{head_num} grid {grid_index} attention map')
    fig.tight_layout(h_pad = 2)
    ax[0].imshow(grid_image, cmap='gray')
    ax[0].axis('off')
    
    ax[1].imshow(grid_image, cmap='gray')
    ax[1].imshow(mask/np.max(mask), alpha=alpha, cmap='rainbow')
    ax[1].axis('off')
    plt.show()
    # save the image
    # fig.savefig(f'{path}grid2grid{img_name[:-4]}_layer{ly}_{head_num}_{grid_index}.png', bbox_inches = 'tight',)
# check grid_size
def highlight_grid(image, grid_indexes, grid_size=32):
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)
    # print(image.size)
    W, H = image.size
    h = H / grid_size[0]
    w = W / grid_size[1]
    image = image.copy()
    for grid_index in grid_indexes:
        x, y = np.unravel_index(grid_index, (grid_size[0], grid_size[1]))
        a= ImageDraw.ImageDraw(image)
        # a = image
        a.rectangle([(y*w,x*h),(y*w+w,x*h+h)],fill =None,outline ='red',width =2)
    return image
def SelfAttn_(attention_maps, stage_num, output_path, img_name, num_img, each_layer = 18, dataset_name = ''):
    # num_img = 0 ->range(0-18), 1 -> range(18-36), 2 -> range(36-54)
    if 1 and 2 and 3 and 4 in stage_num:
        stage_nm = 'all_ST'
        st, ed = num_img * each_layer, (num_img + 1) * each_layer
        for lyr in range(st, ed):
            visualize_heads(attention_maps[lyr], img_name, lyr, output_path, stage_nm, dataset_name)
            clear_output(wait = True)
def avg_SelfAttn_ST(path, layer_start, layer_num, stage_num, dataset_num):
    att = []
    for _ in range(layer_num):
        total_path = os.listdir(f"{path}")
        sm_stage = total_path[layer_start:layer_num]
    print(sm_stage)
    end_ = sm_stage[0].split('.')[-1]
    print(end_)
    # H, W, C = att[n].shape
    for num in range(layer_num):
        att.append(np.array(Image.open(f"{path}{sm_stage[num]}").convert('RGB')))
    # att.append(np.array(Image.open(f"{path}{sm_stage[num]}").convert('RGB')) for num in range(layer_num))
    # print(att[0].shape)
    final_att = np.mean(np.stack(att), axis = 0)
    CreateNF(f"{path}stage_\\")
    # print(f"{path}stage_\\st_{stage_num}_{dataset_num}{end_}")
    cv2.imwrite(f"{path}stage_\\st_{stage_num}_{dataset_num}.{end_}", final_att.astype(np.uint8))
# def Fus_Attn():
    