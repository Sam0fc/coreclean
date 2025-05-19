import torch
import torchvision
from . import seg_utils
from . import segmentation
import matplotlib.pyplot as plt
import tqdm
import csv


def train_model(model, loader, optimiser,epochs=1,test_loader=None, start_epoch=0):
    cel = True 
    if cel:
        criterion = torch.nn.CrossEntropyLoss()
    else:
        criterion = seg_utils.IoUloss(softmax=True)

    epoch_loss = []
    for i, epoch in enumerate(range(epochs)):
        segmentation.to_device(model.train())

        running_loss = 0.0
        running_samples = 0
        for batch_idx, (batch, targets) in tqdm.tqdm(enumerate(loader)):

            optimiser.zero_grad()
            batch = segmentation.to_device(batch)
            targets = segmentation.to_device(targets)
            
            outputs = model(batch)
            
            if cel:
                targets = targets.long()
            
            loss = criterion(outputs, targets)
            #print('Batch {}: Loss: {:.4f}'.format(batch_idx, loss.item()))
            loss.backward()
            optimiser.step()
            
            running_loss += loss.item()
            running_samples += batch.size(0)
            epoch_loss.append(loss.item())

            #print(seg_utils.calc_iou(model, test_loader))
        torch.save(model.state_dict(), f'bg{epoch + start_epoch}.pth')

        print('Trained on {} samples'.format(running_samples))
        print('Average loss: {:.4f}'.format(running_loss / running_samples))
        
        segmentation.to_device(model.eval())
      

        iou = seg_utils.calc_iou(model, test_loader)
        print(iou)
        with open('iou_scores.csv', 'a', newline='') as ioufile:
            iouwriter = csv.writer(ioufile)
            iouwriter.writerow([epoch + start_epoch, iou])
        with open('epoch_loss.csv', 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['Epoch', 'Loss','IOU'])
            for epoch_idx, loss in enumerate(epoch_loss):
                csvwriter.writerow([epoch_idx + 1 + start_epoch, loss])

    plt.plot(epoch_loss)
    plt.show()



if __name__ == '__main__':
    full_dataset = seg_utils.CustomImageDataset(
        image_dir="./coreclean/Dataset/crop_labelled_patches/",
        transform=None,  # No transform applied initially
    )
    print(len(full_dataset))
    iou_file = open('iou_scores.csv', 'w', newline='')
    iou_writer = csv.writer(iou_file)
    iou_writer.writerow(['Epoch', 'IoU'])
    iou_file.close()
    train_data, test_data = torch.utils.data.random_split(full_dataset, [int(len(full_dataset)*0.6), int(len(full_dataset)) - int(len(full_dataset)*0.6)])

   #Apply RandomFlip transform only to train_data without altering its size
    train_data = torch.utils.data.Subset(
        seg_utils.CustomImageDataset(
            image_dir="./coreclean/Dataset/crop_labelled_patches/",
            color_transform=None,
        ),
        train_data.indices
    )
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True)


    #model = segmentation.to_device(segmentation.FCN8s(n_class=2))
    #vgg16 = torchvision.models.vgg16(pretrained=True)
    #model.copy_params_from_vgg16(vgg16,copy_fc8=False) 
    model = segmentation.to_device(segmentation.SegNet(kernel_size=3))
    #model.load_state_dict(torch.load('./segnet6.pth'))

    optimiser = torch.optim.Adam(model.parameters(), lr=0.001)
    

    train_model(model, train_loader, optimiser,epochs=3,test_loader=test_loader,start_epoch=0)
    print(seg_utils.calc_iou(model, test_loader))
