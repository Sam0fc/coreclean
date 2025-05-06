import torch
import seg_utils
import segmentation
import matplotlib.pyplot as plt
import tqdm

def train_model(model, loader, optimiser):
    segmentation.to_device(model.train())
    cel = True 
    if cel:
        criterion = torch.nn.CrossEntropyLoss()
    else:
        criterion = seg_utils.IoUloss(softmax=True)

    running_loss = 0.0
    running_samples = 0 
    epoch_loss = []
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

    plt.plot(epoch_loss)
    plt.show()
    print('Trained on {} samples'.format(running_samples))
    print('Average loss: {:.4f}'.format(running_loss / running_samples))


if __name__ == '__main__':
    full_dataset = seg_utils.CustomImageDataset(
        image_dir="./Dataset/labelled_patches",
        transform=None,
        color_transform=None
    )
    print(len(full_dataset))

    train_data, test_data = torch.utils.data.random_split(full_dataset, [int(len(full_dataset)*0.8), int(len(full_dataset)) - int(len(full_dataset)*0.8)])
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True)

    model = segmentation.to_device(segmentation.SegNet(kernel_size=3))

    optimiser = torch.optim.Adam(model.parameters(), lr=0.001)
    

    train_model(model, train_loader, optimiser)
    print( seg_utils.calc_iou(model, test_loader))
    torch.save(model.state_dict(), 'segmentation_model.pth')

