from net import Decoder
import torch
import matplotlib.pyplot as plt

input_data = torch.tensor([48.2976, -2.0373, -29.1018, 12.2312])
# input_data = torch.tensor([23.9500, -11.1840, 36.5084, -33.9963])
# input_data = torch.tensor([9.9927, -0.2395, -1.1840, -2.4094])
# input_data = torch.tensor([20.2314, -10.3747, 11.7729, -8.3415])
# input_data = torch.tensor([6.8456, 5.0325, -4.7677, -0.9231])
# input_data = torch.tensor([12.7765, -8.3813, 9.4535, -3.7266])
# input_data = torch.tensor([5.8932, 0.2757, -5.0638, 6.0578])
# input_data = torch.tensor([7.7216, 11.7913, 1.8647, -13.9440])
# input_data = torch.tensor([8.0073, -0.2398, 7.3321, -9.5089])
# input_data = torch.tensor([2.0922, 4.1933, -0.1578, -3.3220])
# input_data = (torch.randn(4))*50

print(input_data)
decoder = Decoder().eval()
model_dict = decoder.state_dict()
pretrained_dict = torch.load('model/net.pth')
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)
decoder.load_state_dict(model_dict)
with torch.no_grad():
    output = decoder(input_data)
    plt.imshow(output[0], cmap='gray')
    plt.show()
