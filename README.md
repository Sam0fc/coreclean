# CoreClean
CoreClean is a Python-based tool designed for preprocessing and cleaning sediment core data with CNNs. 

## Features


## Installation
Clone the repository and install the required dependencies:
```bash
git clone https://github.com/yourusername/CoreClean.git
```
### Dependencies 
- matplotlib
- tqdm
- pandas
- numpy
- cv2
- tiffile
- torch
- torchvision
- natsort

## Usage
Put coreclean in a directory, and run this code. When prompted, put your model in model.pth and image in TO_PROCESS/ 
```python
import coreclean

print('coreclean')
coreclean.setup_directories()
print('enter image and model')
input()
print('removing background')
coreclean.remove_background()
print('making patches')
coreclean.make_patches()
print('processing patches')
coreclean.process_patches()
print('stitching patches')
coreclean.stitch_patches()
print('done')
# Example usage
```
Model and image for testing are available in the main directory, you'll have to move them to the right place.

## Documentation
Documentation is an active WIP; Most code has docstrings.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact
For questions or feedback, please contact [sh2154@cam.ac.uk](mailto:sh2154@cam.ac.uk).