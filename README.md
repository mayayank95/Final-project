# Final-project

This repo try to solve the Kaggle competition in 2022, [UW-Madison GI Tract Image Segmentation](https://www.kaggle.com/competitions/uw-madison-gi-tract-image-segmentation/overview). 

To get the data for the different notebooks, we use the Kaggle API to download the data for this competition. 

Create an account on Kaggle. Once logged in, go to the "Rules" page for the W-Madison GI Tract Image Segmentation competition and accept the rules by clicking on the button. (required for downloading the dataset)
The following two questions assume you are on Google Colab. If running locally, instead follow the instructions on Kaggle's API documentation and create the required folders yourself.

On Kaggle, go to "Account > Create API Token" on the site, to download the API Token file kaggle.json. Upload this file to the Google Colab runtime, move it to the directory ~/.kaggle and set its permissions using the code below.


masks_paths.csv - Table with the paths of the images and the corresponding segmentation masks in RLE-format.

At the beginning of each notebook, change parentDir to the directory that includes all the files in your drive. 

FinalProject_Main_notebook.ipynb - The notebook that includes the training and evaluation of the model with all the necessary steps before.
